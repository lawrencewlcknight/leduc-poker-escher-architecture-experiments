"""Forensic solver instrumentation and one-factor mechanism ablations."""

from __future__ import annotations

import time

import numpy as np
import torch

from adaptive_escher import AdaptiveLambdaController
from adaptive_escher.solver import (
    AdaptiveResidualPredictiveEscher,
    PersistentFrozenTargetQValueTrainer,
)
from vr_deep_cfr.variants import (
    VRPDCFRPlusQValueTrainer,
    VRPDCFRPlusRegretTrainer,
)

from .diagnostics import (
    ExactLeducOracle,
    ExactWeightedAverageStrategy,
    aggregate_estimator_rows,
    aggregate_q_rows,
    build_policy_table,
    exact_exploitability,
)


class ForensicLambdaController(AdaptiveLambdaController):
    """Expose the four estimator regimes without changing other mechanisms."""

    MODES = {"scheduled", "fixed_one", "fixed_zero", "residual_only"}

    def __init__(self, *args, mode: str, **kwargs):
        if mode not in self.MODES:
            raise ValueError(f"Unknown lambda mode {mode!r}")
        self.mode = mode
        super().__init__(*args, **kwargs)

    def schedule_floor(self, iteration: int) -> float:
        if self.mode == "fixed_one":
            return 1.0
        if self.mode in {"fixed_zero", "residual_only"}:
            return 0.0
        return super().schedule_floor(iteration)

    def value(self, player: int, action: int, iteration: int) -> float:
        if self.mode == "fixed_one":
            return 1.0
        if self.mode == "fixed_zero":
            return 0.0
        residual = float(self.residual_ema[player, action])
        uncertainty_lambda = residual / (residual + self.residual_scale)
        if self.mode == "residual_only":
            return float(np.clip(uncertainty_lambda, 0.0, 1.0))
        return float(
            np.clip(
                max(super().schedule_floor(iteration), uncertainty_lambda),
                0.0,
                1.0,
            )
        )


class NonPredictivePDCFRPlusRegretTrainer(VRPDCFRPlusRegretTrainer):
    """PDCFR+ cumulative update with the instantaneous predictor disabled."""

    def train_model(self, T):
        final_loss = None
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            final_loss = float(loss.item())
            if train_step % 100 == 0:
                self.logger.info(
                    f"[{train_step}/{self.train_steps}] non-predictive regret "
                    f"loss: {final_loss}"
                )
        self.target_model.load_state_dict(self.model.state_dict())
        # The immediate model remains at its zero-output initialisation. In
        # predictive regret matching the remaining scalar factor cancels on
        # normalisation, yielding cumulative-only regret matching.
        return final_loss

    def get_policy(self, state, T) -> np.ndarray:
        del T
        return self.regret_matching(self.get_regrets(state), state.legal_actions())


class ReinitializingForensicQValueTrainer(VRPDCFRPlusQValueTrainer):
    """Upstream reinitialised Q learner with a diagnostic version counter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_version = 0

    def train_model(self, T):
        loss = super().train_model(T)
        if loss is not None:
            self.target_version += 1
        return loss


class ForensicAdaptiveResidualPredictiveEscher(
    AdaptiveResidualPredictiveEscher
):
    """Experiment 3 solver with exact, evaluation-only Leduc diagnostics."""

    def __init__(
        self,
        *args,
        diagnostic_variant_id: str,
        lambda_mode: str = "scheduled",
        use_predictive_accumulator: bool = True,
        q_mode: str = "persistent",
        **kwargs,
    ):
        if q_mode not in {"persistent", "reinitialized"}:
            raise ValueError(f"Unknown Q mode {q_mode!r}")
        self.diagnostic_variant_id = str(diagnostic_variant_id)
        self.lambda_mode = str(lambda_mode)
        self.use_predictive_accumulator = bool(use_predictive_accumulator)
        self.q_mode = str(q_mode)
        super().__init__(*args, **kwargs)
        self.lambda_controller = ForensicLambdaController(
            self.num_players,
            self.action_size,
            mode=self.lambda_mode,
            lambda_start=self.lambda_start,
            schedule_half_life=self.lambda_schedule_half_life,
            schedule_power=self.lambda_schedule_power,
            residual_ema_decay=self.lambda_residual_ema_decay,
            residual_scale=self.lambda_residual_scale,
            initial_residual=self.lambda_initial_residual,
        )
        self.exact_average_strategy = ExactWeightedAverageStrategy(
            self.game, gamma=self.gamma
        )
        self.q_oracle_diagnostic_rows = []
        self.estimator_diagnostic_rows = []
        self.strategy_diagnostic_rows = []
        self._latest_predictor_error_preupdate = {
            player: np.nan for player in range(self.num_players)
        }
        self._latest_predictor_error_postupdate = {
            player: np.nan for player in range(self.num_players)
        }

    def init_regret_trainers(self):
        trainer_class = (
            VRPDCFRPlusRegretTrainer
            if self.use_predictive_accumulator
            else NonPredictivePDCFRPlusRegretTrainer
        )
        self.regret_trainers = [
            trainer_class(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.reinitialize_imm_regret_networks,
                self.use_regret_matching_argmax,
                self.device,
                self.alpha,
            )
            for _ in range(self.num_players)
        ]

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        if self.q_mode == "persistent":
            self.q_value_trainer = PersistentFrozenTargetQValueTrainer(
                history_size,
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.baseline_buffer_size,
                self.baseline_batch_size,
                self.baseline_network_train_steps,
                self.logger,
                self.regret_trainers,
                self.device,
                gradient_clip_norm=self.q_gradient_clip_norm,
            )
        else:
            self.q_value_trainer = ReinitializingForensicQValueTrainer(
                history_size,
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.baseline_buffer_size,
                self.baseline_batch_size,
                self.baseline_network_train_steps,
                self.logger,
                self.regret_trainers,
                self.device,
            )

    def predictive_strategy(self, state) -> np.ndarray:
        player = int(state.current_player())
        return self.regret_trainers[player].get_policy(
            state, max(1, self.num_iteration)
        )

    def nonpredictive_strategy(self, state) -> np.ndarray:
        player = int(state.current_player())
        trainer = self.regret_trainers[player]
        regrets = trainer.get_regrets(state)
        return trainer.regret_matching(regrets, state.legal_actions())

    def neural_average_strategy(self, state) -> np.ndarray:
        return self.ave_policy_trainer.action_probabilities(
            state, probs_as_dict=False
        )

    def _measure_predictor_error(self, player: int, *, postupdate: bool) -> None:
        trainer = self.regret_trainers[player]
        destination = (
            self._latest_predictor_error_postupdate
            if postupdate
            else self._latest_predictor_error_preupdate
        )
        if len(trainer.buffer) == 0:
            destination[player] = np.nan
            return
        infostates, targets, legal_mask, _ = trainer.buffer.sample(-1)
        with torch.no_grad():
            predictions = trainer.predict(trainer.imm_model, infostates, legal_mask)
            squared_error = torch.square(predictions - targets) * legal_mask
            denominator = torch.clamp(torch.sum(legal_mask), min=1.0)
            error = torch.sum(squared_error) / denominator
        destination[player] = float(error.item())

    def iteration(self):
        self.lambda_controller.reset_diagnostics()
        self._minimum_sample_probability = 1.0
        self.num_iteration += 1
        self.exact_average_strategy.observe_iteration(
            self.num_iteration, self.predictive_strategy
        )
        for player in range(self.num_players):
            self.collect_training_data(player)
            self._measure_predictor_error(player, postupdate=False)
            self.train_regret(player)
            self._measure_predictor_error(player, postupdate=True)
        for player in range(self.num_players):
            self.train_baseline(player)
        if self.num_iteration % self.evaluation_frequency == 0:
            self._run_checkpoint(checkpoint_kind="outer_iteration")

    def evaluate(
        self,
        *,
        checkpoint_kind="outer_iteration",
        checkpoint_target_nodes=None,
    ):
        diagnostic_start = time.perf_counter()
        checkpoint_index = len(self.checkpoint_rows)
        predictive_table = build_policy_table(self.game, self.predictive_strategy)
        nonpredictive_table = build_policy_table(
            self.game, self.nonpredictive_strategy
        )
        neural_average_table = build_policy_table(
            self.game, self.neural_average_strategy
        )
        exact_average_table = self.exact_average_strategy.table()

        predictive_exp = exact_exploitability(self.game, predictive_table)
        nonpredictive_exp = exact_exploitability(self.game, nonpredictive_table)
        neural_average_exp = exact_exploitability(self.game, neural_average_table)
        exact_average_exp = exact_exploitability(self.game, exact_average_table)

        oracle = ExactLeducOracle(self, predictive_table)
        q_rows = oracle.q_oracle_rows()
        estimator_rows = oracle.estimator_rows()
        common = {
            "variant_id": self.diagnostic_variant_id,
            "checkpoint_index": checkpoint_index,
            "iteration": int(self.num_iteration),
            "nodes_touched": float(self.nodes_touched),
            "checkpoint_kind": str(checkpoint_kind),
        }
        self.q_oracle_diagnostic_rows.extend(
            [{**common, **row} for row in q_rows]
        )
        self.estimator_diagnostic_rows.extend(
            [{**common, **row} for row in estimator_rows]
        )

        q_aggregate = aggregate_q_rows(q_rows)
        estimator_aggregate = aggregate_estimator_rows(estimator_rows)
        def mean_finite(values) -> float:
            errors = np.asarray(list(values), dtype=float)
            finite = errors[np.isfinite(errors)]
            return float(np.mean(finite)) if finite.size else np.nan

        predictor_preupdate_error = mean_finite(
            self._latest_predictor_error_preupdate.values()
        )
        predictor_postupdate_error = mean_finite(
            self._latest_predictor_error_postupdate.values()
        )
        strategy_row = {
            **common,
            "current_predictive_exploitability": predictive_exp,
            "current_nonpredictive_exploitability": nonpredictive_exp,
            "predictive_exploitability_improvement": (
                nonpredictive_exp - predictive_exp
            ),
            "exact_average_exploitability": exact_average_exp,
            "neural_average_exploitability": neural_average_exp,
            "average_policy_distillation_gap": (
                neural_average_exp - exact_average_exp
            ),
            "predictor_preupdate_mse": predictor_preupdate_error,
            "predictor_postupdate_mse": predictor_postupdate_error,
            **q_aggregate,
            **estimator_aggregate,
        }
        self.strategy_diagnostic_rows.append(strategy_row)
        for key, value in strategy_row.items():
            if key not in common:
                self.logger.record(key, value)
        for player, value in self._latest_predictor_error_preupdate.items():
            self.logger.record(f"predictor_preupdate_mse_player_{player}", value)
        for player, value in self._latest_predictor_error_postupdate.items():
            self.logger.record(f"predictor_postupdate_mse_player_{player}", value)
        self.logger.record(
            "forensic_diagnostic_wall_clock_seconds",
            time.perf_counter() - diagnostic_start,
        )
        return super().evaluate(
            checkpoint_kind=checkpoint_kind,
            checkpoint_target_nodes=checkpoint_target_nodes,
        )
