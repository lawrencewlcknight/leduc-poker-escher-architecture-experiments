"""Cross-fitted direct Monte Carlo control critics for unbiased neural CFR."""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import torch

from unbiased_escher.solver import (
    CrossFittedQMember,
    UnbiasedControlVariateEscher,
)


class MonteCarloControlBuffer:
    """Current-iteration `(history, action, return)` supervision."""

    def __init__(self, capacity: int, history_size: int, device: str):
        if capacity <= 0:
            raise ValueError("Monte Carlo buffer capacity must be positive")
        self.capacity = int(capacity)
        self.history_size = int(history_size)
        self.device = str(device)
        self.histories = np.zeros((capacity, history_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.cursor = 0
        self.size = 0
        self.seen_count = 0

    def clear(self) -> None:
        self.cursor = 0
        self.size = 0
        self.seen_count = 0

    def add(self, history, action: int, sampled_return_player_0: float) -> None:
        self.histories[self.cursor] = np.asarray(history, dtype=np.float32)
        self.actions[self.cursor] = int(action)
        self.returns[self.cursor] = float(sampled_return_player_0)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.seen_count += 1

    def sample(self, count: int):
        count = min(int(count), self.size)
        indices = random.sample(range(self.size), count)
        return (
            torch.as_tensor(
                self.histories[indices],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.as_tensor(
                self.actions[indices],
                dtype=torch.int64,
                device=self.device,
            ),
            torch.as_tensor(
                self.returns[indices],
                dtype=torch.float32,
                device=self.device,
            ),
        )

    def moments(self):
        if not self.size:
            return np.nan, np.nan, np.nan
        values = self.returns[: self.size].astype(np.float64)
        return (
            float(np.mean(values)),
            float(np.var(values)),
            float(np.max(np.abs(values))),
        )

    def __len__(self) -> int:
        return self.size


class MonteCarloControlCritic(CrossFittedQMember):
    """Persistent frozen-target critic fitted without TD bootstrapping."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = MonteCarloControlBuffer(
            self.buffer_size,
            self.input_size,
            self.device,
        )

    def add_monte_carlo_target(
        self,
        history,
        action: int,
        sampled_return_player_0: float,
    ) -> None:
        self.buffer.add(history, action, sampled_return_player_0)

    def train_model(self, iteration: int):
        del iteration
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return None
        if len(self.buffer) == 0 or self.train_steps <= 0:
            return None
        self.model.train()
        final_loss = None
        for train_step in range(self.train_steps):
            histories, actions, sampled_returns = self.buffer.sample(
                self.batch_size
            )
            predictions = self.model(histories).gather(
                1,
                actions.unsqueeze(1),
            ).squeeze(1)
            loss = self.loss_fn(predictions, sampled_returns)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )
            self.optimizer.step()
            final_loss = float(loss.item())
            if train_step % 100 == 0:
                self.logger.info(
                    f"train_step[{train_step}/{self.train_steps}]: "
                    f"direct Monte Carlo Q loss {final_loss}"
                )
        # Collection and inference never observe the partially fitted model.
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_version += 1
        return final_loss


class CrossFittedMonteCarloQEnsemble:
    """Disjoint current-iteration return folds with leave-one-fold-out use."""

    def __init__(
        self,
        *,
        ensemble_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers,
        learning_rate: float,
        total_buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger,
        regret_trainers,
        device: str,
        gradient_clip_norm: float,
    ):
        if ensemble_size < 2:
            raise ValueError("Monte Carlo cross-fitting requires at least two folds")
        self.ensemble_size = int(ensemble_size)
        capacity = max(1, int(total_buffer_size) // ensemble_size)
        self.members = [
            MonteCarloControlCritic(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                capacity,
                batch_size,
                train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
            )
            for _ in range(ensemble_size)
        ]
        self.active_fold = 0
        self.current_iteration = 0
        self.last_loss = None
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.stats: Dict[str, float] = {
            "target_count": 0.0,
            "target_sum": 0.0,
            "target_square_sum": 0.0,
            "target_abs_max": 0.0,
        }

    def begin_iteration(self, iteration: int) -> None:
        self.current_iteration = int(iteration)
        self._reset_diagnostics()
        for member in self.members:
            member.buffer.clear()

    def begin_trajectory(self, trajectory_id: int) -> int:
        self.active_fold = int(trajectory_id) % self.ensemble_size
        return self.active_fold

    def heldout_member_indices(self) -> List[int]:
        return [
            index for index in range(self.ensemble_size) if index != self.active_fold
        ]

    def predictions(self, state, player: int) -> np.ndarray:
        return np.stack(
            [
                self.members[index].get_baseline(state, player)
                for index in self.heldout_member_indices()
            ],
            axis=0,
        )

    def get_baseline_and_disagreement(self, state, player: int):
        predictions = self.predictions(state, player)
        return np.mean(predictions, axis=0), np.var(predictions, axis=0)

    def get_baseline(self, state, player: int) -> np.ndarray:
        values, _ = self.get_baseline_and_disagreement(state, player)
        return values

    def observe_control_return(
        self,
        *,
        state,
        player: int,
        action: int,
        sampled_return: float,
        iteration: int,
    ) -> None:
        if int(iteration) != self.current_iteration:
            raise RuntimeError("Monte Carlo target was observed in the wrong phase")
        # Q networks represent player-0 utility; inference flips the sign for
        # player 1 in PersistentFrozenTargetQValueTrainer.get_baseline.
        target_player_0 = (
            float(sampled_return) if int(player) == 0 else -float(sampled_return)
        )
        history = self.members[self.active_fold].get_history_tensor(state)
        self.members[self.active_fold].add_monte_carlo_target(
            history,
            action,
            target_player_0,
        )
        self.stats["target_count"] += 1.0
        self.stats["target_sum"] += target_player_0
        self.stats["target_square_sum"] += target_player_0 * target_player_0
        self.stats["target_abs_max"] = max(
            self.stats["target_abs_max"],
            abs(target_player_0),
        )

    def add_data(self, *args) -> None:
        # Experiment 6's DFS also emits one-step transition tuples. Direct MC
        # supervision deliberately discards them, eliminating TD bootstrapping.
        del args

    def train_model(self, iteration: int):
        losses = [member.train_model(iteration) for member in self.members]
        finite = [float(loss) for loss in losses if loss is not None]
        self.last_loss = float(np.mean(finite)) if finite else None
        return self.last_loss

    def fold_sizes(self) -> List[int]:
        return [len(member.buffer) for member in self.members]

    def fold_seen_counts(self) -> List[int]:
        return [member.buffer.seen_count for member in self.members]

    def diagnostics(self) -> Dict[str, float]:
        count = self.stats["target_count"]
        denominator = max(count, 1.0)
        mean = self.stats["target_sum"] / denominator
        variance = max(
            self.stats["target_square_sum"] / denominator - mean * mean,
            0.0,
        )
        return {
            "mc_control_loss": self.last_loss,
            "mc_target_count": count,
            "mc_target_mean": mean if count else np.nan,
            "mc_target_variance": variance if count else np.nan,
            "mc_target_abs_max": (
                self.stats["target_abs_max"] if count else np.nan
            ),
            "mc_target_version_min": min(
                member.target_version for member in self.members
            ),
            "mc_target_version_max": max(
                member.target_version for member in self.members
            ),
        }


class MonteCarloControlCriticEscher(UnbiasedControlVariateEscher):
    """Experiment 6 with frozen-phase, direct Monte Carlo control fitting."""

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = CrossFittedMonteCarloQEnsemble(
            ensemble_size=self.q_ensemble_size,
            history_size=history_size,
            state_size=self.infostate_size,
            action_size=self.action_size,
            network_layers=self.network_layers,
            learning_rate=self.learning_rate,
            total_buffer_size=self.baseline_buffer_size,
            batch_size=self.baseline_batch_size,
            train_steps=self.baseline_network_train_steps,
            logger=self.logger,
            regret_trainers=self.regret_trainers,
            device=self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
        )

    def iteration(self):
        """Collect both players under one frozen strategy, then fit all models."""

        self._reset_architecture_diagnostics()
        self.num_iteration += 1
        self.q_value_trainer.begin_iteration(self.num_iteration)
        for player in range(self.num_players):
            trainer = self.regret_trainers[player]
            if getattr(trainer, "predictor_enabled", False):
                trainer.set_prediction_gate(
                    0.0
                    if self.force_prediction_gate_zero
                    else self.gate_controller.value(player)
                )

        # No regret, Q, calibration, gate, or average-policy parameter changes
        # occur until both traversers have completed collection.
        holdout_errors = []
        for player in range(self.num_players):
            self.collect_training_data(player)
            holdout_errors.append(self._predictor_holdout_error(player))

        for player in range(self.num_players):
            self.train_regret(player)
        for player, (prediction_mse, zero_mse) in enumerate(holdout_errors):
            self.gate_controller.observe(player, prediction_mse, zero_mse)
            if self.force_prediction_gate_zero or not self.use_instantaneous_predictor:
                self.gate_controller.gates[player] = 0.0
            self.logger.record(f"predictor_holdout_mse_player_{player}", prediction_mse)
            self.logger.record(f"predictor_zero_mse_player_{player}", zero_mse)

        calibration_loss = (
            self.calibration_trainer.train_model()
            if self.calibration_trainer is not None
            else None
        )
        self.logger.record("calibration_loss", calibration_loss)
        q_loss = self.q_value_trainer.train_model(self.num_iteration)
        if q_loss is not None:
            self.logger.record("baseline_loss_0", q_loss)
            self.logger.record("baseline_loss_1", q_loss)
        if self.num_iteration % self.evaluation_frequency == 0:
            self._run_checkpoint(checkpoint_kind="outer_iteration")

    def evaluate(self, **kwargs):
        for key, value in self.q_value_trainer.diagnostics().items():
            self.logger.record(key, value)
        for fold, seen_count in enumerate(
            self.q_value_trainer.fold_seen_counts()
        ):
            self.logger.record(f"mc_fold_{fold}_seen_count", seen_count)
        return super().evaluate(**kwargs)

