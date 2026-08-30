"""Exact conditional moments of the implemented UCV estimator in Leduc."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import inspect
import random
from typing import Any, Dict, List, MutableMapping, Tuple

import numpy as np
import torch

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table,
)
from unbiased_escher.estimator import (
    control_variate_advantage,
    residual_adaptive_sampling_policy,
    variance_optimal_beta,
)
from unbiased_escher.solver import UnbiasedControlVariateEscher

from .config import VARIANTS


PolicyKey = Tuple[int, str]
PolicyTable = Dict[PolicyKey, np.ndarray]


def _history_key(state) -> Tuple[int, ...]:
    return tuple(int(action) for action in state.history())


def _infoset_key(state) -> PolicyKey:
    player = int(state.current_player())
    return player, str(state.information_state_string(player))


def _normalise(values, legal_actions, action_size: int) -> np.ndarray:
    result = np.zeros(int(action_size), dtype=np.float64)
    legal_actions = [int(action) for action in legal_actions]
    supplied = np.asarray(values, dtype=np.float64)
    result[legal_actions] = np.maximum(supplied[legal_actions], 0.0)
    mass = float(np.sum(result))
    if not np.isfinite(mass) or mass <= 0.0:
        result[legal_actions] = 1.0 / float(len(legal_actions))
    else:
        result /= mass
    return result


def policy_table_for_mode(solver, mode: str) -> PolicyTable:
    """Freeze either the implemented current policy or its gate-zero analogue."""

    def strategy(state):
        player = int(state.current_player())
        trainer = solver.regret_trainers[player]
        if mode == "current":
            return trainer.get_policy(state, solver.num_iteration)
        if mode != "cumulative_only":
            raise ValueError(f"Unknown policy mode: {mode}")
        cumulative = np.asarray(trainer.get_regrets(state), dtype=np.float64)
        scores = np.maximum(cumulative, 0.0)
        return trainer.regret_matching(scores, state.legal_actions())

    return build_policy_table(solver.game, strategy)


@dataclass(frozen=True)
class ScalarMoments:
    mean: float
    variance: float


@dataclass(frozen=True)
class StateControls:
    q_hat: np.ndarray
    disagreement: np.ndarray
    residual_mean: np.ndarray
    predicted_variance: np.ndarray
    beta: np.ndarray
    sampling_policy: np.ndarray
    policy: np.ndarray


def _mixture(probabilities, means, variances) -> ScalarMoments:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    mean = float(np.dot(probabilities, means))
    second = float(np.dot(probabilities, variances + np.square(means)))
    return ScalarMoments(mean=mean, variance=max(0.0, second - mean * mean))


def _weighted_mean(weights, values) -> float:
    return float(np.dot(np.asarray(weights, dtype=float), np.asarray(values, dtype=float)))


class ExactUCVOracle:
    """Enumerate the implementation's sampling law with all networks frozen."""

    def __init__(
        self,
        solver,
        *,
        policy_table: PolicyTable,
        variant_id: str,
        fold: int,
    ):
        if variant_id not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant_id}")
        if fold < 0 or fold >= int(solver.q_value_trainer.ensemble_size):
            raise ValueError("fold is outside the Q ensemble")
        self.solver = solver
        self.game = solver.game
        self.policy_table = policy_table
        self.variant_id = str(variant_id)
        self.spec = VARIANTS[variant_id]
        self.fold = int(fold)
        self.action_size = int(self.game.num_distinct_actions())
        self.iteration = max(1, int(solver.num_iteration))
        self._value_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        self._return_cache: Dict[Tuple[int, Tuple[int, ...]], ScalarMoments] = {}
        self._action_moments: Dict[Tuple[int, Tuple[int, ...]], Dict[str, Any]] = {}
        self._controls: Dict[Tuple[int, Tuple[int, ...]], StateControls] = {}

    def policy(self, state) -> np.ndarray:
        return _normalise(
            self.policy_table[_infoset_key(state)],
            state.legal_actions(),
            self.action_size,
        )

    def exact_value(self, state, traverser: int) -> float:
        key = int(traverser), _history_key(state)
        if key in self._value_cache:
            return self._value_cache[key]
        if state.is_terminal():
            value = float(state.returns()[traverser]) / float(self.solver.max_utility)
        elif state.is_chance_node():
            value = sum(
                float(probability) * self.exact_value(state.child(action), traverser)
                for action, probability in state.chance_outcomes()
            )
        else:
            policy = self.policy(state)
            value = sum(
                float(policy[action]) * self.exact_value(state.child(action), traverser)
                for action in state.legal_actions()
            )
        self._value_cache[key] = float(value)
        return float(value)

    def exact_q(self, state, traverser: int) -> np.ndarray:
        result = np.zeros(self.action_size, dtype=np.float64)
        for action in state.legal_actions():
            result[action] = self.exact_value(state.child(action), traverser)
        return result

    def controls(self, state, traverser: int) -> StateControls:
        key = int(traverser), _history_key(state)
        if key in self._controls:
            return self._controls[key]
        legal_mask = np.asarray(state.legal_actions_mask(), dtype=np.float64)
        policy = self.policy(state)
        q_hat, disagreement = self.solver.q_value_trainer.get_baseline_and_disagreement(
            state, traverser
        )
        q_hat = np.asarray(q_hat, dtype=np.float64)
        disagreement = np.asarray(disagreement, dtype=np.float64)
        if self.spec["calibration_mode"] == "frozen_predictor":
            residual_mean, predicted_variance, _ = (
                self.solver.calibration_trainer.predict_all(
                    state.information_state_tensor(traverser),
                    self.iteration,
                    disagreement,
                    traverser,
                )
            )
        else:
            residual_mean = np.zeros(self.action_size, dtype=np.float64)
            predicted_variance = np.ones(self.action_size, dtype=np.float64)

        beta_mode = self.spec["beta_mode"]
        if beta_mode == "adaptive":
            beta = variance_optimal_beta(
                q_hat,
                residual_mean,
                beta_min=self.solver.beta_min,
                beta_max=self.solver.beta_max,
                ridge=self.solver.beta_ridge,
            )
        elif beta_mode == "fixed_one":
            beta = np.ones(self.action_size, dtype=np.float64)
        elif beta_mode == "zero":
            beta = np.zeros(self.action_size, dtype=np.float64)
        else:  # pragma: no cover - frozen config guards this
            raise ValueError(beta_mode)

        if int(state.current_player()) == int(traverser):
            if self.spec["sampling_mode"] == "adaptive":
                sampling = residual_adaptive_sampling_policy(
                    predicted_variance,
                    legal_mask,
                    uniform_floor_mass=self.solver.sampling_uniform_floor_mass,
                    minimum_variance=self.solver.calibration_minimum_variance,
                )
            else:
                sampling = _normalise(legal_mask, state.legal_actions(), self.action_size)
        else:
            sampling = policy.copy()
        result = StateControls(
            q_hat=q_hat,
            disagreement=disagreement,
            residual_mean=np.asarray(residual_mean, dtype=np.float64),
            predicted_variance=np.asarray(predicted_variance, dtype=np.float64),
            beta=np.asarray(beta, dtype=np.float64),
            sampling_policy=np.asarray(sampling, dtype=np.float64),
            policy=policy,
        )
        self._controls[key] = result
        return result

    def estimator_return_moments(self, state, traverser: int) -> ScalarMoments:
        cache_key = int(traverser), _history_key(state)
        if cache_key in self._return_cache:
            return self._return_cache[cache_key]
        if state.is_terminal():
            result = ScalarMoments(
                float(state.returns()[traverser]) / float(self.solver.max_utility),
                0.0,
            )
            self._return_cache[cache_key] = result
            return result
        if state.is_chance_node():
            outcomes = list(state.chance_outcomes())
            children = [
                self.estimator_return_moments(state.child(action), traverser)
                for action, _ in outcomes
            ]
            result = _mixture(
                [probability for _, probability in outcomes],
                [child.mean for child in children],
                [child.variance for child in children],
            )
            self._return_cache[cache_key] = result
            return result

        controls = self.controls(state, traverser)
        legal_actions = [int(action) for action in state.legal_actions()]
        sampled_actions = [
            action
            for action in legal_actions
            if float(controls.sampling_policy[action]) > 0.0
        ]
        probabilities = [
            float(controls.sampling_policy[action]) for action in sampled_actions
        ]
        conditional_return_means = []
        conditional_return_vars = []
        q_means: MutableMapping[int, List[float]] = defaultdict(list)
        q_vars: MutableMapping[int, List[float]] = defaultdict(list)
        advantage_means: MutableMapping[int, List[float]] = defaultdict(list)
        advantage_vars: MutableMapping[int, List[float]] = defaultdict(list)

        legal_mask = np.asarray(state.legal_actions_mask(), dtype=np.float64)
        for sampled_action in sampled_actions:
            probability = float(controls.sampling_policy[sampled_action])
            child = self.estimator_return_moments(
                state.child(sampled_action), traverser
            )
            estimate_zero = control_variate_advantage(
                controls.q_hat,
                beta=controls.beta,
                sampled_action=sampled_action,
                sample_probability=probability,
                sampled_return=0.0,
                policy=controls.policy,
                legal_actions_mask=legal_mask,
            )
            estimate_one = control_variate_advantage(
                controls.q_hat,
                beta=controls.beta,
                sampled_action=sampled_action,
                sample_probability=probability,
                sampled_return=1.0,
                policy=controls.policy,
                legal_actions_mask=legal_mask,
            )
            return_slope = estimate_one.policy_value - estimate_zero.policy_value
            conditional_return_means.append(
                estimate_zero.policy_value + return_slope * child.mean
            )
            conditional_return_vars.append(return_slope * return_slope * child.variance)
            for target_action in legal_actions:
                q_slope = (
                    estimate_one.q_values[target_action]
                    - estimate_zero.q_values[target_action]
                )
                q_means[target_action].append(
                    float(estimate_zero.q_values[target_action]) + q_slope * child.mean
                )
                q_vars[target_action].append(q_slope * q_slope * child.variance)
                advantage_slope = (
                    estimate_one.advantages[target_action]
                    - estimate_zero.advantages[target_action]
                )
                advantage_means[target_action].append(
                    float(estimate_zero.advantages[target_action])
                    + advantage_slope * child.mean
                )
                advantage_vars[target_action].append(
                    advantage_slope * advantage_slope * child.variance
                )

        result = _mixture(probabilities, conditional_return_means, conditional_return_vars)
        self._return_cache[cache_key] = result
        if int(state.current_player()) == int(traverser):
            exact_q = self.exact_q(state, traverser)
            exact_advantage = exact_q - float(np.dot(controls.policy, exact_q))
            action_rows = {
                "q_mean": np.zeros(self.action_size, dtype=np.float64),
                "q_variance": np.zeros(self.action_size, dtype=np.float64),
                "advantage_mean": np.zeros(self.action_size, dtype=np.float64),
                "advantage_variance": np.zeros(self.action_size, dtype=np.float64),
                "exact_q": exact_q,
                "exact_advantage": exact_advantage,
            }
            for action in legal_actions:
                q_moments = _mixture(probabilities, q_means[action], q_vars[action])
                advantage_moments = _mixture(
                    probabilities,
                    advantage_means[action],
                    advantage_vars[action],
                )
                action_rows["q_mean"][action] = q_moments.mean
                action_rows["q_variance"][action] = q_moments.variance
                action_rows["advantage_mean"][action] = advantage_moments.mean
                action_rows["advantage_variance"][action] = advantage_moments.variance
            self._action_moments[cache_key] = action_rows
        return result

    def rows(self) -> List[Dict[str, Any]]:
        grouped: MutableMapping[Tuple[int, str, int], List[Dict[str, float]]] = (
            defaultdict(list)
        )
        previous_fold = int(self.solver.q_value_trainer.active_fold)
        self.solver.q_value_trainer.active_fold = self.fold
        try:
            for traverser in range(int(self.game.num_players())):
                self.estimator_return_moments(self.game.new_initial_state(), traverser)

                def walk(state, reach: float) -> None:
                    if state.is_terminal():
                        return
                    if state.is_chance_node():
                        for action, probability in state.chance_outcomes():
                            walk(state.child(action), reach * float(probability))
                        return
                    player = int(state.current_player())
                    if player == traverser:
                        cache_key = traverser, _history_key(state)
                        moments = self._action_moments[cache_key]
                        controls = self._controls[cache_key]
                        infoset = str(state.information_state_string(traverser))
                        for action in state.legal_actions():
                            action = int(action)
                            grouped[(traverser, infoset, action)].append(
                                {
                                    "weight": float(reach),
                                    "exact_q": float(moments["exact_q"][action]),
                                    "exact_advantage": float(
                                        moments["exact_advantage"][action]
                                    ),
                                    "q_mean": float(moments["q_mean"][action]),
                                    "q_variance": float(
                                        moments["q_variance"][action]
                                    ),
                                    "advantage_mean": float(
                                        moments["advantage_mean"][action]
                                    ),
                                    "advantage_variance": float(
                                        moments["advantage_variance"][action]
                                    ),
                                    "q_hat": float(controls.q_hat[action]),
                                    "disagreement": float(
                                        controls.disagreement[action]
                                    ),
                                    "residual_mean": float(
                                        controls.residual_mean[action]
                                    ),
                                    "predicted_variance": float(
                                        controls.predicted_variance[action]
                                    ),
                                    "beta": float(controls.beta[action]),
                                    "sampling_probability": float(
                                        controls.sampling_policy[action]
                                    ),
                                    "policy_probability": float(
                                        controls.policy[action]
                                    ),
                                }
                            )
                        behaviour = controls.sampling_policy
                    else:
                        behaviour = self.policy(state)
                    for action in state.legal_actions():
                        probability = float(behaviour[action])
                        if probability > 0.0:
                            walk(state.child(action), reach * probability)

                walk(self.game.new_initial_state(), 1.0)
        finally:
            self.solver.q_value_trainer.active_fold = previous_fold

        rows = []
        for (player, infoset, action), samples in sorted(grouped.items()):
            raw_weights = np.asarray([sample["weight"] for sample in samples])
            reach_mass = float(np.sum(raw_weights))
            weights = raw_weights / reach_mass
            q_mean = _weighted_mean(weights, [sample["q_mean"] for sample in samples])
            q_truth = _weighted_mean(weights, [sample["exact_q"] for sample in samples])
            q_second = _weighted_mean(
                weights,
                [
                    sample["q_variance"] + sample["q_mean"] ** 2
                    for sample in samples
                ],
            )
            q_variance = max(0.0, q_second - q_mean * q_mean)
            advantage_mean = _weighted_mean(
                weights, [sample["advantage_mean"] for sample in samples]
            )
            advantage_truth = _weighted_mean(
                weights, [sample["exact_advantage"] for sample in samples]
            )
            advantage_second = _weighted_mean(
                weights,
                [
                    sample["advantage_variance"]
                    + sample["advantage_mean"] ** 2
                    for sample in samples
                ],
            )
            advantage_variance = max(
                0.0, advantage_second - advantage_mean * advantage_mean
            )
            q_bias = q_mean - q_truth
            advantage_bias = advantage_mean - advantage_truth
            row = {
                "variant_id": self.variant_id,
                "variant_label": self.spec["label"],
                "policy_mode": self.spec["policy_mode"],
                "fold": self.fold,
                "player": player,
                "information_state": infoset,
                "action": action,
                "num_histories": len(samples),
                "sampling_reach_mass": reach_mass,
                "exact_action_value": q_truth,
                "estimator_action_value_mean": q_mean,
                "action_value_bias": q_bias,
                "action_value_variance": q_variance,
                "action_value_mse": q_variance + q_bias * q_bias,
                "exact_advantage": advantage_truth,
                "estimator_advantage_mean": advantage_mean,
                "advantage_bias": advantage_bias,
                "advantage_variance": advantage_variance,
                "advantage_mse": advantage_variance
                + advantage_bias * advantage_bias,
            }
            for key in (
                "q_hat",
                "disagreement",
                "residual_mean",
                "predicted_variance",
                "beta",
                "sampling_probability",
                "policy_probability",
            ):
                row[key] = _weighted_mean(weights, [sample[key] for sample in samples])
            rows.append(row)
        return rows


def _model_digest(named_state_dicts) -> str:
    digest = hashlib.sha256()
    for prefix, state_dict in named_state_dicts:
        digest.update(str(prefix).encode())
        for name, tensor in sorted(state_dict.items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(name).encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rng_digest() -> str:
    digest = hashlib.sha256()
    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode())
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(str(numpy_state[2:]).encode())
    digest.update(repr(random.getstate()).encode())
    digest.update(torch.random.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()


def frozen_state_fingerprint(solver) -> Dict[str, Any]:
    """Fingerprint every inference input and RNG used by the exact diagnostic."""

    tensors = []
    for player, trainer in enumerate(solver.regret_trainers):
        tensors.append((f"regret_{player}", trainer.model.state_dict()))
        if hasattr(trainer, "imm_model"):
            tensors.append((f"immediate_{player}", trainer.imm_model.state_dict()))
    for fold, member in enumerate(solver.q_value_trainer.members):
        tensors.append((f"q_target_{fold}", member.target_model.state_dict()))
    if solver.calibration_trainer is not None:
        tensors.append(
            ("calibration_target", solver.calibration_trainer.target_model.state_dict())
        )
    return {
        "model_sha256": _model_digest(tensors),
        "q_target_versions": [
            int(member.target_version) for member in solver.q_value_trainer.members
        ],
        "calibration_target_version": int(solver.calibration_trainer.target_version),
        "active_fold": int(solver.q_value_trainer.active_fold),
        "current_prediction_gates": [
            float(getattr(trainer, "prediction_gate", 0.0))
            for trainer in solver.regret_trainers
        ],
        "next_prediction_gates": [
            float(solver.gate_controller.value(player))
            for player in range(solver.num_players)
        ],
        "rng_sha256": _rng_digest(),
    }


def predictability_audit() -> Dict[str, Any]:
    """Audit source ordering required by the predictable-control argument."""

    dfs_source = inspect.getsource(UnbiasedControlVariateEscher.dfs)
    iteration_source = inspect.getsource(UnbiasedControlVariateEscher.iteration)
    dfs_markers = {
        "q_prediction": "get_baseline_and_disagreement",
        "calibration_prediction": "calibration_trainer.predict_all",
        "beta_selection": "variance_optimal_beta",
        "sampling_selection": "_traverser_sampling_policy",
        "action_draw": "np.random.choice",
        "sampled_target": "sampled_return = self.dfs",
        "calibration_update": "calibration_trainer.add",
        "q_replay_update": "q_value_trainer.add_data",
    }
    dfs_positions = {key: dfs_source.index(marker) for key, marker in dfs_markers.items()}
    selection_before_target = all(
        dfs_positions[key] < dfs_positions["sampled_target"]
        for key in (
            "q_prediction",
            "calibration_prediction",
            "beta_selection",
            "sampling_selection",
            "action_draw",
        )
    )
    target_before_updates = all(
        dfs_positions["sampled_target"] < dfs_positions[key]
        for key in ("calibration_update", "q_replay_update")
    )
    gate_set = iteration_source.index("trainer.set_prediction_gate")
    collection = iteration_source.index("self.collect_training_data")
    gate_observe = iteration_source.index("self.gate_controller.observe")
    gate_is_lagged = gate_set < collection < gate_observe
    checks = {
        "q_calibration_beta_sampling_selected_before_current_target": selection_before_target,
        "current_target_observed_before_calibration_and_q_updates": target_before_updates,
        "prediction_gate_set_before_collection_and_updated_after_collection": gate_is_lagged,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "dfs_source_sha256": hashlib.sha256(dfs_source.encode()).hexdigest(),
        "iteration_source_sha256": hashlib.sha256(iteration_source.encode()).hexdigest(),
        "dfs_marker_offsets": dfs_positions,
    }


__all__ = [
    "ExactUCVOracle",
    "frozen_state_fingerprint",
    "policy_table_for_mode",
    "predictability_audit",
]
