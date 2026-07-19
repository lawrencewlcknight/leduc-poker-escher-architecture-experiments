"""Parallel, coupled multi-action residual correction for Experiment 6."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from unbiased_escher.solver import UnbiasedControlVariateEscher
from unbiased_escher import variance_optimal_beta


@dataclass(frozen=True)
class ActionCorrection:
    """One included action's observable correction diagnostics."""

    action: int
    q_residual: float
    control_residual: float
    importance_correction: float


@dataclass(frozen=True)
class MultiActionControlVariateEstimate:
    """All-action estimate obtained from a nonempty sampled subset."""

    q_values: np.ndarray
    advantages: np.ndarray
    control_values: np.ndarray
    policy_value: float
    policy_weighted_advantage: float
    corrections: Tuple[ActionCorrection, ...]


@dataclass(frozen=True)
class AdaptiveSubsetDecision:
    """A conditioned-Poisson action subset and its exact marginals."""

    selected_actions: Tuple[int, ...]
    raw_inclusion_probabilities: np.ndarray
    inclusion_probabilities: np.ndarray
    control_residual_second_moments: np.ndarray
    centering_influence_norms: np.ndarray
    regret_noise_scales: np.ndarray
    empty_probability: float
    expected_subset_size: float
    diagonal_variance_proxy: float


def _normalised_legal_policy(policy, legal) -> np.ndarray:
    result = np.where(legal, np.asarray(policy, dtype=np.float64), 0.0)
    mass = float(np.sum(result))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("current_policy must have positive finite legal mass")
    return result / mass


def adaptive_nonempty_subset(
    q_values,
    *,
    beta,
    predicted_residual_means,
    predicted_residual_variances,
    current_policy,
    legal_actions_mask,
    uniform_floor_mass: float,
    rollout_cost_scale: float,
    rng: np.random.Generator,
    minimum_variance: float = 1e-6,
) -> AdaptiveSubsetDecision:
    """Sample a nonempty adaptive Bernoulli subset with known marginals.

    Before conditioning, legal action ``a`` is included independently with
    probability ``r_a``. Rejection of the empty set gives the exact marginal
    ``p_a = r_a / (1 - prod_b(1-r_b))`` used by the residual correction.

    The unconstrained cost-regularised rule is
    ``r_a = regret_noise_scale[a] / rollout_cost_scale``. It minimises the
    diagonal proxy ``w_a/r_a + rollout_cost_scale**2*r_a`` for
    ``w_a=E[(G-C_a)^2] ||(I-1*pi^T)e_a||^2``. Clipping and a uniform floor
    make it a stable, full-support adaptive subset rule.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not 0.0 < uniform_floor_mass <= 1.0:
        raise ValueError("uniform_floor_mass must lie in (0, 1]")
    if rollout_cost_scale <= 0.0 or minimum_variance <= 0.0:
        raise ValueError("rollout_cost_scale and minimum_variance must be positive")
    q_values = np.asarray(q_values, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    residual_means = np.asarray(predicted_residual_means, dtype=np.float64)
    residual_variances = np.asarray(
        predicted_residual_variances,
        dtype=np.float64,
    )
    policy = np.asarray(current_policy, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    arrays = (beta, residual_means, residual_variances, policy, legal_mask)
    if q_values.ndim != 1 or any(value.shape != q_values.shape for value in arrays):
        raise ValueError("All subset inputs must be matching one-dimensional arrays")
    legal = legal_mask > 0.0
    legal_count = int(np.sum(legal))
    if legal_count == 0:
        raise ValueError("legal_actions_mask must contain a legal action")
    for name, values in (
        ("q_values", q_values),
        ("beta", beta),
        ("predicted_residual_means", residual_means),
        ("predicted_residual_variances", residual_variances),
        ("current_policy", policy),
    ):
        if not np.all(np.isfinite(values[legal])):
            raise ValueError(f"{name} must be finite on legal actions")

    legal_policy = _normalised_legal_policy(policy, legal)
    safe_variances = np.where(
        legal,
        np.maximum(residual_variances, minimum_variance),
        0.0,
    )
    control_residual_means = residual_means + (1.0 - beta) * q_values
    second_moments = np.where(
        legal,
        safe_variances + np.square(control_residual_means),
        0.0,
    )
    influence_squared = np.where(
        legal,
        np.maximum(
            1.0
            - 2.0 * legal_policy
            + float(legal_count) * np.square(legal_policy),
            0.0,
        ),
        0.0,
    )
    influence_norms = np.sqrt(influence_squared)
    regret_noise_scales = np.where(
        legal,
        np.sqrt(second_moments) * influence_norms,
        0.0,
    )

    floor_per_action = float(uniform_floor_mass) / float(legal_count)
    if legal_count == 1:
        # Conditioning would make the only legal action certain anyway; avoid
        # wasting random draws rejecting an empty singleton subset.
        raw_probabilities = np.where(legal, 1.0, 0.0)
    else:
        raw_probabilities = np.where(
            legal,
            np.clip(
                regret_noise_scales / float(rollout_cost_scale),
                floor_per_action,
                1.0,
            ),
            0.0,
        )
    empty_probability = float(np.prod(1.0 - raw_probabilities[legal]))
    nonempty_probability = 1.0 - empty_probability
    if not np.isfinite(nonempty_probability) or nonempty_probability <= 0.0:
        raise RuntimeError("Adaptive subset has no probability of being nonempty")
    inclusion_probabilities = np.where(
        legal,
        raw_probabilities / nonempty_probability,
        0.0,
    )

    legal_indices = np.flatnonzero(legal)
    while True:
        included = rng.random(legal_count) < raw_probabilities[legal]
        if np.any(included):
            break
    selected_actions = tuple(int(action) for action in legal_indices[included])
    diagonal_proxy = float(
        np.sum(
            second_moments[legal]
            * influence_squared[legal]
            / inclusion_probabilities[legal]
        )
    )
    return AdaptiveSubsetDecision(
        selected_actions=selected_actions,
        raw_inclusion_probabilities=raw_probabilities,
        inclusion_probabilities=inclusion_probabilities,
        control_residual_second_moments=second_moments,
        centering_influence_norms=influence_norms,
        regret_noise_scales=regret_noise_scales,
        empty_probability=empty_probability,
        expected_subset_size=float(np.sum(inclusion_probabilities[legal])),
        diagonal_variance_proxy=diagonal_proxy,
    )


def multi_action_control_variate_advantage(
    q_values,
    *,
    beta,
    selected_returns: Mapping[int, float],
    inclusion_probabilities,
    policy,
    legal_actions_mask,
) -> MultiActionControlVariateEstimate:
    """Apply exact marginal inclusion correction to every sampled action."""

    values = np.asarray(q_values, dtype=np.float64)
    coefficients = np.asarray(beta, dtype=np.float64)
    inclusion = np.asarray(inclusion_probabilities, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    if (
        values.ndim != 1
        or coefficients.shape != values.shape
        or inclusion.shape != values.shape
        or legal_mask.shape != values.shape
    ):
        raise ValueError("Estimator arrays must have matching one-dimensional shapes")
    legal = legal_mask > 0.0
    if not selected_returns:
        raise ValueError("selected_returns must be nonempty")
    if not np.all(np.isfinite(values[legal])) or not np.all(
        np.isfinite(coefficients[legal])
    ):
        raise ValueError("Legal Q values and beta coefficients must be finite")
    legal_policy = _normalised_legal_policy(policy, legal)
    controls = np.where(legal, coefficients * values, 0.0)
    corrected = controls.copy()
    corrections = []
    for raw_action, raw_return in sorted(selected_returns.items()):
        action = int(raw_action)
        sampled_return = float(raw_return)
        if action < 0 or action >= values.size or not legal[action]:
            raise ValueError("Every selected action must be legal")
        probability = float(inclusion[action])
        if not np.isfinite(probability) or not 0.0 < probability <= 1.0:
            raise ValueError("Selected-action inclusion probabilities must lie in (0, 1]")
        if not np.isfinite(sampled_return):
            raise ValueError("Every sampled return must be finite")
        control_residual = sampled_return - controls[action]
        importance_correction = control_residual / probability
        corrected[action] += importance_correction
        corrections.append(
            ActionCorrection(
                action=action,
                q_residual=float(sampled_return - values[action]),
                control_residual=float(control_residual),
                importance_correction=float(importance_correction),
            )
        )
    policy_value = float(np.dot(legal_policy, corrected))
    advantages = np.where(legal, corrected - policy_value, 0.0)
    return MultiActionControlVariateEstimate(
        q_values=corrected,
        advantages=advantages,
        control_values=controls,
        policy_value=policy_value,
        policy_weighted_advantage=float(np.dot(legal_policy, advantages)),
        corrections=tuple(corrections),
    )


def _clone_generator(generator: np.random.Generator) -> np.random.Generator:
    clone = np.random.default_rng()
    clone.bit_generator.state = deepcopy(generator.bit_generator.state)
    return clone


@dataclass
class CoupledRolloutStreams:
    """Independent random streams cloned across counterfactual actions."""

    chance: np.random.Generator
    opponent: np.random.Generator
    subset: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> "CoupledRolloutStreams":
        chance_seed, opponent_seed, subset_seed = np.random.SeedSequence(
            int(seed)
        ).spawn(3)
        return cls(
            chance=np.random.default_rng(chance_seed),
            opponent=np.random.default_rng(opponent_seed),
            subset=np.random.default_rng(subset_seed),
        )

    def clone(self) -> "CoupledRolloutStreams":
        return CoupledRolloutStreams(
            chance=_clone_generator(self.chance),
            opponent=_clone_generator(self.opponent),
            subset=_clone_generator(self.subset),
        )


@dataclass
class _CorrectionRecord:
    beta: float
    predicted_variance: float
    disagreement: float
    correction: ActionCorrection
    centering_error: float


@dataclass
class _RolloutTrace:
    nodes: int = 0
    ideal_parallel_span_nodes: int = 0
    regret_events: list = field(default_factory=list)
    average_policy_events: list = field(default_factory=list)
    q_events: list = field(default_factory=list)
    calibration_events: list = field(default_factory=list)
    control_return_events: list = field(default_factory=list)
    correction_records: list = field(default_factory=list)
    subset_records: list = field(default_factory=list)
    actual_parallel_batches: int = 0

    def absorb_events(self, other: "_RolloutTrace") -> None:
        self.regret_events.extend(other.regret_events)
        self.average_policy_events.extend(other.average_policy_events)
        self.q_events.extend(other.q_events)
        self.calibration_events.extend(other.calibration_events)
        self.control_return_events.extend(other.control_return_events)
        self.correction_records.extend(other.correction_records)
        self.subset_records.extend(other.subset_records)
        self.actual_parallel_batches += other.actual_parallel_batches


class _DiagnosticEstimate:
    """Adapter for Experiment 6's correction-level diagnostic recorder."""

    def __init__(self, record: _CorrectionRecord):
        self.q_residual = record.correction.q_residual
        self.control_residual = record.correction.control_residual
        self.importance_correction = record.correction.importance_correction
        self.policy_weighted_advantage = record.centering_error


class ParallelMultiActionResidualEscher(UnbiasedControlVariateEscher):
    """Experiment 6 with adaptive, coupled action-subset rollouts."""

    def __init__(
        self,
        *args,
        subset_rollout_cost_scale: float = 2.0,
        parallel_action_workers: int = 3,
        **kwargs,
    ):
        self.subset_rollout_cost_scale = float(subset_rollout_cost_scale)
        self.parallel_action_workers = int(parallel_action_workers)
        if self.subset_rollout_cost_scale <= 0.0:
            raise ValueError("subset_rollout_cost_scale must be positive")
        if self.parallel_action_workers <= 0:
            raise ValueError("parallel_action_workers must be positive")
        self._action_executor = None
        super().__init__(*args, **kwargs)
        if self.parallel_action_workers > 1:
            self._action_executor = ThreadPoolExecutor(
                max_workers=self.parallel_action_workers,
                thread_name_prefix="escher-action",
            )

    def _reset_architecture_diagnostics(self) -> None:
        super()._reset_architecture_diagnostics()
        self._parallel_subset_stats: Dict[str, float] = {
            "information_set_count": 0.0,
            "selected_action_sum": 0.0,
            "selected_action_max": 0.0,
            "expected_subset_size_sum": 0.0,
            "multi_action_count": 0.0,
            "raw_inclusion_sum": 0.0,
            "inclusion_sum": 0.0,
            "inclusion_action_count": 0.0,
            "raw_inclusion_min": 1.0,
            "raw_inclusion_max": 0.0,
            "inclusion_min": 1.0,
            "inclusion_max": 0.0,
            "empty_probability_sum": 0.0,
            "regret_noise_scale_sum": 0.0,
            "variance_proxy_sum": 0.0,
            "paired_return_squared_difference_sum": 0.0,
            "paired_return_count": 0.0,
            "coupled_group_count": 0.0,
            "actual_parallel_batch_count": 0.0,
            "trajectory_count": 0.0,
            "trajectory_work_nodes_sum": 0.0,
            "trajectory_parallel_span_nodes_sum": 0.0,
        }

    @staticmethod
    def _categorical(generator: np.random.Generator, probabilities) -> int:
        values = np.asarray(probabilities, dtype=np.float64)
        mass = float(np.sum(values))
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("categorical probabilities must have positive mass")
        values = values / mass
        index = int(np.searchsorted(np.cumsum(values), generator.random(), side="right"))
        return min(index, values.size - 1)

    def _skip_chance_state_with_streams(self, state, streams):
        chance_nodes = 0
        while state.current_player() == -1:
            chance_nodes += 1
            actions, probabilities = zip(*state.chance_outcomes())
            index = self._categorical(streams.chance, probabilities)
            state.apply_action(actions[index])
        return state, chance_nodes

    def _rollout_action(
        self,
        child_state,
        traverser: int,
        streams: CoupledRolloutStreams,
        *,
        allow_parallel: bool,
    ):
        next_state, chance_nodes = self._skip_chance_state_with_streams(
            child_state,
            streams,
        )
        value, trace = self._rollout(
            next_state,
            traverser,
            streams,
            allow_parallel=allow_parallel,
        )
        trace.nodes += chance_nodes
        trace.ideal_parallel_span_nodes += chance_nodes
        return next_state, value, trace

    def _branch_rollouts(
        self,
        state,
        traverser: int,
        selected_actions: Sequence[int],
        streams: CoupledRolloutStreams,
        *,
        allow_parallel: bool,
    ):
        branch_inputs = [
            (int(action), state.child(int(action)), streams.clone())
            for action in selected_actions
        ]
        use_executor = (
            allow_parallel
            and len(branch_inputs) > 1
            and self._action_executor is not None
        )
        if use_executor:
            futures = [
                self._action_executor.submit(
                    self._rollout_action,
                    child_state,
                    traverser,
                    branch_streams,
                    allow_parallel=False,
                )
                for _, child_state, branch_streams in branch_inputs
            ]
            results = [future.result() for future in futures]
        else:
            results = [
                self._rollout_action(
                    child_state,
                    traverser,
                    branch_streams,
                    allow_parallel=allow_parallel,
                )
                for _, child_state, branch_streams in branch_inputs
            ]
        return [
            (action, next_state, value, trace)
            for (action, _, _), (next_state, value, trace) in zip(
                branch_inputs,
                results,
            )
        ], use_executor

    def _rollout(
        self,
        state,
        traverser: int,
        streams: CoupledRolloutStreams,
        *,
        allow_parallel: bool,
    ):
        player = state.current_player()
        if player == -4:
            return (
                state.returns()[traverser] / self.max_utility,
                _RolloutTrace(nodes=1, ideal_parallel_span_nodes=1),
            )

        legal_mask = np.asarray(state.legal_actions_mask(), dtype=np.float64)
        legal = legal_mask > 0.0
        policy = self.regret_trainers[player].get_policy(state, self.num_iteration)
        q_values, disagreement = self.q_value_trainer.get_baseline_and_disagreement(
            state,
            traverser,
        )
        infostate = state.information_state_tensor(traverser)
        if self.calibration_trainer is not None:
            residual_means, predicted_variances, calibration_features = (
                self.calibration_trainer.predict_all(
                    infostate,
                    self.num_iteration,
                    disagreement,
                    traverser,
                )
            )
        else:
            residual_means = np.zeros(self.action_size, dtype=np.float64)
            predicted_variances = np.ones(self.action_size, dtype=np.float64)
            calibration_features = None
        if self.fixed_control_variate_beta is None:
            beta = variance_optimal_beta(
                q_values,
                residual_means,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                ridge=self.beta_ridge,
            )
        else:
            beta = np.full(
                self.action_size,
                self.fixed_control_variate_beta,
                dtype=np.float64,
            )

        subset_decision = None
        if player == traverser:
            subset_decision = adaptive_nonempty_subset(
                q_values,
                beta=beta,
                predicted_residual_means=residual_means,
                predicted_residual_variances=predicted_variances,
                current_policy=policy,
                legal_actions_mask=legal_mask,
                uniform_floor_mass=self.sampling_uniform_floor_mass,
                rollout_cost_scale=self.subset_rollout_cost_scale,
                rng=streams.subset,
                minimum_variance=self.calibration_minimum_variance,
            )
            selected_actions = subset_decision.selected_actions
            inclusion_probabilities = subset_decision.inclusion_probabilities
        else:
            opponent_policy = np.where(legal, policy, 0.0)
            opponent_policy /= float(np.sum(opponent_policy))
            action = self._categorical(streams.opponent, opponent_policy)
            selected_actions = (action,)
            inclusion_probabilities = opponent_policy

        branches, used_executor = self._branch_rollouts(
            state,
            traverser,
            selected_actions,
            streams,
            allow_parallel=allow_parallel,
        )
        selected_returns = {
            action: float(value) for action, _, value, _ in branches
        }
        estimate = multi_action_control_variate_advantage(
            q_values,
            beta=beta,
            selected_returns=selected_returns,
            inclusion_probabilities=inclusion_probabilities,
            policy=policy,
            legal_actions_mask=legal_mask,
        )

        trace = _RolloutTrace(
            nodes=1 + sum(branch_trace.nodes for _, _, _, branch_trace in branches),
            ideal_parallel_span_nodes=(
                1
                + max(
                    branch_trace.ideal_parallel_span_nodes
                    for _, _, _, branch_trace in branches
                )
            ),
            actual_parallel_batches=int(used_executor),
        )
        for _, _, _, branch_trace in branches:
            trace.absorb_events(branch_trace)
        if subset_decision is not None:
            pair_squared_differences = []
            returns = list(selected_returns.values())
            for left in range(len(returns)):
                for right in range(left + 1, len(returns)):
                    pair_squared_differences.append(
                        float((returns[left] - returns[right]) ** 2)
                    )
            trace.subset_records.append(
                {
                    "decision": subset_decision,
                    "pair_squared_differences": pair_squared_differences,
                }
            )

        centering_error = abs(float(estimate.policy_weighted_advantage))
        correction_by_action = {
            correction.action: correction for correction in estimate.corrections
        }
        for action, next_state, sampled_return, _ in branches:
            correction = correction_by_action[action]
            trace.correction_records.append(
                _CorrectionRecord(
                    beta=float(beta[action]),
                    predicted_variance=float(predicted_variances[action]),
                    disagreement=float(disagreement[action]),
                    correction=correction,
                    centering_error=centering_error,
                )
            )
            if calibration_features is not None:
                trace.calibration_events.append(
                    (calibration_features[action], correction.q_residual)
                )
            trace.control_return_events.append(
                (state, traverser, action, sampled_return, self.num_iteration)
            )
            trace.q_events.append(
                (
                    self.get_history_tensor(state),
                    action,
                    self.get_history_tensor(next_state),
                    (
                        self.get_infostate_tensor(next_state)
                        if not next_state.is_terminal()
                        else None
                    ),
                    (
                        next_state.legal_actions_mask()
                        if not next_state.is_terminal()
                        else None
                    ),
                    next_state.current_player(),
                    int(next_state.is_terminal()),
                    next_state.returns()[0] / self.max_utility,
                )
            )

        if player == traverser:
            trace.regret_events.append(
                (
                    player,
                    self.get_infostate_tensor(state),
                    estimate.advantages,
                    legal_mask,
                    self.num_iteration,
                )
            )
        else:
            trace.average_policy_events.append(
                (
                    self.get_infostate_tensor(state),
                    policy,
                    legal_mask,
                    self.num_iteration,
                )
            )
        return estimate.policy_value, trace

    def _commit_trace(self, trace: _RolloutTrace) -> None:
        self.nodes_touched += int(trace.nodes)
        for player, infostate, advantages, legal_mask, iteration in trace.regret_events:
            self.regret_trainers[player].add_data(
                infostate,
                advantages,
                legal_mask,
                iteration,
            )
        for event in trace.average_policy_events:
            self.ave_policy_trainer.add_data(*event)
        observe_control_return = getattr(
            self.q_value_trainer,
            "observe_control_return",
            None,
        )
        if observe_control_return is not None:
            for state, player, action, value, iteration in trace.control_return_events:
                observe_control_return(
                    state=state,
                    player=player,
                    action=action,
                    sampled_return=value,
                    iteration=iteration,
                )
        if self.calibration_trainer is not None:
            for features, residual in trace.calibration_events:
                self.calibration_trainer.add(features, residual)
        for event in trace.q_events:
            self.q_value_trainer.add_data(*event)
        for record in trace.correction_records:
            self._record_estimate_diagnostics(
                beta=record.beta,
                predicted_variance=record.predicted_variance,
                disagreement=record.disagreement,
                estimate=_DiagnosticEstimate(record),
            )

        stats = self._parallel_subset_stats
        for subset_record in trace.subset_records:
            decision = subset_record["decision"]
            legal = decision.inclusion_probabilities > 0.0
            selected_size = len(decision.selected_actions)
            legal_count = int(np.sum(legal))
            stats["information_set_count"] += 1.0
            stats["selected_action_sum"] += float(selected_size)
            stats["selected_action_max"] = max(
                stats["selected_action_max"],
                float(selected_size),
            )
            stats["expected_subset_size_sum"] += decision.expected_subset_size
            stats["multi_action_count"] += float(selected_size > 1)
            stats["raw_inclusion_sum"] += float(
                np.sum(decision.raw_inclusion_probabilities[legal])
            )
            stats["inclusion_sum"] += float(
                np.sum(decision.inclusion_probabilities[legal])
            )
            stats["inclusion_action_count"] += float(legal_count)
            stats["raw_inclusion_min"] = min(
                stats["raw_inclusion_min"],
                float(np.min(decision.raw_inclusion_probabilities[legal])),
            )
            stats["raw_inclusion_max"] = max(
                stats["raw_inclusion_max"],
                float(np.max(decision.raw_inclusion_probabilities[legal])),
            )
            stats["inclusion_min"] = min(
                stats["inclusion_min"],
                float(np.min(decision.inclusion_probabilities[legal])),
            )
            stats["inclusion_max"] = max(
                stats["inclusion_max"],
                float(np.max(decision.inclusion_probabilities[legal])),
            )
            stats["empty_probability_sum"] += decision.empty_probability
            stats["regret_noise_scale_sum"] += float(
                np.sum(decision.regret_noise_scales[legal])
            )
            stats["variance_proxy_sum"] += decision.diagonal_variance_proxy
            pair_differences = subset_record["pair_squared_differences"]
            stats["paired_return_squared_difference_sum"] += float(
                np.sum(pair_differences)
            )
            stats["paired_return_count"] += float(len(pair_differences))
            stats["coupled_group_count"] += float(selected_size > 1)
            self._minimum_sample_probability = min(
                self._minimum_sample_probability,
                float(np.min(decision.inclusion_probabilities[legal])),
            )
        stats["actual_parallel_batch_count"] += float(trace.actual_parallel_batches)
        stats["trajectory_count"] += 1.0
        stats["trajectory_work_nodes_sum"] += float(trace.nodes)
        stats["trajectory_parallel_span_nodes_sum"] += float(
            trace.ideal_parallel_span_nodes
        )

    def collect_training_data(self, player):
        self.regret_trainers[player].reset_buffer()
        for _ in range(self.num_traversals):
            self.episode += 1
            self.q_value_trainer.begin_trajectory(self.episode)
            seed = int(np.random.randint(0, np.iinfo(np.int64).max))
            streams = CoupledRolloutStreams.from_seed(seed)
            root_state, chance_nodes = self._skip_chance_state_with_streams(
                self.game.new_initial_state(),
                streams,
            )
            _, trace = self._rollout(
                root_state,
                player,
                streams,
                allow_parallel=True,
            )
            trace.nodes += chance_nodes
            trace.ideal_parallel_span_nodes += chance_nodes
            self._commit_trace(trace)
            self._maybe_run_early_node_checkpoint()

    def dfs(
        self,
        state,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        opp_sample_reach=1.0,
        sample_reach=1.0,
    ):
        del my_reach, opp_reach, opp_sample_reach, sample_reach
        seed = int(np.random.randint(0, np.iinfo(np.int64).max))
        value, trace = self._rollout(
            state,
            traverser,
            CoupledRolloutStreams.from_seed(seed),
            allow_parallel=True,
        )
        self._commit_trace(trace)
        return value

    def evaluate(self, **kwargs):
        stats = self._parallel_subset_stats
        state_count = stats["information_set_count"]
        state_denominator = max(state_count, 1.0)
        action_count = stats["inclusion_action_count"]
        action_denominator = max(action_count, 1.0)
        pair_count = stats["paired_return_count"]
        span = stats["trajectory_parallel_span_nodes_sum"]
        work = stats["trajectory_work_nodes_sum"]
        self.logger.record("subset_information_set_count", state_count)
        self.logger.record(
            "sampled_subset_size_mean",
            stats["selected_action_sum"] / state_denominator,
        )
        self.logger.record("sampled_subset_size_max", stats["selected_action_max"])
        self.logger.record(
            "expected_subset_size_mean",
            stats["expected_subset_size_sum"] / state_denominator,
        )
        self.logger.record(
            "multi_action_information_set_fraction",
            stats["multi_action_count"] / state_denominator,
        )
        self.logger.record(
            "raw_action_inclusion_probability_mean",
            stats["raw_inclusion_sum"] / action_denominator,
        )
        self.logger.record(
            "raw_action_inclusion_probability_min",
            stats["raw_inclusion_min"] if action_count else np.nan,
        )
        self.logger.record(
            "raw_action_inclusion_probability_max",
            stats["raw_inclusion_max"] if action_count else np.nan,
        )
        self.logger.record(
            "action_inclusion_probability_mean",
            stats["inclusion_sum"] / action_denominator,
        )
        self.logger.record(
            "action_inclusion_probability_min",
            stats["inclusion_min"] if action_count else np.nan,
        )
        self.logger.record(
            "action_inclusion_probability_max",
            stats["inclusion_max"] if action_count else np.nan,
        )
        self.logger.record(
            "raw_empty_subset_probability_mean",
            stats["empty_probability_sum"] / state_denominator,
        )
        self.logger.record(
            "predicted_regret_noise_scale_mean",
            stats["regret_noise_scale_sum"] / action_denominator,
        )
        self.logger.record(
            "subset_diagonal_variance_proxy_mean",
            stats["variance_proxy_sum"] / state_denominator,
        )
        self.logger.record(
            "coupled_return_pair_squared_difference_mean",
            (
                stats["paired_return_squared_difference_sum"] / pair_count
                if pair_count
                else np.nan
            ),
        )
        self.logger.record(
            "common_random_number_group_count",
            stats["coupled_group_count"],
        )
        self.logger.record(
            "actual_parallel_action_batch_count",
            stats["actual_parallel_batch_count"],
        )
        self.logger.record(
            "ideal_parallel_node_speedup",
            work / span if span > 0.0 else np.nan,
        )
        self.logger.record(
            "ideal_parallelisable_node_fraction",
            (work - span) / work if work > 0.0 else np.nan,
        )
        return super().evaluate(**kwargs)

    def solve(self):
        try:
            return super().solve()
        finally:
            if self._action_executor is not None:
                self._action_executor.shutdown(wait=True)
                self._action_executor = None
