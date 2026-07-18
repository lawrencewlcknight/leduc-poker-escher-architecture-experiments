"""Exact Leduc diagnostics for Experiment 5.

All routines in this module enumerate the small Leduc tree. They are evaluation
oracles only: their visits are deliberately excluded from the solver's training
``nodes_touched`` counter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
from open_spiel.python.algorithms import exploitability
from open_spiel.python.policy import tabular_policy_from_callable


PolicyKey = Tuple[int, str]
PolicyTable = Dict[PolicyKey, np.ndarray]


def _history_key(state) -> Tuple[int, ...]:
    return tuple(int(action) for action in state.history())


def _infoset_key(state) -> PolicyKey:
    player = int(state.current_player())
    return player, str(state.information_state_string(player))


def _normalised_policy(values, legal_actions, action_size: int) -> np.ndarray:
    legal_actions = [int(action) for action in legal_actions]
    result = np.zeros(int(action_size), dtype=np.float64)
    supplied = np.asarray(values, dtype=np.float64)
    result[legal_actions] = np.maximum(supplied[legal_actions], 0.0)
    mass = float(np.sum(result))
    if not np.isfinite(mass) or mass <= 0.0:
        result[legal_actions] = 1.0 / float(len(legal_actions))
    else:
        result /= mass
    return result


def build_policy_table(game, strategy: Callable[[Any], np.ndarray]) -> PolicyTable:
    """Enumerate a behavioural strategy and verify infoset consistency."""

    action_size = int(game.num_distinct_actions())
    table: PolicyTable = {}

    def walk(state) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        key = _infoset_key(state)
        policy = _normalised_policy(strategy(state), state.legal_actions(), action_size)
        previous = table.get(key)
        if previous is not None and not np.allclose(
            previous, policy, atol=1e-10, rtol=0.0
        ):
            raise ValueError(f"Strategy is inconsistent inside infoset {key!r}")
        table[key] = policy
        for action in state.legal_actions():
            walk(state.child(action))

    walk(game.new_initial_state())
    return table


def table_action_probabilities(table: Mapping[PolicyKey, np.ndarray], state):
    legal_actions = [int(action) for action in state.legal_actions()]
    policy = table.get(_infoset_key(state))
    if policy is None:
        probability = 1.0 / float(len(legal_actions))
        return {action: probability for action in legal_actions}
    values = _normalised_policy(policy, legal_actions, len(policy))
    return {action: float(values[action]) for action in legal_actions}


def exact_exploitability(game, table: Mapping[PolicyKey, np.ndarray]) -> float:
    policy = tabular_policy_from_callable(
        game,
        lambda state: table_action_probabilities(table, state),
    )
    return float(exploitability.exploitability(game, policy))


class ExactWeightedAverageStrategy:
    """Exact reach- and iteration-weighted tabular average strategy."""

    def __init__(self, game, *, gamma: float):
        self.game = game
        self.gamma = float(gamma)
        self.action_size = int(game.num_distinct_actions())
        self.numerators: Dict[PolicyKey, np.ndarray] = {}
        self.denominators: Dict[PolicyKey, float] = {}

    def observe_iteration(
        self,
        iteration: int,
        strategy: Callable[[Any], np.ndarray],
    ) -> None:
        if iteration <= 0:
            raise ValueError("iteration must be positive")
        iteration_weight = float(iteration) ** self.gamma
        seen: Dict[PolicyKey, Tuple[float, np.ndarray]] = {}

        def walk(state, own_reaches) -> None:
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    walk(state.child(action), own_reaches)
                return
            player = int(state.current_player())
            key = _infoset_key(state)
            policy = _normalised_policy(
                strategy(state), state.legal_actions(), self.action_size
            )
            own_reach = float(own_reaches[player])
            previous = seen.get(key)
            if previous is None:
                seen[key] = (own_reach, policy)
                weight = iteration_weight * own_reach
                if key not in self.numerators:
                    self.numerators[key] = np.zeros(self.action_size, dtype=np.float64)
                    self.denominators[key] = 0.0
                self.numerators[key] += weight * policy
                self.denominators[key] += weight
            else:
                previous_reach, previous_policy = previous
                if not np.isclose(previous_reach, own_reach, atol=1e-10, rtol=0.0):
                    raise ValueError(f"Own reach differs inside infoset {key!r}")
                if not np.allclose(previous_policy, policy, atol=1e-10, rtol=0.0):
                    raise ValueError(f"Policy differs inside infoset {key!r}")
            for action in state.legal_actions():
                child_reaches = list(own_reaches)
                child_reaches[player] *= float(policy[action])
                walk(state.child(action), child_reaches)

        walk(self.game.new_initial_state(), [1.0] * int(self.game.num_players()))

    def table(self) -> PolicyTable:
        result = {}
        for key, numerator in self.numerators.items():
            denominator = float(self.denominators[key])
            if denominator > 0.0:
                result[key] = numerator / denominator
        return result

    def exploitability(self) -> float:
        return exact_exploitability(self.game, self.table())


@dataclass(frozen=True)
class ScalarMoments:
    mean: float
    variance: float


def _mixture_moments(probabilities, means, variances) -> ScalarMoments:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    mean = float(np.dot(probabilities, means))
    second = float(np.dot(probabilities, variances + np.square(means)))
    return ScalarMoments(mean, max(0.0, second - mean * mean))


class ExactLeducOracle:
    """Exact strategy, Q, and frozen-controller estimator diagnostics."""

    def __init__(self, solver, policy_table: PolicyTable):
        self.solver = solver
        self.game = solver.game
        self.policy_table = policy_table
        self.action_size = int(self.game.num_distinct_actions())
        self.iteration = max(1, int(solver.num_iteration))
        self._value_cache: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        self._qhat_cache: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
        self._return_cache: Dict[Tuple[int, Tuple[int, ...]], ScalarMoments] = {}
        self._advantage_moments: Dict[
            Tuple[int, Tuple[int, ...]], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    def policy(self, state) -> np.ndarray:
        key = _infoset_key(state)
        return _normalised_policy(
            self.policy_table[key], state.legal_actions(), self.action_size
        )

    def exact_value(self, state, traverser: int) -> float:
        key = int(traverser), _history_key(state)
        cached = self._value_cache.get(key)
        if cached is not None:
            return cached
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
        values = np.zeros(self.action_size, dtype=np.float64)
        for action in state.legal_actions():
            values[action] = self.exact_value(state.child(action), traverser)
        return values

    def qhat(self, state, traverser: int) -> np.ndarray:
        key = int(traverser), _history_key(state)
        if key not in self._qhat_cache:
            self._qhat_cache[key] = np.asarray(
                self.solver.q_value_trainer.get_baseline(state, traverser),
                dtype=np.float64,
            )
        return self._qhat_cache[key].copy()

    def estimator_return_moments(self, state, traverser: int) -> ScalarMoments:
        cache_key = int(traverser), _history_key(state)
        cached = self._return_cache.get(cache_key)
        if cached is not None:
            return cached
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
            result = _mixture_moments(
                [probability for _, probability in outcomes],
                [child.mean for child in children],
                [child.variance for child in children],
            )
            self._return_cache[cache_key] = result
            return result

        player = int(state.current_player())
        legal_actions = [int(action) for action in state.legal_actions()]
        policy = self.policy(state)
        if player == traverser:
            behaviour = np.zeros(self.action_size, dtype=np.float64)
            behaviour[legal_actions] = 1.0 / float(len(legal_actions))
        else:
            behaviour = policy
        qhat = self.qhat(state, traverser)
        policy_qhat = float(np.dot(policy, qhat))
        sampled_actions = [
            action for action in legal_actions if float(behaviour[action]) > 0.0
        ]
        conditional_return_means = []
        conditional_return_vars = []
        advantage_conditional_means = defaultdict(list)
        advantage_conditional_vars = defaultdict(list)

        for action in sampled_actions:
            probability = float(behaviour[action])
            child = self.estimator_return_moments(state.child(action), traverser)
            lambda_value = float(
                self.solver.lambda_controller.value(traverser, action, self.iteration)
            )
            correction_scale = lambda_value / probability
            return_coefficient = float(policy[action]) * correction_scale
            return_intercept = (
                policy_qhat - return_coefficient * float(qhat[action])
            )
            conditional_return_means.append(
                return_intercept + return_coefficient * child.mean
            )
            conditional_return_vars.append(
                return_coefficient * return_coefficient * child.variance
            )

            for target_action in legal_actions:
                base = float(qhat[target_action] - policy_qhat)
                centering_coefficient = (
                    (1.0 if target_action == action else 0.0)
                    - float(policy[action])
                )
                coefficient = centering_coefficient * correction_scale
                intercept = base - coefficient * float(qhat[action])
                advantage_conditional_means[target_action].append(
                    intercept + coefficient * child.mean
                )
                advantage_conditional_vars[target_action].append(
                    coefficient * coefficient * child.variance
                )

        probabilities = [float(behaviour[action]) for action in sampled_actions]
        result = _mixture_moments(
            probabilities,
            conditional_return_means,
            conditional_return_vars,
        )
        self._return_cache[cache_key] = result

        if player == traverser:
            exact_q = self.exact_q(state, traverser)
            exact_advantage = exact_q - float(np.dot(policy, exact_q))
            means = np.zeros(self.action_size, dtype=np.float64)
            variances = np.zeros(self.action_size, dtype=np.float64)
            for target_action in legal_actions:
                moments = _mixture_moments(
                    probabilities,
                    advantage_conditional_means[target_action],
                    advantage_conditional_vars[target_action],
                )
                means[target_action] = moments.mean
                variances[target_action] = moments.variance
            self._advantage_moments[cache_key] = (
                means,
                variances,
                exact_advantage,
            )
        return result

    def _enumerate_estimator_tree(self, traverser: int) -> None:
        self.estimator_return_moments(self.game.new_initial_state(), traverser)

    def estimator_rows(self) -> List[Dict[str, Any]]:
        grouped: MutableMapping[Tuple[int, str, int], List[Dict[str, float]]] = (
            defaultdict(list)
        )
        for traverser in range(int(self.game.num_players())):
            self._enumerate_estimator_tree(traverser)

            def walk(state, reach: float) -> None:
                if state.is_terminal():
                    return
                if state.is_chance_node():
                    for action, probability in state.chance_outcomes():
                        walk(state.child(action), reach * float(probability))
                    return
                player = int(state.current_player())
                policy = self.policy(state)
                legal_actions = [int(action) for action in state.legal_actions()]
                if player == traverser:
                    cache_key = traverser, _history_key(state)
                    means, variances, exact_advantage = self._advantage_moments[
                        cache_key
                    ]
                    infoset = str(state.information_state_string(traverser))
                    for action in legal_actions:
                        grouped[(traverser, infoset, action)].append(
                            {
                                "weight": float(reach),
                                "mean": float(means[action]),
                                "variance": float(variances[action]),
                                "truth": float(exact_advantage[action]),
                            }
                        )
                    behaviour = np.zeros(self.action_size, dtype=np.float64)
                    behaviour[legal_actions] = 1.0 / float(len(legal_actions))
                else:
                    behaviour = policy
                for action in legal_actions:
                    probability = float(behaviour[action])
                    if probability > 0.0:
                        walk(state.child(action), reach * probability)

            walk(self.game.new_initial_state(), 1.0)

        rows = []
        for (player, infoset, action), samples in sorted(grouped.items()):
            weights = np.asarray([sample["weight"] for sample in samples], dtype=float)
            if float(np.sum(weights)) <= 0.0:
                weights = np.ones_like(weights)
            weights /= float(np.sum(weights))
            means = np.asarray([sample["mean"] for sample in samples], dtype=float)
            variances = np.asarray(
                [sample["variance"] for sample in samples], dtype=float
            )
            truths = np.asarray([sample["truth"] for sample in samples], dtype=float)
            estimator_mean = float(np.dot(weights, means))
            exact_mean = float(np.dot(weights, truths))
            total_variance = float(
                np.dot(weights, variances + np.square(means - estimator_mean))
            )
            mse = float(np.dot(weights, variances + np.square(means - truths)))
            rows.append(
                {
                    "player": player,
                    "information_state": infoset,
                    "action": action,
                    "num_histories": len(samples),
                    "sampling_reach_mass": float(
                        sum(sample["weight"] for sample in samples)
                    ),
                    "exact_advantage": exact_mean,
                    "estimator_mean": estimator_mean,
                    "estimator_bias": estimator_mean - exact_mean,
                    "estimator_variance": total_variance,
                    "estimator_mse": mse,
                }
            )
        return rows

    def q_oracle_rows(self) -> List[Dict[str, Any]]:
        grouped = defaultdict(list)

        def walk(state, on_policy_reach: float) -> None:
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, probability in state.chance_outcomes():
                    walk(state.child(action), on_policy_reach * float(probability))
                return
            player = int(state.current_player())
            infoset = str(state.information_state_string(player))
            qhat = self.qhat(state, 0)
            oracle = self.exact_q(state, 0)
            policy = self.policy(state)
            for action in state.legal_actions():
                grouped[(player, infoset, int(action))].append(
                    {
                        "weight": float(on_policy_reach),
                        "qhat": float(qhat[action]),
                        "oracle": float(oracle[action]),
                    }
                )
                walk(
                    state.child(action),
                    on_policy_reach * float(policy[action]),
                )

        walk(self.game.new_initial_state(), 1.0)
        rows = []
        for (player, infoset, action), samples in sorted(grouped.items()):
            qhat = np.asarray([sample["qhat"] for sample in samples], dtype=float)
            oracle = np.asarray([sample["oracle"] for sample in samples], dtype=float)
            error = qhat - oracle
            reach = np.asarray([sample["weight"] for sample in samples], dtype=float)
            if float(np.sum(reach)) > 0.0:
                normalised_reach = reach / float(np.sum(reach))
                weighted_rmse = float(np.sqrt(np.dot(normalised_reach, error * error)))
            else:
                weighted_rmse = np.nan
            rows.append(
                {
                    "player_to_act": player,
                    "information_state": infoset,
                    "action": action,
                    "num_histories": len(samples),
                    "on_policy_reach_mass": float(np.sum(reach)),
                    "q_estimate_mean": float(np.mean(qhat)),
                    "q_oracle_mean": float(np.mean(oracle)),
                    "q_error_mean": float(np.mean(error)),
                    "q_error_mae": float(np.mean(np.abs(error))),
                    "q_error_rmse": float(np.sqrt(np.mean(error * error))),
                    "q_error_on_policy_rmse": weighted_rmse,
                }
            )
        return rows


def aggregate_estimator_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {
            "estimator_bias_abs_mean": np.nan,
            "estimator_variance_mean": np.nan,
            "estimator_mse_mean": np.nan,
        }
    weights = np.asarray(
        [max(float(row["sampling_reach_mass"]), 0.0) for row in rows], dtype=float
    )
    if float(np.sum(weights)) <= 0.0:
        weights = np.ones_like(weights)
    weights /= float(np.sum(weights))
    return {
        "estimator_bias_abs_mean": float(
            np.dot(weights, [abs(float(row["estimator_bias"])) for row in rows])
        ),
        "estimator_variance_mean": float(
            np.dot(weights, [float(row["estimator_variance"]) for row in rows])
        ),
        "estimator_mse_mean": float(
            np.dot(weights, [float(row["estimator_mse"]) for row in rows])
        ),
    }


def aggregate_q_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"q_oracle_mae": np.nan, "q_oracle_rmse": np.nan}
    return {
        "q_oracle_mae": float(np.mean([float(row["q_error_mae"]) for row in rows])),
        "q_oracle_rmse": float(
            np.sqrt(np.mean([float(row["q_error_rmse"]) ** 2 for row in rows]))
        ),
    }
