"""Public-state, second-round resolving utilities for Leduc poker.

The resolvers in this module deliberately operate on a *public belief state*:
all private deals compatible with the observed public card and first-round
betting sequence are retained and weighted by their blueprint reach.  They do
not condition the solve on the acting player's private card alone.

This is a controlled Leduc experiment rather than a claim of theoretically
safe subgame solving.  In particular, no opt-out gadget or opponent
counterfactual-value constraint is imposed.  Exact whole-game exploitability
is therefore measured after stitching every resolved continuation back into
the frozen blueprint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
import time
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from open_spiel.python import policy


_PUBLIC_RE = re.compile(r"\[Public: (?P<public>-?\d+)\]")
_ROUND_ONE_RE = re.compile(r"\[Round1: (?P<actions>[^\]]*)\]")


def _normalise(
    probabilities: Mapping[int, float], legal_actions: Sequence[int]
) -> dict[int, float]:
    legal = tuple(int(action) for action in legal_actions)
    values = np.asarray(
        [max(0.0, float(probabilities.get(action, 0.0))) for action in legal],
        dtype=np.float64,
    )
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        values.fill(1.0 / len(values))
    else:
        values /= total
    return {action: float(value) for action, value in zip(legal, values)}


def infoset_key(state) -> str:
    player = int(state.current_player())
    return str(state.information_state_string(player))


def is_second_round(state) -> bool:
    if state.is_terminal() or state.is_chance_node():
        return False
    return "[Round 2]" in state.information_state_string(state.current_player())


def is_second_round_root(state) -> bool:
    if not is_second_round(state):
        return False
    return "[Round2: ]" in state.information_state_string(state.current_player())


def public_root_key(state) -> str:
    """Return a key containing only observable second-round root information."""
    if not is_second_round_root(state):
        raise ValueError("State is not a second-round Leduc public root")
    text = state.information_state_string(state.current_player())
    public_match = _PUBLIC_RE.search(text)
    round_one_match = _ROUND_ONE_RE.search(text)
    if public_match is None or round_one_match is None:
        raise ValueError(f"Could not parse Leduc public root from {text!r}")
    public_card = int(public_match.group("public"))
    actions = "-".join(round_one_match.group("actions").split()) or "none"
    return f"public_{public_card}_round1_{actions}"


@dataclass(frozen=True)
class RootHistory:
    state: object
    probability: float


def collect_public_roots(game, blueprint) -> dict[str, tuple[RootHistory, ...]]:
    """Enumerate blueprint-conditioned private histories at each public root."""
    grouped: MutableMapping[str, list[tuple[object, float]]] = defaultdict(list)

    def walk(state, reach: float) -> None:
        if state.is_terminal():
            return
        if is_second_round_root(state):
            grouped[public_root_key(state)].append((state.clone(), float(reach)))
            return
        if state.is_chance_node():
            for action, probability_value in state.chance_outcomes():
                walk(state.child(action), reach * float(probability_value))
            return
        probabilities = _normalise(
            blueprint.action_probabilities(state), state.legal_actions()
        )
        for action, probability_value in probabilities.items():
            walk(state.child(action), reach * probability_value)

    walk(game.new_initial_state(), 1.0)
    result = {}
    for key, rows in grouped.items():
        total = float(sum(probability_value for _, probability_value in rows))
        if total <= 0.0:
            continue
        result[key] = tuple(
            RootHistory(state=state, probability=probability_value / total)
            for state, probability_value in rows
        )
    if not result:
        raise RuntimeError("No reachable second-round public roots were found")
    return dict(sorted(result.items()))


class StitchedResolvedPolicy(policy.Policy):
    """Use a frozen blueprint before the flop and resolved rows afterwards."""

    def __init__(self, game, blueprint, resolved_table: Mapping[str, Mapping[int, float]]):
        super().__init__(game, list(range(game.num_players())))
        self.blueprint = blueprint
        self.resolved_table = {
            str(key): {int(action): float(value) for action, value in row.items()}
            for key, row in resolved_table.items()
        }

    def action_probabilities(self, state, player_id=None):
        if is_second_round(state):
            row = self.resolved_table.get(infoset_key(state))
            if row is not None:
                return _normalise(row, state.legal_actions())
        return _normalise(
            self.blueprint.action_probabilities(state, player_id),
            state.legal_actions(
                state.current_player() if player_id is None else int(player_id)
            ),
        )


class _TabularResolverBase:
    def __init__(self, roots: Sequence[RootHistory], *, seed: int = 0):
        self.roots = tuple(roots)
        self.rng = np.random.default_rng(int(seed))
        self.regrets: Dict[str, np.ndarray] = {}
        self.strategy_sum: Dict[str, np.ndarray] = {}
        self.legal_actions: Dict[str, tuple[int, ...]] = {}
        self.baseline_sum: Dict[tuple[str, int], float] = defaultdict(float)
        self.baseline_count: Dict[tuple[str, int], int] = defaultdict(int)
        self.nodes_touched = 0
        self.terminal_nodes_touched = 0
        self.updates = 0

    def _row(self, state) -> tuple[str, tuple[int, ...], np.ndarray]:
        key = infoset_key(state)
        legal = tuple(int(action) for action in state.legal_actions())
        self.legal_actions.setdefault(key, legal)
        if self.legal_actions[key] != legal:
            raise RuntimeError(f"Legal actions changed inside information set {key}")
        if key not in self.regrets:
            self.regrets[key] = np.zeros(len(legal), dtype=np.float64)
            self.strategy_sum[key] = np.zeros(len(legal), dtype=np.float64)
        positive = np.maximum(self.regrets[key], 0.0)
        total = float(positive.sum())
        strategy = (
            positive / total
            if total > 0.0
            else np.full(len(legal), 1.0 / len(legal), dtype=np.float64)
        )
        return key, legal, strategy

    def average_policy_table(self) -> dict[str, dict[int, float]]:
        result = {}
        for key, legal in self.legal_actions.items():
            values = np.maximum(self.strategy_sum[key], 0.0)
            total = float(values.sum())
            if total <= 0.0:
                positive = np.maximum(self.regrets[key], 0.0)
                total = float(positive.sum())
                values = (
                    positive
                    if total > 0.0
                    else np.ones(len(legal), dtype=np.float64)
                )
                total = float(values.sum())
            result[key] = {
                action: float(value / total) for action, value in zip(legal, values)
            }
        return result

    def run_to(self, target_updates: int) -> None:
        raise NotImplementedError


class CFRPlusPublicResolver(_TabularResolverBase):
    """Full-tree CFR+ over a blueprint-conditioned public-root belief state."""

    def _traverse(
        self,
        state,
        traverser: int,
        player_reach: tuple[float, float],
        root_probability: float,
        regret_delta: MutableMapping[str, np.ndarray],
        average_delta: MutableMapping[str, np.ndarray],
    ) -> float:
        self.nodes_touched += 1
        if state.is_terminal():
            self.terminal_nodes_touched += 1
            return float(state.returns()[traverser])
        if state.is_chance_node():
            return float(
                sum(
                    probability_value
                    * self._traverse(
                        state.child(action),
                        traverser,
                        player_reach,
                        root_probability,
                        regret_delta,
                        average_delta,
                    )
                    for action, probability_value in state.chance_outcomes()
                )
            )
        actor = int(state.current_player())
        key, legal, strategy = self._row(state)
        if actor == traverser:
            average_delta.setdefault(key, np.zeros(len(legal), dtype=np.float64))
            average_delta[key] += (
                float(self.updates + 1)
                * root_probability
                * player_reach[actor]
                * strategy
            )
        action_values = np.empty(len(legal), dtype=np.float64)
        for index, action in enumerate(legal):
            next_reach = list(player_reach)
            next_reach[actor] *= float(strategy[index])
            action_values[index] = self._traverse(
                state.child(action),
                traverser,
                tuple(next_reach),
                root_probability,
                regret_delta,
                average_delta,
            )
        node_value = float(np.dot(strategy, action_values))
        if actor == traverser:
            opponent = 1 - actor
            regret_delta.setdefault(key, np.zeros(len(legal), dtype=np.float64))
            regret_delta[key] += (
                root_probability
                * player_reach[opponent]
                * (action_values - node_value)
            )
        return node_value

    def run_to(self, target_updates: int) -> None:
        while self.updates < int(target_updates):
            average_delta: MutableMapping[str, np.ndarray] = {}
            for traverser in (0, 1):
                regret_delta: MutableMapping[str, np.ndarray] = {}
                for root in self.roots:
                    self._traverse(
                        root.state,
                        traverser,
                        (1.0, 1.0),
                        root.probability,
                        regret_delta,
                        average_delta,
                    )
                for key, delta in regret_delta.items():
                    self.regrets[key] = np.maximum(self.regrets[key] + delta, 0.0)
            for key, delta in average_delta.items():
                self.strategy_sum[key] += delta
            self.updates += 1


class UCVExternalSamplingPublicResolver(_TabularResolverBase):
    """External-sampling CFR+ with predictable tabular control variates.

    Traverser actions are enumerated.  At opponent nodes one action is sampled
    from the current full-support strategy.  A pre-update running action-value
    baseline supplies the control variate, so the returned node-value estimate
    remains conditionally unbiased.  This is the cleanest local analogue of
    UCV-ESCHER while retaining tabular regrets and avoiding neural fitting.
    """

    def __init__(
        self,
        roots: Sequence[RootHistory],
        *,
        seed: int,
        exploration: float = 0.05,
        use_control_variate: bool = True,
    ):
        super().__init__(roots, seed=seed)
        self.exploration = float(exploration)
        self.use_control_variate = bool(use_control_variate)
        if not 0.0 <= self.exploration < 1.0:
            raise ValueError("exploration must be in [0, 1)")
        root_probabilities = np.asarray(
            [root.probability for root in self.roots], dtype=np.float64
        )
        self.root_probabilities = root_probabilities / root_probabilities.sum()
        self.control_variate_correction_sum = 0.0
        self.control_variate_correction_sq_sum = 0.0
        self.control_variate_samples = 0

    def _baseline(self, key: str, action: int) -> float:
        count = self.baseline_count[(key, int(action))]
        return (
            self.baseline_sum[(key, int(action))] / count if count else 0.0
        )

    def _traverse(
        self,
        state,
        traverser: int,
        regret_delta: MutableMapping[str, np.ndarray],
        average_delta: MutableMapping[str, np.ndarray],
        sampling_weight: float,
    ) -> float:
        self.nodes_touched += 1
        if state.is_terminal():
            self.terminal_nodes_touched += 1
            return float(state.returns()[traverser])
        if state.is_chance_node():
            outcomes = tuple(state.chance_outcomes())
            probabilities = np.asarray([row[1] for row in outcomes], dtype=np.float64)
            selected = int(self.rng.choice(len(outcomes), p=probabilities / probabilities.sum()))
            return self._traverse(
                state.child(outcomes[selected][0]),
                traverser,
                regret_delta,
                average_delta,
                sampling_weight,
            )
        actor = int(state.current_player())
        key, legal, strategy = self._row(state)
        # In two-player external sampling, update a player's average strategy
        # on the opponent's regret pass.  Correct for the full-support sampling
        # mixture used on earlier nodes of that same player's path.
        if actor == (traverser + 1) % 2:
            average_delta.setdefault(key, np.zeros(len(legal), dtype=np.float64))
            average_delta[key] += (
                float(self.updates + 1) * sampling_weight * strategy
            )
        if actor == traverser:
            action_values = np.asarray(
                [
                    self._traverse(
                        state.child(action),
                        traverser,
                        regret_delta,
                        average_delta,
                        sampling_weight,
                    )
                    for action in legal
                ],
                dtype=np.float64,
            )
            node_value = float(np.dot(strategy, action_values))
            regret_delta.setdefault(key, np.zeros(len(legal), dtype=np.float64))
            regret_delta[key] += action_values - node_value
            return node_value

        sampling = (1.0 - self.exploration) * strategy + self.exploration / len(legal)
        selected = int(self.rng.choice(len(legal), p=sampling))
        action = legal[selected]
        baseline_values = (
            np.asarray(
                [self._baseline(key, candidate) for candidate in legal],
                dtype=np.float64,
            )
            if self.use_control_variate
            else np.zeros(len(legal), dtype=np.float64)
        )
        baseline_node = float(np.dot(strategy, baseline_values))
        child_value = self._traverse(
            state.child(action),
            traverser,
            regret_delta,
            average_delta,
            sampling_weight * float(strategy[selected] / sampling[selected]),
        )
        correction = float(
            strategy[selected]
            / sampling[selected]
            * (child_value - baseline_values[selected])
        )
        estimate = baseline_node + correction
        self.control_variate_correction_sum += correction
        self.control_variate_correction_sq_sum += correction * correction
        self.control_variate_samples += 1
        # Update only after constructing the target: the baseline used above is
        # measurable before the current sample, preserving predictability.
        if self.use_control_variate:
            self.baseline_sum[(key, action)] += child_value
            self.baseline_count[(key, action)] += 1
        return float(estimate)

    def run_to(self, target_updates: int) -> None:
        while self.updates < int(target_updates):
            average_delta: MutableMapping[str, np.ndarray] = {}
            for traverser in (0, 1):
                root_index = int(
                    self.rng.choice(len(self.roots), p=self.root_probabilities)
                )
                regret_delta: MutableMapping[str, np.ndarray] = {}
                self._traverse(
                    self.roots[root_index].state,
                    traverser,
                    regret_delta,
                    average_delta,
                    1.0,
                )
                for key, delta in regret_delta.items():
                    self.regrets[key] = np.maximum(self.regrets[key] + delta, 0.0)
            for key, delta in average_delta.items():
                self.strategy_sum[key] += delta
            self.updates += 1

    def variance_diagnostics(self) -> dict[str, float]:
        count = self.control_variate_samples
        if count == 0:
            return {
                "control_variate_samples": 0,
                "control_variate_correction_mean": float("nan"),
                "control_variate_correction_variance": float("nan"),
            }
        mean = self.control_variate_correction_sum / count
        variance = max(
            0.0,
            self.control_variate_correction_sq_sum / count - mean * mean,
        )
        return {
            "control_variate_samples": int(count),
            "control_variate_correction_mean": float(mean),
            "control_variate_correction_variance": float(variance),
        }


def solve_public_roots(
    roots_by_key: Mapping[str, Sequence[RootHistory]],
    *,
    resolver_kind: str,
    target_updates: int,
    seed: int,
    exploration: float = 0.05,
) -> tuple[dict[str, dict[int, float]], list[dict]]:
    """Solve every public root and return one stitched continuation table."""
    table: dict[str, dict[int, float]] = {}
    diagnostics = []
    for root_index, (root_key, roots) in enumerate(sorted(roots_by_key.items())):
        local_seed = int(np.random.SeedSequence([seed, root_index]).generate_state(1)[0])
        if resolver_kind == "cfr_plus":
            resolver = CFRPlusPublicResolver(roots, seed=local_seed)
        elif resolver_kind in {"external_sampling", "ucv_external_sampling"}:
            resolver = UCVExternalSamplingPublicResolver(
                roots,
                seed=local_seed,
                exploration=exploration,
                use_control_variate=resolver_kind == "ucv_external_sampling",
            )
        else:
            raise ValueError(f"Unknown resolver kind: {resolver_kind}")
        started = time.perf_counter()
        resolver.run_to(int(target_updates))
        elapsed = time.perf_counter() - started
        local_table = resolver.average_policy_table()
        overlap = set(table).intersection(local_table)
        if overlap:
            raise RuntimeError(
                f"Information sets occurred in multiple public roots: {sorted(overlap)[:3]}"
            )
        table.update(local_table)
        row = {
            "public_root": root_key,
            "resolver_kind": resolver_kind,
            "resolver_seed": int(seed),
            "target_updates": int(target_updates),
            "num_private_root_histories": len(roots),
            "solve_seconds": float(elapsed),
            "nodes_touched": int(resolver.nodes_touched),
            "terminal_nodes_touched": int(resolver.terminal_nodes_touched),
            "num_resolved_information_sets": len(local_table),
        }
        if isinstance(resolver, UCVExternalSamplingPublicResolver):
            row.update(resolver.variance_diagnostics())
            row["control_variate_enabled"] = resolver.use_control_variate
        diagnostics.append(row)
    return table, diagnostics


__all__ = [
    "CFRPlusPublicResolver",
    "RootHistory",
    "StitchedResolvedPolicy",
    "UCVExternalSamplingPublicResolver",
    "collect_public_roots",
    "infoset_key",
    "is_second_round",
    "is_second_round_root",
    "public_root_key",
    "solve_public_roots",
]
