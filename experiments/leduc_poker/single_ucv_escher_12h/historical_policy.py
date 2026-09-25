"""Historical regret-network persistence and weighted-policy reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    PolicyKey,
    PolicyTable,
    build_policy_table,
    table_action_probabilities,
)
from vr_deep_cfr.solver import MLP


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_state_dict(model) -> dict:
    return {
        str(name): value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _normalise(values, legal_actions, action_size: int) -> np.ndarray:
    result = np.zeros(int(action_size), dtype=np.float64)
    legal = [int(action) for action in legal_actions]
    supplied = np.asarray(values, dtype=np.float64)
    result[legal] = np.maximum(supplied[legal], 0.0)
    mass = float(result.sum())
    if mass > 0.0 and np.isfinite(mass):
        result /= mass
    else:
        result[legal] = 1.0 / float(len(legal))
    return result


def _table_max_abs_difference(left: Mapping[PolicyKey, np.ndarray], right: Mapping[PolicyKey, np.ndarray]) -> float:
    if set(left) != set(right):
        return float("inf")
    return max(
        (float(np.max(np.abs(np.asarray(left[key]) - np.asarray(right[key])))) for key in left),
        default=0.0,
    )


class FrozenIterationPolicy:
    """Playable policy reconstructed solely from one stored regret-network pair."""

    def __init__(self, payload: Mapping):
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported historical-network snapshot schema")
        if payload.get("policy_rule") != "positive_regret_matching":
            raise ValueError("Unsupported historical-network policy rule")
        self.action_size = int(payload["action_size"])
        self.use_argmax = bool(payload["use_regret_matching_argmax"])
        self.models = []
        for state in payload["player_model_state_dicts"]:
            model = MLP(
                int(payload["infostate_size"]),
                list(payload["network_layers"]),
                self.action_size,
            )
            model.load_state_dict(state)
            model.eval()
            self.models.append(model)

    def action_probabilities(self, state) -> np.ndarray:
        player = int(state.current_player())
        legal = [int(action) for action in state.legal_actions()]
        features = torch.as_tensor(
            state.information_state_tensor(player), dtype=torch.float32
        )
        mask = torch.as_tensor(state.legal_actions_mask(), dtype=torch.float32)
        with torch.no_grad():
            regrets = (self.models[player](features) * mask).cpu().numpy()
        positive = np.zeros(self.action_size, dtype=np.float64)
        positive[legal] = np.maximum(regrets[legal], 0.0)
        mass = float(positive.sum())
        if mass > 0.0 and np.isfinite(mass):
            return positive / mass
        fallback = np.zeros(self.action_size, dtype=np.float64)
        if self.use_argmax:
            fallback[legal[int(np.argmax(regrets[legal]))]] = 1.0
        else:
            fallback[legal] = 1.0 / float(len(legal))
        return fallback


@dataclass(frozen=True)
class HistoricalEntry:
    iteration: int
    iteration_weight: float
    path: Path
    sha256: str
    size_bytes: int
    policy_table: PolicyTable

    def manifest_row(self, root: Path) -> dict:
        return {
            "iteration": int(self.iteration),
            "iteration_weight": float(self.iteration_weight),
            "relative_path": str(self.path.relative_to(root)),
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
        }


def load_snapshot(path: Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def save_iteration_snapshot(solver, path: Path, *, iteration: int) -> HistoricalEntry:
    if iteration <= 0:
        raise ValueError("iteration must be positive")
    if getattr(solver, "use_instantaneous_predictor", True):
        raise ValueError("Historical snapshot contract requires the non-predictive candidate")
    table = build_policy_table(
        solver.game,
        lambda state: solver.regret_trainers[int(state.current_player())].get_policy(
            state, int(iteration)
        ),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "iteration": int(iteration),
        "iteration_weight": float(iteration) ** float(solver.gamma),
        "gamma": float(solver.gamma),
        "infostate_size": int(solver.infostate_size),
        "action_size": int(solver.action_size),
        "network_layers": list(solver.network_layers),
        "use_regret_matching_argmax": bool(solver.use_regret_matching_argmax),
        "policy_rule": "positive_regret_matching",
        "player_model_state_dicts": [
            _cpu_state_dict(trainer.model) for trainer in solver.regret_trainers
        ],
        "policy_table": table,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    reconstructed = build_policy_table(
        solver.game, FrozenIterationPolicy(payload).action_probabilities
    )
    difference = _table_max_abs_difference(table, reconstructed)
    if not np.isfinite(difference) or difference > 1e-7:
        raise RuntimeError(
            f"Historical network round trip changed iteration {iteration} policy by {difference}"
        )
    return HistoricalEntry(
        iteration=int(iteration),
        iteration_weight=float(payload["iteration_weight"]),
        path=path,
        sha256=sha256_file(path),
        size_bytes=int(path.stat().st_size),
        policy_table=table,
    )


def load_entry(path: Path) -> HistoricalEntry:
    payload = load_snapshot(path)
    return HistoricalEntry(
        iteration=int(payload["iteration"]),
        iteration_weight=float(payload["iteration_weight"]),
        path=Path(path),
        sha256=sha256_file(path),
        size_bytes=int(Path(path).stat().st_size),
        policy_table={
            key: np.asarray(value, dtype=np.float64)
            for key, value in payload["policy_table"].items()
        },
    )


class HistoricalNetworkArchive:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: list[HistoricalEntry] = [
            load_entry(path) for path in sorted(self.root.glob("iteration_*.pt"))
        ]
        observed = [entry.iteration for entry in self.entries]
        if observed and observed != list(range(1, max(observed) + 1)):
            raise ValueError(f"Historical snapshot sequence has gaps: {observed}")

    def capture(self, solver, iteration: int) -> HistoricalEntry:
        path = self.root / f"iteration_{int(iteration):06d}.pt"
        if path.is_file():
            entry = load_entry(path)
            if entry.iteration != int(iteration):
                raise ValueError(f"Historical snapshot iteration mismatch: {path}")
        else:
            entry = save_iteration_snapshot(solver, path, iteration=int(iteration))
        existing = {row.iteration: row for row in self.entries}
        existing[entry.iteration] = entry
        self.entries = [existing[index] for index in sorted(existing)]
        return entry

    def through(self, iteration: int) -> list[HistoricalEntry]:
        selected = [row for row in self.entries if row.iteration <= int(iteration)]
        expected = list(range(1, int(iteration) + 1))
        if [row.iteration for row in selected] != expected:
            raise ValueError("Historical archive is incomplete at evaluation checkpoint")
        return selected

    def manifest_rows(self, worker_root: Path) -> list[dict]:
        return [row.manifest_row(Path(worker_root).resolve()) for row in self.entries]


def weighted_average_table(game, components: Sequence[tuple[PolicyTable, float]]) -> PolicyTable:
    """Convert a weighted normal-form mixture to its behavioural equivalent."""
    if not components or any(float(weight) <= 0.0 for _, weight in components):
        raise ValueError("Mixture components require positive mass")
    action_size = int(game.num_distinct_actions())
    numerators: dict[PolicyKey, np.ndarray] = {}
    denominators: dict[PolicyKey, float] = {}

    for table, mixture_weight in components:
        seen: dict[PolicyKey, tuple[float, np.ndarray]] = {}

        def walk(state, own_reaches) -> None:
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, _ in state.chance_outcomes():
                    walk(state.child(action), own_reaches)
                return
            player = int(state.current_player())
            key = player, str(state.information_state_string(player))
            policy = np.zeros(action_size, dtype=np.float64)
            supplied = table_action_probabilities(table, state)
            for action, probability in supplied.items():
                policy[int(action)] = float(probability)
            own_reach = float(own_reaches[player])
            previous = seen.get(key)
            if previous is None:
                seen[key] = own_reach, policy
                weight = float(mixture_weight) * own_reach
                numerators.setdefault(key, np.zeros(action_size, dtype=np.float64))
                denominators.setdefault(key, 0.0)
                numerators[key] += weight * policy
                denominators[key] += weight
            else:
                if not np.isclose(previous[0], own_reach, atol=1e-10, rtol=0.0):
                    raise ValueError(f"Own reach differs inside infoset {key!r}")
                if not np.allclose(previous[1], policy, atol=1e-10, rtol=0.0):
                    raise ValueError(f"Policy differs inside infoset {key!r}")
            for action in state.legal_actions():
                reaches = list(own_reaches)
                reaches[player] *= float(policy[action])
                walk(state.child(action), reaches)

        walk(game.new_initial_state(), [1.0] * int(game.num_players()))

    return {
        key: numerator / float(denominators[key])
        for key, numerator in numerators.items()
        if float(denominators[key]) > 0.0
    }


def streaming_weighted_reservoir_indices(
    entries: Sequence[HistoricalEntry], capacity: int, *, seed: int
) -> list[int]:
    """K independent streaming slots, each an exact weighted sample of the prefix."""
    if capacity <= 0 or not entries:
        raise ValueError("capacity and entries must be positive")
    rng = np.random.default_rng(int(seed))
    slots = np.zeros(int(capacity), dtype=np.int64)
    cumulative = 0.0
    for index, entry in enumerate(entries):
        weight = float(entry.iteration_weight)
        cumulative += weight
        if index == 0:
            slots.fill(0)
            continue
        replace = rng.random(int(capacity)) < weight / cumulative
        slots[replace] = int(index)
    return [int(value) for value in slots]


def bounded_mixture_table(
    game, entries: Sequence[HistoricalEntry], capacity: int, *, seed: int
) -> tuple[PolicyTable, list[int]]:
    indices = streaming_weighted_reservoir_indices(entries, capacity, seed=seed)
    return (
        weighted_average_table(
            game, [(entries[index].policy_table, 1.0) for index in indices]
        ),
        [entries[index].iteration for index in indices],
    )


def table_max_abs_difference(left: Mapping[PolicyKey, np.ndarray], right: Mapping[PolicyKey, np.ndarray]) -> float:
    return _table_max_abs_difference(left, right)


__all__ = [
    "FrozenIterationPolicy",
    "HistoricalEntry",
    "HistoricalNetworkArchive",
    "bounded_mixture_table",
    "load_snapshot",
    "sha256_file",
    "streaming_weighted_reservoir_indices",
    "table_max_abs_difference",
    "weighted_average_table",
]
