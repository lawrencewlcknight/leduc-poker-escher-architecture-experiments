"""Save and load lightweight ESCHER average-policy snapshots."""

from __future__ import annotations

from pathlib import Path
import pickle
import random
import re
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import tensorflow as tf

from open_spiel.python import policy

from .networks import PolicyNetwork

OUTPUT_FILE_PREFIX = "leduc_poker_escher_"

SNAPSHOT_RE = re.compile(
    r"leduc_poker_escher_seed_(?P<seed>\d+)_"
    r"(?P<arm>checkpointed|continuous_baseline)_"
    r"policy_snapshot_(?P<iteration>\d+)_iters\.pkl$"
)


def prefixed_output_filename(filename: str) -> str:
    """Return a filename with the standard ESCHER output prefix exactly once."""
    filename = str(filename)
    if Path(filename).name != filename:
        raise ValueError("Pass only a filename, not a path.")
    return filename if filename.startswith(OUTPUT_FILE_PREFIX) else f"{OUTPUT_FILE_PREFIX}{filename}"


def save_pickle(obj: Any, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_full_solver_checkpoint(
    solver,
    path: str | Path,
    *,
    include_rng: bool = True,
) -> None:
    """Save solver weights, replay buffers, counters, and optional RNG state."""
    checkpoint = solver.extract_full_model()
    checkpoint["version"] = 1
    checkpoint["type"] = "escher_full_solver_checkpoint"
    checkpoint["nodes_visited"] = int(getattr(solver, "_nodes_visited", 0))
    if include_rng:
        checkpoint["python_random_state"] = random.getstate()
        checkpoint["numpy_random_state"] = np.random.get_state()
        try:
            checkpoint["tf_global_generator_state"] = (
                tf.random.get_global_generator().state.numpy()
            )
        except Exception:
            checkpoint["tf_global_generator_state"] = None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(checkpoint, path)


def load_full_solver_checkpoint(
    solver,
    path: str | Path,
    *,
    restore_rng: bool = True,
):
    """Restore a checkpoint produced by ``save_full_solver_checkpoint``."""
    checkpoint = load_pickle(path)
    solver.load_full_model(checkpoint)
    if "nodes_visited" in checkpoint:
        solver._nodes_visited = int(checkpoint["nodes_visited"])

    if restore_rng:
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if checkpoint.get("tf_global_generator_state") is not None:
            try:
                tf.random.get_global_generator().reset(
                    checkpoint["tf_global_generator_state"]
                )
            except Exception:
                pass
    return solver


def policy_snapshot_path(
    snapshot_dir: str | Path,
    seed: int,
    iteration: int,
    arm: str = "checkpointed",
) -> Path:
    return Path(snapshot_dir) / prefixed_output_filename(
        f"seed_{int(seed)}_{arm}_policy_snapshot_{int(iteration)}_iters.pkl"
    )


def full_checkpoint_path(
    checkpoint_dir: str | Path,
    seed: int,
    iteration: int,
    arm: str = "checkpointed",
) -> Path:
    return Path(checkpoint_dir) / prefixed_output_filename(
        f"seed_{int(seed)}_{arm}_full_checkpoint_{int(iteration)}_iters.pkl"
    )


def save_policy_snapshot(
    solver,
    path: str | Path,
    *,
    seed: int,
    iteration: int,
    arm: str,
    config: Dict[str, Any],
    stage_label: str,
) -> None:
    """Save only the playable average-policy network and lightweight metadata."""
    snapshot = {
        "version": 1,
        "type": "escher_policy_snapshot",
        "algorithm": "ESCHER",
        "game": str(config.get("game_name", "leduc_poker")),
        "arm": str(arm),
        "seed": int(seed),
        "checkpoint_iteration": int(iteration),
        "solver_internal_iteration": int(getattr(solver, "_iteration", -1)),
        "nodes_visited": int(solver.get_num_nodes()),
        "stage_label": str(stage_label),
        "policy_weights": solver.get_policy_weights(),
        "policy_network_layers": list(config["policy_network_layers"]),
        "input_size": int(getattr(solver, "_embedding_size")),
        "num_actions": int(getattr(solver, "_num_actions")),
    }
    save_pickle(snapshot, path)


def infer_policy_architecture_from_weights(policy_weights: Iterable[Any]) -> tuple[int, tuple[int, ...], int]:
    """Infer policy network dimensions from saved Keras weights."""
    two_d_weights = [np.asarray(weight) for weight in policy_weights if np.asarray(weight).ndim == 2]
    if len(two_d_weights) < 2:
        raise ValueError("Could not infer ESCHER policy architecture from weights.")
    input_size = int(two_d_weights[0].shape[0])
    num_actions = int(two_d_weights[-1].shape[1])
    policy_network_layers = tuple(int(weight.shape[1]) for weight in two_d_weights[:-1])
    return input_size, policy_network_layers, num_actions


class LoadedESCHERPolicy(policy.Policy):
    """OpenSpiel-compatible policy wrapper for saved ESCHER policy snapshots."""

    def __init__(self, game, snapshot_path: str | Path):
        all_players = list(range(game.num_players()))
        super().__init__(game, all_players)
        self._game = game
        self.snapshot_path = Path(snapshot_path)
        snapshot = load_pickle(snapshot_path)
        if "policy_weights" not in snapshot and "policy_weights" in snapshot.get("model", {}):
            snapshot = snapshot["model"]

        self.snapshot = snapshot
        self.arm = snapshot.get("arm", "unknown")
        self.seed = int(snapshot.get("seed", -1))
        self.checkpoint_iteration = int(snapshot.get("checkpoint_iteration", -1))
        self.nodes_visited = int(snapshot.get("nodes_visited", -1))

        policy_weights = snapshot["policy_weights"]
        input_size = int(snapshot.get("input_size", 0) or 0)
        policy_network_layers = snapshot.get("policy_network_layers")
        num_actions = int(snapshot.get("num_actions", 0) or 0)
        if not input_size or not policy_network_layers or not num_actions:
            input_size, policy_network_layers, num_actions = infer_policy_architecture_from_weights(
                policy_weights
            )

        self.input_size = int(input_size)
        self.policy_network_layers = tuple(policy_network_layers)
        self.num_actions = int(num_actions)
        self._policy_network = PolicyNetwork(
            self.input_size,
            self.policy_network_layers,
            self.num_actions,
        )
        dummy_x = tf.zeros((1, self.input_size), dtype=tf.float32)
        dummy_mask = tf.ones((1, self.num_actions), dtype=tf.float32)
        _ = self._policy_network((dummy_x, dummy_mask))
        self._policy_network.set_weights(policy_weights)

    def action_probabilities(self, state, player_id: Optional[int] = None):
        cur_player = state.current_player() if player_id is None else player_id
        legal_actions = state.legal_actions(cur_player)
        mask = tf.constant(state.legal_actions_mask(cur_player), dtype=tf.float32)
        info_state = tf.constant(state.information_state_tensor(cur_player), dtype=tf.float32)
        if len(info_state.shape) == 1:
            info_state = tf.expand_dims(info_state, axis=0)
            mask = tf.expand_dims(mask, axis=0)
        probs = self._policy_network((info_state, mask)).numpy()[0]
        legal_probs = {action: float(probs[action]) for action in legal_actions}
        total = float(sum(legal_probs.values()))
        if total <= 0 or not np.isfinite(total):
            uniform = 1.0 / len(legal_actions)
            return {action: uniform for action in legal_actions}
        return {action: prob / total for action, prob in legal_probs.items()}


def discover_policy_snapshots(snapshot_dir: str | Path) -> List[Dict[str, Any]]:
    """Return inventory rows for ESCHER policy snapshots in ``snapshot_dir``."""
    rows = []
    for path in sorted(Path(snapshot_dir).glob("leduc_poker_escher_seed_*_policy_snapshot_*_iters.pkl")):
        match = SNAPSHOT_RE.match(path.name)
        if not match:
            continue
        rows.append({
            "seed": int(match.group("seed")),
            "arm": match.group("arm"),
            "iteration": int(match.group("iteration")),
            "path": str(path.resolve()),
            "size_mb": path.stat().st_size / (1024 ** 2),
        })
    return rows
