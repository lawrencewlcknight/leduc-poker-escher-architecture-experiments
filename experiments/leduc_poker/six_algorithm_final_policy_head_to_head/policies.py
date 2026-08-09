"""Portable policy adapters and audited snapshot selection for Experiment 17."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import pickle
import re
import shutil
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from open_spiel.python import policy

from escher_poker.policy_snapshots import LoadedESCHERPolicy
from vr_deep_cfr.policy_snapshots import LoadedVRPolicy

from .config import (
    DEEP_CFR,
    DREAM,
    ESCHER,
    EXISTING_SNAPSHOT_ALGORITHMS,
    UCV_ESCHER,
    VR_DEEP_DCFR_PLUS,
    VR_DEEP_PDCFR_PLUS,
)


AUDITED_DEEP_CFR_FINAL_NODES = {
    1234: 14_882_576,
    2025: 14_890_101,
    31415: 14_887_820,
    27182: 14_957_023,
    16180: 14_963_817,
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TorchStateDictPolicy(policy.Policy):
    """Inference-only adapter for the selected Deep CFR and DREAM MLPs."""

    def __init__(self, game, snapshot_path: str | Path, algorithm_id: str):
        super().__init__(game, list(range(game.num_players())))
        self.path = Path(snapshot_path)
        self.algorithm_id = str(algorithm_id)
        snapshot = torch.load(self.path, map_location="cpu", weights_only=False)
        if self.algorithm_id == DEEP_CFR:
            state = snapshot["policy_state_dict"]
            pattern = re.compile(r"model\.(\d+)\._weight$")
            bias_key = lambda index: f"model.{index}._bias"
        elif self.algorithm_id == DREAM:
            state = snapshot["policy_network_state_dict"]
            pattern = re.compile(r"net\.(\d+)\.layer\.weight$")
            bias_key = lambda index: f"net.{index}.layer.bias"
        else:  # pragma: no cover - guarded by load_policy
            raise ValueError(f"Unsupported Torch adapter: {self.algorithm_id}")

        indexed = []
        for name, tensor in state.items():
            match = pattern.match(name)
            if match:
                indexed.append((int(match.group(1)), tensor.detach().cpu()))
        indexed.sort(key=lambda item: item[0])
        if not indexed:
            raise ValueError(f"No MLP weights found in {self.path}")
        self.layers = [
            (weight, state[bias_key(index)].detach().cpu())
            for index, weight in indexed
        ]
        self.metadata = {
            key: value
            for key, value in snapshot.items()
            if key not in {"policy_state_dict", "policy_network_state_dict"}
        }

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal_actions = list(state.legal_actions(player))
        if not legal_actions:
            return {}
        value = torch.as_tensor(
            state.information_state_tensor(player), dtype=torch.float32
        )
        with torch.no_grad():
            for index, (weight, bias) in enumerate(self.layers):
                value = torch.nn.functional.linear(value, weight, bias)
                if index < len(self.layers) - 1:
                    value = torch.relu(value)
            legal_probabilities = torch.softmax(value[legal_actions], dim=-1).numpy()
        return {
            int(action): float(probability)
            for action, probability in zip(legal_actions, legal_probabilities)
        }


class NumpyESCHERPolicy(policy.Policy):
    """TensorFlow-free exact inference for the selected plain Keras ESCHER MLP."""

    def __init__(self, game, snapshot_path: str | Path):
        super().__init__(game, list(range(game.num_players())))
        self.path = Path(snapshot_path)
        with open(self.path, "rb") as handle:
            snapshot = pickle.load(handle)
        if snapshot.get("type") != "escher_policy_snapshot":
            raise ValueError(f"Not an ESCHER policy snapshot: {self.path}")
        if bool(snapshot.get("policy_network_layer_norm", False)):
            raise ValueError("Experiment 17 selected ESCHER must not use LayerNorm")
        if str(snapshot.get("policy_network_residual_mode", "none")) != "none":
            raise ValueError("Experiment 17 selected ESCHER must use residual_mode=none")
        if int(snapshot.get("policy_network_head_depth", 0)) != 0:
            raise ValueError("Experiment 17 selected ESCHER must use a plain action head")
        weights = [np.asarray(value, dtype=np.float32) for value in snapshot["policy_weights"]]
        if len(weights) % 2 or not weights:
            raise ValueError(f"Invalid Keras Dense weight list in {self.path}")
        self.layers = list(zip(weights[0::2], weights[1::2]))
        self.activation = str(
            snapshot.get("policy_network_activation", "leakyrelu")
        ).lower()
        self.metadata = {
            key: value for key, value in snapshot.items() if key != "policy_weights"
        }

    def _activate(self, value: np.ndarray) -> np.ndarray:
        if self.activation == "leakyrelu":
            return np.where(value >= 0, value, 0.2 * value)
        if self.activation == "relu":
            return np.maximum(value, 0)
        if self.activation == "elu":
            return np.where(value >= 0, value, np.expm1(value))
        if self.activation == "tanh":
            return np.tanh(value)
        if self.activation == "swish":
            return value / (1.0 + np.exp(-value))
        if self.activation == "gelu":
            return 0.5 * value * (
                1.0
                + np.vectorize(math.erf)(value / math.sqrt(2.0))
            )
        raise ValueError(f"Unsupported ESCHER activation: {self.activation}")

    def action_probabilities(self, state, player_id=None):
        player = state.current_player() if player_id is None else int(player_id)
        legal_actions = list(state.legal_actions(player))
        if not legal_actions:
            return {}
        value = np.asarray(state.information_state_tensor(player), dtype=np.float32)
        for index, (kernel, bias) in enumerate(self.layers):
            value = value @ kernel + bias
            if index < len(self.layers) - 1:
                value = self._activate(value)
        legal_logits = np.asarray(value[legal_actions], dtype=np.float64)
        legal_logits -= float(np.max(legal_logits))
        probabilities = np.exp(legal_logits)
        probabilities /= float(np.sum(probabilities))
        return {
            int(action): float(probability)
            for action, probability in zip(legal_actions, probabilities)
        }


def load_policy(game, algorithm_id: str, snapshot_path: str | Path):
    if algorithm_id in {DEEP_CFR, DREAM}:
        return TorchStateDictPolicy(game, snapshot_path, algorithm_id)
    if algorithm_id == ESCHER:
        return NumpyESCHERPolicy(game, snapshot_path)
    if algorithm_id == UCV_ESCHER:
        return LoadedESCHERPolicy(game, snapshot_path)
    if algorithm_id in {VR_DEEP_DCFR_PLUS, VR_DEEP_PDCFR_PLUS}:
        return LoadedVRPolicy(game, snapshot_path)
    raise ValueError(f"Unknown algorithm id: {algorithm_id}")


def read_snapshot_metadata(algorithm_id: str, path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".pt":
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
    elif path.suffix == ".pkl":
        with open(path, "rb") as handle:
            snapshot = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported snapshot extension: {path}")

    if algorithm_id == DEEP_CFR:
        expected_type = "deep_cfr_policy_snapshot"
        observed_type = snapshot.get("type")
        checkpoint = int(snapshot["checkpoint_iteration"])
        nodes = snapshot.get("nodes_touched")
        game = snapshot.get("game")
    elif algorithm_id == DREAM:
        expected_type = "dream_policy_snapshot"
        observed_type = snapshot.get("kind")
        checkpoint = int(snapshot["checkpoint_iteration"])
        nodes = snapshot.get("nodes_touched")
        game = snapshot.get("game_name")
    elif algorithm_id in {ESCHER, UCV_ESCHER}:
        expected_type = "escher_policy_snapshot"
        observed_type = snapshot.get("type")
        if algorithm_id == ESCHER and "policy_weights" not in snapshot:
            raise ValueError(f"{path} is not a legacy Keras ESCHER policy")
        if algorithm_id == UCV_ESCHER and "policy_state_dict" not in snapshot:
            raise ValueError(f"{path} is not a PyTorch UCV-ESCHER policy")
        checkpoint = int(snapshot["checkpoint_iteration"])
        nodes = snapshot.get("nodes_visited")
        game = snapshot.get("game")
    else:
        expected_type = "vr_deep_cfr_policy_snapshot"
        observed_type = snapshot.get("type")
        if snapshot.get("algorithm_id") != algorithm_id:
            raise ValueError(
                f"{path} contains {snapshot.get('algorithm_id')!r}, expected {algorithm_id!r}"
            )
        checkpoint = int(snapshot["iteration"])
        nodes = snapshot.get("nodes_touched")
        game = snapshot.get("game")

    if observed_type != expected_type:
        raise ValueError(
            f"{path} has snapshot type {observed_type!r}; expected {expected_type!r}"
        )
    if str(game) != "leduc_poker":
        raise ValueError(f"{path} is for game {game!r}, not Leduc poker")
    seed = int(snapshot["seed"])
    nodes_source = "snapshot_metadata"
    if algorithm_id == DEEP_CFR and nodes is None:
        nodes = AUDITED_DEEP_CFR_FINAL_NODES.get(seed)
        nodes_source = "source_experiment_training_stage_metrics.csv"
    return {
        "algorithm_id": algorithm_id,
        "seed": seed,
        "checkpoint": checkpoint,
        "nodes_touched": None if nodes is None else int(nodes),
        "nodes_source": nodes_source,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _candidate_files(directory: Path) -> Iterable[Path]:
    for suffix in ("*.pt", "*.pkl"):
        yield from directory.rglob(suffix)


def select_final_snapshots(
    source_directories: Mapping[str, Path],
    seeds: Sequence[int],
    archive_root: Path,
) -> list[dict]:
    """Select and archive exactly one final existing snapshot per seed/algorithm."""
    required_seeds = {int(seed) for seed in seeds}
    rows = []
    for algorithm_id in EXISTING_SNAPSHOT_ALGORITHMS:
        directory = Path(source_directories[algorithm_id])
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {algorithm_id} snapshots: {directory}")
        by_seed: Dict[int, list[dict]] = {seed: [] for seed in required_seeds}
        for path in sorted(_candidate_files(directory)):
            try:
                metadata = read_snapshot_metadata(algorithm_id, path)
            except (KeyError, TypeError, ValueError, pickle.UnpicklingError):
                continue
            if metadata["seed"] in required_seeds:
                by_seed[metadata["seed"]].append(metadata)

        for seed in sorted(required_seeds):
            candidates = by_seed[seed]
            if not candidates:
                raise FileNotFoundError(
                    f"No {algorithm_id} snapshot found for seed {seed} in {directory}"
                )
            selected = max(
                candidates,
                key=lambda row: (
                    -1 if row["nodes_touched"] is None else row["nodes_touched"],
                    row["checkpoint"],
                ),
            )
            destination = archive_root / algorithm_id / Path(selected["path"]).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected["path"], destination)
            archived = read_snapshot_metadata(algorithm_id, destination)
            archived["source_path"] = selected["path"]
            if archived["sha256"] != selected["sha256"]:
                raise RuntimeError(f"Checksum changed while archiving {selected['path']}")
            rows.append(archived)
    return rows


def snapshot_directories(snapshot_root: str | Path) -> Dict[str, Path]:
    root = Path(snapshot_root)
    return {
        algorithm_id: root / algorithm_id
        for algorithm_id in EXISTING_SNAPSHOT_ALGORITHMS
    }


def local_audited_snapshot_directories() -> Dict[str, Path]:
    """Return the audited workstation sources when running in the monorepo."""
    repository = Path(__file__).resolve().parents[3]
    monorepo = repository.parents[1]
    return {
        DEEP_CFR: monorepo
        / "leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments/cloud_outputs/"
        "leduc-deep-cfr-exp27-20260801-164400/outputs/cloud/leduc-deep-cfr-exp27-/"
        "leduc_poker_deep_cfr_final_candidate_checkpoint_head_to_head_20260801_154711/"
        "snapshots",
        DREAM: monorepo
        / "leduc_poker_dream/leduc-poker-dream-experiments/cloud_outputs/"
        "leduc-dream-exp43-20260802-155924/outputs/cloud/leduc-dream-exp43/"
        "20260802_150224/snapshots",
        ESCHER: monorepo
        / "leduc_poker_escher/leduc-poker-escher-experiments/cloud_outputs/"
        "leduc-escher-exp43-20260801-171016/outputs/cloud/leduc-escher-exp43/"
        "leduc_poker_escher_final_candidate_checkpoint_head_to_head_20260801_161251/"
        "snapshots",
        UCV_ESCHER: repository
        / "cloud_outputs/leduc-escher-arch-exp16-20260802-155627/outputs/cloud/"
        "leduc-escher-arch-exp16/"
        "unbiased_escher_temporal_checkpoint_head_to_head_20260802_145917/snapshots",
    }


def validate_policy_probabilities(game, loaded_policy) -> None:
    tabular = policy.tabular_policy_from_callable(
        game, loaded_policy.action_probabilities
    )
    probabilities = np.asarray(tabular.action_probability_array, dtype=float)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Policy contains non-finite probabilities")
    if np.any(probabilities < -1e-8):
        raise ValueError("Policy contains negative probabilities")
    # Tabular conversion visits every information state and is the authoritative
    # exhaustive compatibility check used before any exact match is evaluated.
    if not math.isfinite(float(np.sum(probabilities))):
        raise ValueError("Policy probability table is invalid")


__all__ = [
    "NumpyESCHERPolicy",
    "load_policy",
    "local_audited_snapshot_directories",
    "read_snapshot_metadata",
    "select_final_snapshots",
    "sha256",
    "snapshot_directories",
    "validate_policy_probabilities",
]
