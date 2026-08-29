"""Shared endpoint, provenance, and snapshot-validation utilities."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import pyspiel
import torch
from open_spiel.python import policy

from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.policies import (
    load_policy,
    validate_policy_probabilities,
)

from .config import GAME_NAME, NODE_ENDPOINT, TIME_ENDPOINT


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)
    return path


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def augment_snapshot(
    path: str | Path,
    *,
    algorithm_id: str,
    seed: int,
    endpoint_id: str,
    nodes_touched: int,
    wall_clock_seconds: float,
    completed_iteration: int,
    repository_commit: str,
    config: Mapping[str, Any],
) -> dict:
    """Add the common held-out endpoint schema to a playable snapshot."""
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        writer = lambda value: torch.save(value, path)
    elif path.suffix == ".pkl":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        def writer(value):
            with open(path, "wb") as handle:
                pickle.dump(value, handle)

    else:
        raise ValueError(f"Unsupported snapshot extension: {path}")

    payload.update(
        {
            "heldout_benchmark_schema_version": 1,
            "heldout_algorithm_id": str(algorithm_id),
            "heldout_endpoint_id": str(endpoint_id),
            "seed": int(seed),
            "nodes_touched": int(nodes_touched),
            "wall_clock_seconds": float(wall_clock_seconds),
            "heldout_completed_iteration": int(completed_iteration),
            "repository_commit": str(repository_commit),
            "frozen_config": json_safe(dict(config)),
        }
    )
    writer(payload)
    return snapshot_record(path, relative_to=path.parent.parent)


def snapshot_record(path: str | Path, *, relative_to: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
    elif path.suffix == ".pkl":
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError(f"Unsupported snapshot extension: {path}")
    return {
        "algorithm_id": str(payload["heldout_algorithm_id"]),
        "endpoint_id": str(payload["heldout_endpoint_id"]),
        "seed": int(payload["seed"]),
        "nodes_touched": int(payload["nodes_touched"]),
        "wall_clock_seconds": float(payload["wall_clock_seconds"]),
        "completed_iteration": int(payload["heldout_completed_iteration"]),
        "repository_commit": str(payload["repository_commit"]),
        "relative_path": str(path.relative_to(Path(relative_to))),
        "filename": path.name,
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def validate_playable_snapshot(algorithm_id: str, path: str | Path) -> None:
    game = pyspiel.load_game(GAME_NAME)
    loaded = load_policy(game, algorithm_id, path)
    validate_policy_probabilities(game, loaded)
    # Constructing the table again here is intentional: it verifies the exact
    # callable interface consumed by exploitability and head-to-head analysis.
    tabular = policy.tabular_policy_from_callable(game, loaded.action_probabilities)
    values = np.asarray(tabular.action_probability_array, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite policy probabilities in {path}")


def validate_endpoint_records(records: Sequence[Mapping[str, Any]]) -> None:
    endpoints = {str(record["endpoint_id"]) for record in records}
    expected = {NODE_ENDPOINT, TIME_ENDPOINT}
    if endpoints != expected:
        raise ValueError(f"Endpoint set mismatch: expected {expected}, observed {endpoints}")
    algorithms = {str(record["algorithm_id"]) for record in records}
    seeds = {int(record["seed"]) for record in records}
    if len(algorithms) != 1 or len(seeds) != 1:
        raise ValueError("One worker result must contain one algorithm and one seed")


__all__ = [
    "augment_snapshot",
    "json_safe",
    "read_json",
    "sha256",
    "snapshot_record",
    "validate_endpoint_records",
    "validate_playable_snapshot",
    "write_csv",
    "write_json",
]
