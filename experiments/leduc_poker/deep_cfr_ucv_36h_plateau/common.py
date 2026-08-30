"""Shared persistence and validation helpers for Experiment 21."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import torch

from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    json_safe,
    read_json,
    sha256,
    validate_playable_snapshot,
    write_csv,
    write_json,
)


def augment_snapshot(
    path: str | Path,
    *,
    algorithm_id: str,
    seed: int,
    checkpoint: Mapping[str, Any],
    nodes_touched: int,
    wall_clock_seconds: float,
    completed_iteration: int,
    repository_commit: str,
    config: Mapping[str, Any],
) -> dict:
    """Embed the convergence checkpoint contract in a playable policy file."""
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)

        def writer(value):
            torch.save(value, path)

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
            "convergence_experiment_schema_version": 1,
            "convergence_algorithm_id": str(algorithm_id),
            "convergence_checkpoint_id": str(checkpoint["checkpoint_id"]),
            "checkpoint_target_active_seconds": float(
                checkpoint["target_active_seconds"]
            ),
            "checkpoint_target_active_hours": float(
                checkpoint["target_active_hours"]
            ),
            "seed": int(seed),
            "nodes_touched": int(nodes_touched),
            "wall_clock_seconds": float(wall_clock_seconds),
            "completed_iteration": int(completed_iteration),
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
        "algorithm_id": str(payload["convergence_algorithm_id"]),
        "checkpoint_id": str(payload["convergence_checkpoint_id"]),
        "checkpoint_target_active_seconds": float(
            payload["checkpoint_target_active_seconds"]
        ),
        "checkpoint_target_active_hours": float(
            payload["checkpoint_target_active_hours"]
        ),
        "seed": int(payload["seed"]),
        "nodes_touched": int(payload["nodes_touched"]),
        "wall_clock_seconds": float(payload["wall_clock_seconds"]),
        "completed_iteration": int(payload["completed_iteration"]),
        "repository_commit": str(payload["repository_commit"]),
        "relative_path": str(path.relative_to(Path(relative_to))),
        "filename": path.name,
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    algorithm_id: str,
    seed: int,
    schedule: Sequence[Mapping[str, Any]],
) -> None:
    expected_ids = [str(row["checkpoint_id"]) for row in schedule]
    observed_ids = [str(row["checkpoint_id"]) for row in records]
    if observed_ids != expected_ids:
        raise ValueError(
            f"Checkpoint IDs differ: expected {expected_ids}, observed {observed_ids}"
        )
    if {str(row["algorithm_id"]) for row in records} != {str(algorithm_id)}:
        raise ValueError("One worker archive must contain exactly one algorithm")
    if {int(row["seed"]) for row in records} != {int(seed)}:
        raise ValueError("One worker archive must contain exactly one seed")
    for record, target in zip(records, schedule):
        if float(record["checkpoint_target_active_seconds"]) != float(
            target["target_active_seconds"]
        ):
            raise ValueError("Snapshot target seconds differ from frozen schedule")
        if float(record["wall_clock_seconds"]) < float(
            target["target_active_seconds"]
        ):
            raise ValueError("Snapshot was captured before its active-time threshold")


__all__ = [
    "augment_snapshot",
    "json_safe",
    "read_json",
    "sha256",
    "snapshot_record",
    "validate_playable_snapshot",
    "validate_records",
    "write_csv",
    "write_json",
]
