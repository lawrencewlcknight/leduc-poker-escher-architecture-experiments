"""Persistence and validation helpers for Experiment 23."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Mapping, Sequence

from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    json_safe,
    read_json,
    sha256,
    validate_playable_snapshot,
    write_csv,
    write_json,
)


def augment_snapshot(
    path: Path,
    *,
    variant_id: str,
    seed: int,
    checkpoint: Mapping,
    nodes_touched: int,
    active_seconds: float,
    completed_iteration: int,
    repository_commit: str,
    config: Mapping,
) -> dict:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload.update(
        {
            "experiment_23_schema_version": 1,
            "variant_id": str(variant_id),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "checkpoint_type": str(checkpoint["checkpoint_type"]),
            "checkpoint_target_active_seconds": checkpoint.get(
                "target_active_seconds"
            ),
            "checkpoint_target_active_hours": checkpoint.get("target_active_hours"),
            "checkpoint_target_nodes": checkpoint.get("target_nodes"),
            "seed": int(seed),
            "nodes_touched": int(nodes_touched),
            "active_seconds": float(active_seconds),
            "completed_iteration": int(completed_iteration),
            "repository_commit": str(repository_commit),
            "frozen_config": json_safe(dict(config)),
        }
    )
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return {
        "variant_id": str(variant_id),
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
        "checkpoint_type": str(checkpoint["checkpoint_type"]),
        "checkpoint_target_active_seconds": checkpoint.get("target_active_seconds"),
        "checkpoint_target_active_hours": checkpoint.get("target_active_hours"),
        "checkpoint_target_nodes": checkpoint.get("target_nodes"),
        "seed": int(seed),
        "nodes_touched": int(nodes_touched),
        "active_seconds": float(active_seconds),
        "completed_iteration": int(completed_iteration),
        "repository_commit": str(repository_commit),
        "relative_path": str(path.relative_to(path.parent.parent)),
        "filename": path.name,
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def validate_records(
    records: Sequence[Mapping],
    *,
    variant_id: str,
    seed: int,
    schedule: Sequence[Mapping],
) -> None:
    expected_ids = [str(row["checkpoint_id"]) for row in schedule]
    observed_ids = [str(row["checkpoint_id"]) for row in records]
    if observed_ids != expected_ids:
        raise ValueError(f"Checkpoint IDs differ: {observed_ids} != {expected_ids}")
    for record, target in zip(records, schedule):
        if record["variant_id"] != variant_id or int(record["seed"]) != int(seed):
            raise ValueError("Snapshot variant or seed differs from worker contract")
        if target["checkpoint_type"] == "active_time":
            if float(record["active_seconds"]) < float(target["target_active_seconds"]):
                raise ValueError("Time snapshot was captured before its threshold")
        elif int(record["nodes_touched"]) < int(target["target_nodes"]):
            raise ValueError("Node snapshot was captured before its threshold")


__all__ = [
    "augment_snapshot",
    "read_json",
    "sha256",
    "validate_playable_snapshot",
    "validate_records",
    "write_csv",
    "write_json",
]
