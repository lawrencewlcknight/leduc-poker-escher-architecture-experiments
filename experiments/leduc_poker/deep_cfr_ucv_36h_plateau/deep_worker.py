#!/usr/bin/env python3
"""Isolated Deep CFR worker for Experiment 21."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


# Do not import the architecture repository's top-level ``experiments`` package
# here. This worker inserts the Deep CFR repository before importing its own
# experiment configuration; importing the architecture package first would
# make resolution depend on package-cache order.
DEEP_CFR = "deep_cfr"
EXPERIMENT_NAME = "deep_cfr_ucv_36h_plateau"
MAX_DEEP_CFR_ITERATIONS = 18_000


class _CheckpointsComplete(Exception):
    """Stop cleanly after the last completed-iteration snapshot."""


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except ImportError:  # pragma: no cover - NumPy is a dependency
        pass
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _augment_snapshot(
    path: Path,
    *,
    torch,
    seed: int,
    checkpoint: Mapping[str, Any],
    nodes_touched: int,
    wall_clock_seconds: float,
    completed_iteration: int,
    repository_commit: str,
    config: Mapping[str, Any],
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(
        {
            "convergence_experiment_schema_version": 1,
            "convergence_algorithm_id": DEEP_CFR,
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
            "frozen_config": _json_safe(dict(config)),
        }
    )
    torch.save(payload, path)
    return {
        "algorithm_id": DEEP_CFR,
        "checkpoint_id": str(checkpoint["checkpoint_id"]),
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
        "relative_path": str(path.relative_to(path.parent.parent)),
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    schedule: Sequence[Mapping[str, Any]],
) -> None:
    expected_ids = [str(row["checkpoint_id"]) for row in schedule]
    observed_ids = [str(row["checkpoint_id"]) for row in records]
    if observed_ids != expected_ids:
        raise ValueError(
            f"Checkpoint IDs differ: expected {expected_ids}, observed {observed_ids}"
        )
    if {int(row["seed"]) for row in records} != {int(seed)}:
        raise ValueError("One worker archive must contain exactly one seed")
    for record, target in zip(records, schedule):
        if float(record["wall_clock_seconds"]) < float(
            target["target_active_seconds"]
        ):
            raise ValueError("Snapshot was captured before its active-time threshold")


def _repository_commit(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _smoke_overrides(config: dict) -> None:
    config.update(
        {
            "num_iterations": 6,
            "num_traversals": 2,
            "evaluation_interval": 1,
            "policy_network_train_every": 1,
            "policy_network_train_steps": 1,
            "advantage_network_train_steps": 1,
            "batch_size_advantage": 2,
            "batch_size_strategy": 2,
            "memory_capacity": 128,
            "compute_exploitability": False,
        }
    )


def run_deep_worker(
    *,
    deep_cfr_repo: Path,
    seed: int,
    schedule: Sequence[Mapping[str, Any]],
    worker_dir: Path,
    smoke: bool,
) -> dict:
    deep_cfr_repo = Path(deep_cfr_repo).resolve()
    if not (deep_cfr_repo / "deep_cfr_poker" / "solver.py").is_file():
        raise FileNotFoundError(f"Not a Deep CFR repository: {deep_cfr_repo}")
    sys.path.insert(0, str(deep_cfr_repo))

    import numpy as np
    import pyspiel
    import torch
    from open_spiel.python import policy

    from deep_cfr_poker.experiment_utils import make_solver
    from deep_cfr_poker.seeding import set_seed
    from deep_cfr_poker.snapshots import LoadedPolicy
    from experiments.leduc_poker.deep_cfr_final_candidate_checkpoint_head_to_head.config import (
        DEFAULT_CONFIG,
    )

    worker_dir = Path(worker_dir).resolve()
    snapshots_dir = worker_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(dict(DEFAULT_CONFIG))
    config.update(
        {
            "experiment_name": EXPERIMENT_NAME,
            "num_iterations": MAX_DEEP_CFR_ITERATIONS,
        }
    )
    if smoke:
        _smoke_overrides(config)

    commit = _repository_commit(deep_cfr_repo)
    set_seed(int(seed))
    game = pyspiel.load_game(str(config["game_name"]))
    solver = make_solver(game, config)
    captured: dict[str, dict] = {}
    training_start = time.perf_counter()
    excluded_snapshot_seconds = 0.0

    def save_checkpoint(active_solver, target: Mapping[str, Any], elapsed: float) -> None:
        checkpoint_id = str(target["checkpoint_id"])
        completed_iteration = int(active_solver._iteration)
        nodes = int(active_solver._nodes_touched)
        path = snapshots_dir / f"{DEEP_CFR}_seed_{seed}_{checkpoint_id}.pt"
        active_solver.save_policy_snapshot(
            path,
            seed=int(seed),
            target_iteration=completed_iteration,
            stage_label=f"Experiment 21 checkpoint {checkpoint_id}",
            experiment_name=EXPERIMENT_NAME,
            game_name=str(config["game_name"]),
            solver_config=dict(config),
        )
        captured[checkpoint_id] = _augment_snapshot(
            path,
            torch=torch,
            seed=int(seed),
            checkpoint=target,
            nodes_touched=nodes,
            wall_clock_seconds=elapsed,
            completed_iteration=completed_iteration,
            repository_commit=commit,
            config=config,
        )

    def capture(active_solver, _callback_iteration: int) -> None:
        nonlocal excluded_snapshot_seconds
        elapsed = max(
            0.0,
            float(time.perf_counter() - training_start) - excluded_snapshot_seconds,
        )
        for target in schedule:
            checkpoint_id = str(target["checkpoint_id"])
            if checkpoint_id in captured:
                continue
            if elapsed < float(target["target_active_seconds"]):
                break
            save_start = time.perf_counter()
            save_checkpoint(active_solver, target, elapsed)
            excluded_snapshot_seconds += time.perf_counter() - save_start
        if len(captured) == len(schedule):
            raise _CheckpointsComplete

    try:
        try:
            solver.solve(post_iteration_callback=capture)
        except _CheckpointsComplete:
            pass
        if len(captured) != len(schedule):
            raise RuntimeError(
                "Deep CFR safety cap reached without every active-time checkpoint"
            )
        records = [captured[str(target["checkpoint_id"])] for target in schedule]
        _validate_records(
            records,
            seed=int(seed),
            schedule=schedule,
        )
        for record in records:
            snapshot_path = worker_dir / record["relative_path"]
            loaded = LoadedPolicy(game, snapshot_path)
            tabular = policy.tabular_policy_from_callable(
                game, loaded.action_probabilities
            )
            values = np.asarray(tabular.action_probability_array, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite probabilities in {snapshot_path}")
        result = {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": DEEP_CFR,
            "algorithm_label": "Deep CFR",
            "seed": int(seed),
            "smoke": bool(smoke),
            "checkpoint_schedule": list(schedule),
            "repository_commit": commit,
            "snapshots": records,
            "status": "complete",
        }
        _write_json(worker_dir / "worker_result.json", result)
        _write_json(
            worker_dir / "SUCCESS.json",
            {"status": "complete", "snapshots": records},
        )
        return result
    finally:
        close = getattr(solver, "close", None)
        if callable(close):
            close()
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-cfr-repo", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule-json", required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_deep_worker(
        deep_cfr_repo=args.deep_cfr_repo,
        seed=args.seed,
        schedule=tuple(json.loads(args.schedule_json)),
        worker_dir=args.worker_dir,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
