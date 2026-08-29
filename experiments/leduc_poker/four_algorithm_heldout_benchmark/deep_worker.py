#!/usr/bin/env python3
"""Isolated Deep CFR worker.

This file deliberately runs in a separate interpreter.  The architecture and
Deep CFR repositories both contain a top-level ``experiments`` package, so
loading both experiment trees in one interpreter would make configuration
provenance depend on import order.
"""

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
from typing import Any, Mapping


DEEP_CFR = "deep_cfr"
NODE_ENDPOINT = "node_15m"
TIME_ENDPOINT = "time_11h"
EXPERIMENT_NAME = "four_algorithm_heldout_benchmark"


class _EndpointsComplete(Exception):
    """Internal control flow used to stop after both snapshots exist."""


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except ImportError:  # pragma: no cover - NumPy is a project dependency
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
            "num_iterations": 3,
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


def _augment_and_record(
    path: Path,
    *,
    torch,
    endpoint_id: str,
    seed: int,
    nodes_touched: int,
    wall_clock_seconds: float,
    completed_iteration: int,
    repository_commit: str,
    config: Mapping[str, Any],
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload.update(
        {
            "heldout_benchmark_schema_version": 1,
            "heldout_algorithm_id": DEEP_CFR,
            "heldout_endpoint_id": endpoint_id,
            "seed": int(seed),
            "nodes_touched": int(nodes_touched),
            "wall_clock_seconds": float(wall_clock_seconds),
            "heldout_completed_iteration": int(completed_iteration),
            "repository_commit": repository_commit,
            "frozen_config": _json_safe(dict(config)),
        }
    )
    torch.save(payload, path)
    return {
        "algorithm_id": DEEP_CFR,
        "endpoint_id": endpoint_id,
        "seed": int(seed),
        "nodes_touched": int(nodes_touched),
        "wall_clock_seconds": float(wall_clock_seconds),
        "completed_iteration": int(completed_iteration),
        "repository_commit": repository_commit,
        "relative_path": str(path.relative_to(path.parent.parent)),
        "filename": path.name,
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def run_deep_worker(
    *,
    deep_cfr_repo: Path,
    seed: int,
    target_nodes: int,
    target_seconds: float,
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
            # Safety cap only: the callback stops after both endpoints exist.
            "num_iterations": 6_000,
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

    def save_endpoint(active_solver, endpoint_id: str, elapsed: float) -> None:
        completed_iteration = int(active_solver._iteration)
        nodes = int(active_solver._nodes_touched)
        path = snapshots_dir / f"{DEEP_CFR}_seed_{seed}_{endpoint_id}.pt"
        active_solver.save_policy_snapshot(
            path,
            seed=int(seed),
            target_iteration=completed_iteration,
            stage_label=f"held-out benchmark endpoint {endpoint_id}",
            experiment_name=EXPERIMENT_NAME,
            game_name=str(config["game_name"]),
            solver_config=dict(config),
        )
        captured[endpoint_id] = _augment_and_record(
            path,
            torch=torch,
            endpoint_id=endpoint_id,
            seed=int(seed),
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
        nodes = int(active_solver._nodes_touched)
        if NODE_ENDPOINT not in captured and nodes >= int(target_nodes):
            save_start = time.perf_counter()
            save_endpoint(active_solver, NODE_ENDPOINT, elapsed)
            excluded_snapshot_seconds += time.perf_counter() - save_start
        if TIME_ENDPOINT not in captured and elapsed >= float(target_seconds):
            save_start = time.perf_counter()
            save_endpoint(active_solver, TIME_ENDPOINT, elapsed)
            excluded_snapshot_seconds += time.perf_counter() - save_start
        if len(captured) == 2:
            raise _EndpointsComplete

    try:
        try:
            solver.solve(post_iteration_callback=capture)
        except _EndpointsComplete:
            pass
        if len(captured) != 2:
            raise RuntimeError(
                f"Safety cap reached without both endpoints; captured={sorted(captured)}"
            )

        records = [captured[NODE_ENDPOINT], captured[TIME_ENDPOINT]]
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
            "algorithm_id": DEEP_CFR,
            "algorithm_label": "Deep CFR",
            "seed": int(seed),
            "smoke": bool(smoke),
            "target_nodes": int(target_nodes),
            "target_active_seconds": float(target_seconds),
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
    parser.add_argument("--target-nodes", type=int, required=True)
    parser.add_argument("--target-seconds", type=float, required=True)
    parser.add_argument("--worker-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_deep_worker(
        deep_cfr_repo=args.deep_cfr_repo,
        seed=args.seed,
        target_nodes=args.target_nodes,
        target_seconds=args.target_seconds,
        worker_dir=args.worker_dir,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
