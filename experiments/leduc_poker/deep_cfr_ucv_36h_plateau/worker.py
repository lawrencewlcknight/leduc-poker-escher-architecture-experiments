"""Train one UCV-ESCHER trajectory and archive every two-hour policy."""

from __future__ import annotations

from copy import deepcopy
import gc
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import torch

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from vr_deep_cfr.logger import Logger

from .common import (
    augment_snapshot,
    validate_playable_snapshot,
    validate_records,
    write_json,
)
from .config import ALGORITHMS, EXPERIMENT_NAME, UCV_CONFIG, UCV_ESCHER


LOGGER = logging.getLogger(__name__)


def _repository_commit(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _smoke_overrides(config: dict) -> None:
    for key in (
        "advantage_network_train_steps",
        "ave_policy_network_train_steps",
        "baseline_network_train_steps",
        "calibration_train_steps",
    ):
        if key in config:
            config[key] = 1
    for key in (
        "advantage_batch_size",
        "ave_policy_batch_size",
        "baseline_batch_size",
        "calibration_batch_size",
    ):
        if key in config:
            config[key] = 2
    for key in (
        "advantage_buffer_size",
        "ave_policy_buffer_size",
        "baseline_buffer_size",
        "calibration_buffer_size",
    ):
        if key in config:
            config[key] = 128
    config.update(
        {
            "num_traversals": 2,
            "max_num_iterations": 3,
            "evaluation_frequency": 1,
            "evaluate_initial_policy": False,
            "early_evaluation_node_thresholds": (),
        }
    )


def _make_solver(seed: int, config: Mapping[str, Any]):
    from unbiased_escher import UnbiasedControlVariateEscher

    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=2 * int(config["num_traversals"]) * int(config["max_num_iterations"]),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    solver = UnbiasedControlVariateEscher(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def _save_checkpoint(
    *,
    solver,
    seed: int,
    target: Mapping[str, Any],
    raw_checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    worker_dir: Path,
    commit: str,
) -> dict:
    checkpoint_id = str(target["checkpoint_id"])
    snapshots_dir = worker_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    path = snapshots_dir / f"{UCV_ESCHER}_seed_{seed}_{checkpoint_id}.pkl"
    iteration = int(raw_checkpoint["iteration"])
    nodes = int(raw_checkpoint["nodes_touched"])
    wall_clock = float(raw_checkpoint["wall_clock_seconds"])
    save_torch_policy_snapshot(
        solver,
        path,
        seed=seed,
        iteration=iteration,
        arm="checkpointed",
        config=dict(config),
        stage_label=f"Experiment 21 checkpoint {checkpoint_id}",
        checkpoint_target_nodes=nodes,
    )
    record = augment_snapshot(
        path,
        algorithm_id=UCV_ESCHER,
        seed=seed,
        checkpoint=target,
        nodes_touched=nodes,
        wall_clock_seconds=wall_clock,
        completed_iteration=iteration,
        repository_commit=commit,
        config=config,
    )
    LOGGER.info(
        "Saved %s seed %s at iteration %s, %s nodes, %.1f active seconds",
        checkpoint_id,
        seed,
        iteration,
        nodes,
        wall_clock,
    )
    return record


def run_ucv_worker(
    *,
    seed: int,
    schedule: Sequence[Mapping[str, Any]],
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    commit = _repository_commit(repository)
    config = deepcopy(UCV_CONFIG)
    if smoke:
        _smoke_overrides(config)
    solver = _make_solver(seed, config)
    solver.target_nodes_touched = 2**62
    captured: dict[str, dict] = {}
    excluded_snapshot_seconds = 0.0

    def capture(active_solver, raw_checkpoint):
        nonlocal excluded_snapshot_seconds
        elapsed = max(
            0.0,
            float(raw_checkpoint["wall_clock_seconds"]) - excluded_snapshot_seconds,
        )
        checkpoint = dict(raw_checkpoint)
        checkpoint["wall_clock_seconds"] = elapsed
        for target in schedule:
            checkpoint_id = str(target["checkpoint_id"])
            if checkpoint_id in captured:
                continue
            if elapsed < float(target["target_active_seconds"]):
                break
            save_start = time.perf_counter()
            captured[checkpoint_id] = _save_checkpoint(
                solver=active_solver,
                seed=seed,
                target=target,
                raw_checkpoint=checkpoint,
                config=config,
                worker_dir=worker_dir,
                commit=commit,
            )
            excluded_snapshot_seconds += time.perf_counter() - save_start
        if len(captured) == len(schedule):
            active_solver.target_nodes_touched = int(active_solver.nodes_touched)

    try:
        curves = solver.solve(post_checkpoint_callback=capture)
        if len(captured) != len(schedule):
            raise RuntimeError("UCV safety cap reached without every active-time checkpoint")
        records = [captured[str(target["checkpoint_id"])] for target in schedule]
        validate_records(
            records,
            algorithm_id=UCV_ESCHER,
            seed=int(seed),
            schedule=schedule,
        )
        for record in records:
            validate_playable_snapshot(
                UCV_ESCHER,
                worker_dir / record["relative_path"],
            )
        result = {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": UCV_ESCHER,
            "algorithm_label": ALGORITHMS[UCV_ESCHER]["algorithm_label"],
            "seed": int(seed),
            "smoke": bool(smoke),
            "checkpoint_schedule": list(schedule),
            "repository_commit": commit,
            "snapshots": records,
            "checkpoint_rows": curves,
            "status": "complete",
        }
        write_json(worker_dir / "worker_result.json", result)
        write_json(
            worker_dir / "SUCCESS.json",
            {"status": "complete", "snapshots": records},
        )
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["run_ucv_worker"]
