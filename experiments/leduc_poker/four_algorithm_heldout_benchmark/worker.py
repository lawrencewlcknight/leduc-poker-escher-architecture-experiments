"""Train one architecture-repository algorithm and capture both endpoints."""

from __future__ import annotations

from copy import deepcopy
import gc
import logging
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import torch

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from vr_deep_cfr.logger import Logger
from vr_deep_cfr.policy_snapshots import save_policy_snapshot as save_vr_policy_snapshot

from .common import (
    augment_snapshot,
    validate_endpoint_records,
    validate_playable_snapshot,
    write_json,
)
from .config import (
    ALGORITHMS,
    NODE_ENDPOINT,
    TIME_ENDPOINT,
    UCV_CONFIG,
    UCV_ESCHER,
    VR_CONFIG,
    VR_DEEP_DCFR_PLUS,
    VR_DEEP_PDCFR_PLUS,
)


LOGGER = logging.getLogger(__name__)


def repository_commit(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
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


def _make_vr_solver(algorithm_id: str, seed: int, config: Mapping[str, Any]):
    from vr_deep_cfr import VRDeepDCFRPlus, VRDeepPDCFRPlus

    solver_class = {
        VR_DEEP_DCFR_PLUS: VRDeepDCFRPlus,
        VR_DEEP_PDCFR_PLUS: VRDeepPDCFRPlus,
    }[algorithm_id]
    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=2 * int(config["num_traversals"]) * int(config["max_num_iterations"]),
        alpha=2.0 if algorithm_id == VR_DEEP_DCFR_PLUS else 2.3,
        gamma=2.0,
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    if algorithm_id == VR_DEEP_PDCFR_PLUS:
        kwargs["reinitialize_imm_regret_networks"] = True
    solver = solver_class(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def _make_ucv_solver(seed: int, config: Mapping[str, Any]):
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


def _save_endpoint(
    *,
    solver,
    algorithm_id: str,
    seed: int,
    endpoint_id: str,
    raw_checkpoint: Mapping[str, Any],
    config: Mapping[str, Any],
    worker_dir: Path,
    commit: str,
) -> dict:
    snapshots_dir = worker_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".pkl" if algorithm_id == UCV_ESCHER else ".pt"
    path = snapshots_dir / f"{algorithm_id}_seed_{seed}_{endpoint_id}{suffix}"
    iteration = int(raw_checkpoint["iteration"])
    nodes = int(raw_checkpoint["nodes_touched"])
    wall_clock = float(raw_checkpoint["wall_clock_seconds"])
    if algorithm_id == UCV_ESCHER:
        save_torch_policy_snapshot(
            solver,
            path,
            seed=seed,
            iteration=iteration,
            arm="checkpointed",
            config=dict(config),
            stage_label=f"held-out benchmark endpoint {endpoint_id}",
            checkpoint_target_nodes=nodes,
        )
    else:
        save_vr_policy_snapshot(
            solver,
            path,
            algorithm_id=algorithm_id,
            algorithm_label=ALGORITHMS[algorithm_id]["algorithm_label"],
            seed=seed,
            config=dict(config),
        )
    record = augment_snapshot(
        path,
        algorithm_id=algorithm_id,
        seed=seed,
        endpoint_id=endpoint_id,
        nodes_touched=nodes,
        wall_clock_seconds=wall_clock,
        completed_iteration=iteration,
        repository_commit=commit,
        config=config,
    )
    LOGGER.info(
        "Saved %s %s seed %s at iteration %s, %s nodes, %.1f seconds",
        algorithm_id,
        endpoint_id,
        seed,
        iteration,
        nodes,
        wall_clock,
    )
    return record


def run_architecture_worker(
    *,
    algorithm_id: str,
    seed: int,
    target_nodes: int,
    target_seconds: float,
    worker_dir: Path,
    smoke: bool,
) -> dict:
    if algorithm_id not in {VR_DEEP_DCFR_PLUS, VR_DEEP_PDCFR_PLUS, UCV_ESCHER}:
        raise ValueError(f"Unsupported architecture worker algorithm: {algorithm_id}")
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[3]
    commit = repository_commit(repository)
    config = deepcopy(UCV_CONFIG if algorithm_id == UCV_ESCHER else VR_CONFIG)
    if smoke:
        _smoke_overrides(config)
    solver = (
        _make_ucv_solver(seed, config)
        if algorithm_id == UCV_ESCHER
        else _make_vr_solver(algorithm_id, seed, config)
    )
    # Neither endpoint alone is a stopping rule. Once both have been captured,
    # the callback lowers this sentinel so the inherited solve loop exits.
    solver.target_nodes_touched = 2**62
    captured: dict[str, dict] = {}
    excluded_snapshot_seconds = 0.0

    def capture(active_solver, raw_checkpoint):
        nonlocal excluded_snapshot_seconds
        nodes = int(raw_checkpoint["nodes_touched"])
        elapsed = max(
            0.0,
            float(raw_checkpoint["wall_clock_seconds"]) - excluded_snapshot_seconds,
        )
        checkpoint = dict(raw_checkpoint)
        checkpoint["wall_clock_seconds"] = elapsed
        if NODE_ENDPOINT not in captured and nodes >= int(target_nodes):
            save_start = time.perf_counter()
            captured[NODE_ENDPOINT] = _save_endpoint(
                solver=active_solver,
                algorithm_id=algorithm_id,
                seed=seed,
                endpoint_id=NODE_ENDPOINT,
                raw_checkpoint=checkpoint,
                config=config,
                worker_dir=worker_dir,
                commit=commit,
            )
            excluded_snapshot_seconds += time.perf_counter() - save_start
        if TIME_ENDPOINT not in captured and elapsed >= float(target_seconds):
            save_start = time.perf_counter()
            captured[TIME_ENDPOINT] = _save_endpoint(
                solver=active_solver,
                algorithm_id=algorithm_id,
                seed=seed,
                endpoint_id=TIME_ENDPOINT,
                raw_checkpoint=checkpoint,
                config=config,
                worker_dir=worker_dir,
                commit=commit,
            )
            excluded_snapshot_seconds += time.perf_counter() - save_start
        if len(captured) == 2:
            active_solver.target_nodes_touched = int(active_solver.nodes_touched)

    try:
        curves = solver.solve(post_checkpoint_callback=capture)
        if len(captured) != 2:
            raise RuntimeError(
                f"Safety cap reached without both endpoints; captured={sorted(captured)}"
            )
        records = [captured[NODE_ENDPOINT], captured[TIME_ENDPOINT]]
        validate_endpoint_records(records)
        for record in records:
            validate_playable_snapshot(
                algorithm_id,
                worker_dir / record["relative_path"],
            )
        result = {
            "schema_version": 1,
            "algorithm_id": algorithm_id,
            "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
            "seed": int(seed),
            "smoke": bool(smoke),
            "target_nodes": int(target_nodes),
            "target_active_seconds": float(target_seconds),
            "repository_commit": commit,
            "snapshots": records,
            "checkpoint_rows": curves,
            "status": "complete",
        }
        write_json(worker_dir / "worker_result.json", result)
        write_json(worker_dir / "SUCCESS.json", {"status": "complete", "snapshots": records})
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["run_architecture_worker"]
