"""Uninterrupted Experiment 7 training with node-threshold policy snapshots."""

from __future__ import annotations

import csv
from copy import deepcopy
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from escher_poker.policy_snapshots import (
    load_pickle,
    policy_snapshot_path,
    save_torch_policy_snapshot,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run import (
    DIAGNOSTIC_FIELDS,
)
from vr_deep_cfr.logger import Logger

from .config import CHECKPOINT_SCHEDULE, validate_config


LOGGER = logging.getLogger(__name__)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_float(value) -> float:
    return float("nan") if value in {None, ""} else float(value)


def _solver_kwargs(seed: int, config: Mapping[str, object]) -> Dict[str, Any]:
    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {
        key: value for key, value in deepcopy(dict(config)).items()
        if key not in control_fields
    }
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    return kwargs


def run_seed(
    *,
    seed: int,
    config: Mapping[str, object],
    checkpoint_node_thresholds: Sequence[int],
    run_dir: Path,
) -> dict:
    """Train one uninterrupted seed and capture five fitted policies."""
    import torch

    from unbiased_escher import UnbiasedControlVariateEscher

    validate_config(config, CHECKPOINT_SCHEDULE, checkpoint_node_thresholds)
    thresholds = tuple(int(value) for value in checkpoint_node_thresholds)
    solver = UnbiasedControlVariateEscher(**_solver_kwargs(seed, config))
    solver.target_nodes_touched = thresholds[-1]
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )

    snapshots_dir = Path(run_dir) / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: List[dict] = []
    next_stage_index = 0

    def capture_after_checkpoint(active_solver, raw_checkpoint: Mapping[str, object]):
        nonlocal next_stage_index
        kind = str(raw_checkpoint.get("checkpoint_kind", ""))
        if kind not in {"outer_iteration", "final_node_budget"}:
            return
        nodes = int(active_solver.nodes_touched)
        while (
            next_stage_index < len(thresholds)
            and nodes >= thresholds[next_stage_index]
        ):
            stage = int(CHECKPOINT_SCHEDULE[next_stage_index])
            target_nodes = int(thresholds[next_stage_index])
            snapshot_path = policy_snapshot_path(
                snapshots_dir,
                seed,
                stage,
                "checkpointed",
            )
            save_torch_policy_snapshot(
                active_solver,
                snapshot_path,
                seed=seed,
                iteration=stage,
                arm="checkpointed",
                config=dict(config),
                stage_label=f"first completed iteration crossing {target_nodes} nodes",
                checkpoint_target_nodes=target_nodes,
            )
            row = {
                "seed": int(seed),
                "checkpoint_iteration": stage,
                "checkpoint_stage": stage,
                "checkpoint_fraction": stage / len(thresholds),
                "checkpoint_target_nodes": target_nodes,
                "outer_iteration": int(active_solver.num_iteration),
                "episode": int(active_solver.episode),
                "nodes_touched": nodes,
                "node_threshold_overshoot": nodes - target_nodes,
                "wall_clock_seconds": float(
                    raw_checkpoint.get("wall_clock_seconds", float("nan"))
                ),
                "exploitability_at_capture": _parse_float(
                    raw_checkpoint.get("exp")
                ),
                "average_policy_value_at_capture": _parse_float(
                    raw_checkpoint.get("average_policy_value")
                ),
                "checkpoint_kind": kind,
                "average_policy_buffer_size": len(
                    active_solver.ave_policy_trainer.buffer
                ),
                "advantage_buffer_size_player_0": len(
                    active_solver.regret_trainers[0].buffer
                ),
                "advantage_buffer_size_player_1": len(
                    active_solver.regret_trainers[1].buffer
                ),
                "q_fold_0_replay_size": int(
                    active_solver.q_value_trainer.fold_sizes()[0]
                ),
                "q_fold_1_replay_size": int(
                    active_solver.q_value_trainer.fold_sizes()[1]
                ),
                "q_fold_2_replay_size": int(
                    active_solver.q_value_trainer.fold_sizes()[2]
                ),
                "policy_snapshot": str(snapshot_path.resolve()),
                "policy_snapshot_sha256": _sha256(snapshot_path),
            }
            stage_rows.append(row)
            next_stage_index += 1
            LOGGER.info(
                "Saved seed %s stage %s at iteration %s and %s nodes",
                seed,
                stage,
                row["outer_iteration"],
                nodes,
            )

    try:
        raw_checkpoints = solver.solve(
            post_checkpoint_callback=capture_after_checkpoint
        )
        captured = tuple(row["checkpoint_stage"] for row in stage_rows)
        if captured != CHECKPOINT_SCHEDULE:
            raise RuntimeError(
                f"Seed {seed} captured stages {captured}, expected "
                f"{CHECKPOINT_SCHEDULE}"
            )
        if int(stage_rows[-1]["nodes_touched"]) < thresholds[-1]:
            raise RuntimeError("Final policy snapshot precedes the final node target")

        final_snapshot = load_pickle(stage_rows[-1]["policy_snapshot"])
        live_state = solver.ave_policy_trainer.model.state_dict()
        for name, tensor in final_snapshot["policy_state_dict"].items():
            if not torch.equal(tensor, live_state[name].detach().cpu()):
                raise RuntimeError(
                    "Final saved policy differs from the uninterrupted final policy"
                )

        final_raw = dict(raw_checkpoints[-1])
        centering = _parse_float(
            final_raw.get("policy_weighted_advantage_abs_mean")
        )
        if np.isfinite(centering) and centering > 1e-8:
            raise RuntimeError(f"Policy-centering invariant failed: {centering}")
        minimum_probability = _parse_float(
            final_raw.get("full_support_sampling_min_probability")
        )
        if np.isfinite(minimum_probability) and minimum_probability <= 0:
            raise RuntimeError("Full-support sampling invariant failed")

        raw_rows = []
        for index, raw in enumerate(raw_checkpoints):
            row = {
                "seed": int(seed),
                "checkpoint_index": int(index),
                "iteration": int(raw["iteration"]),
                "episode": int(raw["episode"]),
                "nodes_touched": float(raw["nodes_touched"]),
                "wall_clock_seconds": float(raw["wall_clock_seconds"]),
                "exploitability": float(raw["exp"]),
                "average_policy_value": float(raw["average_policy_value"]),
                "checkpoint_kind": str(raw.get("checkpoint_kind", "")),
                "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
            }
            for field in DIAGNOSTIC_FIELDS:
                row[field] = _parse_float(raw.get(field))
            raw_rows.append(row)
        return {
            "seed": int(seed),
            "stage_rows": stage_rows,
            "checkpoint_rows": raw_rows,
            "final_nodes_touched": int(solver.nodes_touched),
            "final_outer_iteration": int(solver.num_iteration),
        }
    finally:
        del solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = run_seed(
        seed=int(payload["seed"]),
        config=payload["config"],
        checkpoint_node_thresholds=payload["checkpoint_node_thresholds"],
        run_dir=Path(payload["run_dir"]),
    )
    _write_json(output_path, result)
    return 0


def _run_seed_subprocess(
    *,
    seed: int,
    config: Mapping[str, object],
    checkpoint_node_thresholds: Sequence[int],
    run_dir: Path,
) -> dict:
    stem = f"unbiased_escher_checkpoint_head_to_head_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    _write_json(
        input_path,
        {
            "seed": int(seed),
            "config": dict(config),
            "checkpoint_node_thresholds": list(checkpoint_node_thresholds),
            "run_dir": str(run_dir.resolve()),
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker."
        "unbiased_escher_temporal_checkpoint_head_to_head.run",
        "--worker-input-json",
        str(input_path),
        "--worker-output-json",
        str(output_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"Seed {seed} failed; see {log_path}")
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)


def run_training(
    *,
    config: Mapping[str, object],
    seeds: Sequence[int],
    checkpoint_node_thresholds: Sequence[int],
    run_dir: Path,
    continue_on_error: bool,
) -> dict:
    """Run requested seeds sequentially in isolated worker processes."""
    validate_config(config, CHECKPOINT_SCHEDULE, checkpoint_node_thresholds)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_rows: List[dict] = []
    checkpoint_rows: List[dict] = []
    failed: List[dict] = []

    for seed_value in seeds:
        seed = int(seed_value)
        LOGGER.info("Starting Experiment 16 seed %s", seed)
        try:
            result = _run_seed_subprocess(
                seed=seed,
                config=config,
                checkpoint_node_thresholds=checkpoint_node_thresholds,
                run_dir=run_dir,
            )
            stage_rows.extend(result["stage_rows"])
            checkpoint_rows.extend(result["checkpoint_rows"])
            _write_csv(run_dir / "training_stage_metrics.csv", stage_rows)
            _write_csv(run_dir / "training_checkpoint_curves.csv", checkpoint_rows)
        except Exception as exc:
            LOGGER.exception("Seed %s failed", seed)
            failed.append(
                {
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            if not continue_on_error:
                break

    _write_json(run_dir / "failed_seeds.json", failed)
    return {
        "stage_rows": stage_rows,
        "checkpoint_rows": checkpoint_rows,
        "failed": failed,
        "completed_seeds": sorted(
            {
                int(row["seed"])
                for row in stage_rows
                if sum(
                    1
                    for candidate in stage_rows
                    if int(candidate["seed"]) == int(row["seed"])
                )
                == len(CHECKPOINT_SCHEDULE)
            }
        ),
    }


__all__ = ["run_seed", "run_training"]
