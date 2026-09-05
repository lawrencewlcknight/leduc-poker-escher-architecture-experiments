"""Train one selected-architecture Experiment 24 trajectory."""

from __future__ import annotations

from copy import deepcopy
import gc
import logging
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from experiments.leduc_poker.ucv_three_arm_15m_simplification.diagnostics import (
    DiagnosticUCVSolver,
)
from unbiased_escher.stability import StableUnbiasedControlVariateEscher
from vr_deep_cfr.logger import Logger

from .common import (
    augment_snapshot,
    validate_playable_snapshot,
    validate_records,
    write_csv,
    write_json,
)
from .config import CANDIDATE_CONFIG, CANDIDATE_ID, CANDIDATE_LABEL, EXPERIMENT_NAME


LOGGER = logging.getLogger(__name__)
UCV_POLICY_LOADER_ID = "unbiased_control_variate_escher"
DIAGNOSTIC_FIELDS = (
    "calibration_loss",
    "unbiased_estimator_sample_count",
    "control_variate_beta_mean",
    "control_variate_beta_min",
    "control_variate_beta_max",
    "predicted_residual_variance_mean",
    "q_ensemble_disagreement_mean",
    "q_residual_abs_mean",
    "control_residual_abs_mean",
    "importance_correction_abs_mean",
    "full_support_sampling_min_probability",
    "prediction_gate_player_0",
    "prediction_gate_player_1",
    "prediction_gate_next_player_0",
    "prediction_gate_next_player_1",
    "predictor_relative_skill_player_0",
    "predictor_relative_skill_player_1",
    "regret_policy_learning_rate",
    "regret_policy_gradient_clip_norm",
)


class DiagnosticSelectedUCVSolver(
    DiagnosticUCVSolver, StableUnbiasedControlVariateEscher
):
    """Selected UCV implementation with the existing passive diagnostics."""


def _repository_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _smoke_overrides(config: dict) -> None:
    for key in (
        "advantage_network_train_steps",
        "ave_policy_network_train_steps",
        "baseline_network_train_steps",
        "calibration_train_steps",
    ):
        config[key] = 1
    for key in (
        "advantage_batch_size",
        "ave_policy_batch_size",
        "baseline_batch_size",
        "calibration_batch_size",
    ):
        config[key] = 2
    for key in (
        "advantage_buffer_size",
        "ave_policy_buffer_size",
        "baseline_buffer_size",
        "calibration_buffer_size",
    ):
        config[key] = 128
    config.update(
        {
            "num_traversals": 4,
            "max_num_iterations": 8,
            "evaluation_frequency": 1,
            "evaluate_initial_policy": False,
            "early_evaluation_node_thresholds": (),
        }
    )


def _make_solver(seed: int, config: Mapping[str, Any]) -> DiagnosticSelectedUCVSolver:
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
    solver = DiagnosticSelectedUCVSolver(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def _parse_float(value) -> float:
    return math.nan if value in {None, ""} else float(value)


def _curve_rows(raw_rows, *, seed: int) -> list[dict]:
    rows = []
    for index, raw in enumerate(raw_rows):
        row = {
            "candidate_id": CANDIDATE_ID,
            "candidate_label": CANDIDATE_LABEL,
            "seed": int(seed),
            "checkpoint_index": int(index),
            "iteration": int(raw["iteration"]),
            "episode": int(raw["episode"]),
            "nodes_touched": int(raw["nodes_touched"]),
            "wall_clock_seconds": float(raw["wall_clock_seconds"]),
            "exploitability": float(raw["exp"]),
            "average_policy_value": float(raw["average_policy_value"]),
            "average_policy_loss": _parse_float(raw.get("average_policy_loss")),
            "regret_loss_player_0": _parse_float(raw.get("regret_loss_0")),
            "regret_loss_player_1": _parse_float(raw.get("regret_loss_1")),
            "checkpoint_kind": str(raw.get("checkpoint_kind", "outer_iteration")),
        }
        for field in DIAGNOSTIC_FIELDS:
            row[field] = _parse_float(raw.get(field))
        rows.append(row)
    return rows


def _target_reached(target: Mapping, *, active_seconds: float, nodes: int) -> bool:
    if target["checkpoint_type"] == "active_time":
        return active_seconds >= float(target["target_active_seconds"])
    return nodes >= int(target["target_nodes"])


def _save_checkpoint(
    *,
    solver,
    seed: int,
    target: Mapping,
    raw_checkpoint: Mapping,
    config: Mapping,
    worker_dir: Path,
    commit: str,
) -> dict:
    checkpoint_id = str(target["checkpoint_id"])
    path = worker_dir / "snapshots" / f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_policy_snapshot(
        solver,
        path,
        seed=int(seed),
        iteration=int(raw_checkpoint["iteration"]),
        arm="checkpointed",
        config=dict(config),
        stage_label=f"Experiment 24 {CANDIDATE_ID} {checkpoint_id}",
        checkpoint_target_nodes=(
            int(target["target_nodes"])
            if target.get("target_nodes") is not None
            else int(raw_checkpoint["nodes_touched"])
        ),
    )
    return augment_snapshot(
        path,
        candidate_id=CANDIDATE_ID,
        seed=seed,
        checkpoint=target,
        nodes_touched=int(raw_checkpoint["nodes_touched"]),
        active_seconds=float(raw_checkpoint["active_seconds"]),
        completed_iteration=int(raw_checkpoint["iteration"]),
        repository_commit=commit,
        config=config,
    )


def _assert_selected_config(config: Mapping, curves: Sequence[Mapping]) -> None:
    if config.get("fixed_control_variate_beta") != 1.0:
        raise RuntimeError("Experiment 24 must use fixed beta=1")
    if int(config.get("q_ensemble_size", -1)) != 2:
        raise RuntimeError("Experiment 24 must use two cross-fitted critics")
    if config.get("use_instantaneous_predictor") is not False:
        raise RuntimeError("Experiment 24 must disable the instantaneous predictor")
    if config.get("use_residual_calibration") is not True:
        raise RuntimeError("Experiment 24 must retain residual calibration")
    if any(config.get(key) is not None for key in (
        "regret_policy_gradient_clip_norm",
        "anneal_start_nodes",
        "anneal_end_nodes",
        "anneal_final_learning_rate",
    )):
        raise RuntimeError("Experiment 24 must not use the rejected stability package")
    measured = [row for row in curves if row["unbiased_estimator_sample_count"] > 0]
    if measured and any(
        not np.isclose(row["control_variate_beta_min"], 1.0)
        or not np.isclose(row["control_variate_beta_max"], 1.0)
        for row in measured
    ):
        raise RuntimeError("Observed control-variate beta differed from one")
    if measured and any(
        not np.isclose(row["prediction_gate_player_0"], 0.0)
        or not np.isclose(row["prediction_gate_player_1"], 0.0)
        for row in measured
    ):
        raise RuntimeError("Disabled instantaneous predictor was activated")


def run_worker(
    *,
    seed: int,
    schedule: Sequence[Mapping],
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(CANDIDATE_CONFIG)
    if smoke:
        _smoke_overrides(config)
    commit = _repository_commit()
    solver = _make_solver(seed, config)
    solver.target_nodes_touched = 2**62
    captured: dict[str, dict] = {}
    excluded_snapshot_seconds = 0.0

    def capture(active_solver, raw_checkpoint):
        nonlocal excluded_snapshot_seconds
        active_seconds = max(
            0.0,
            float(raw_checkpoint["wall_clock_seconds"]) - excluded_snapshot_seconds,
        )
        checkpoint = dict(raw_checkpoint)
        checkpoint["active_seconds"] = active_seconds
        for target in schedule:
            checkpoint_id = str(target["checkpoint_id"])
            if checkpoint_id in captured or not _target_reached(
                target,
                active_seconds=active_seconds,
                nodes=int(checkpoint["nodes_touched"]),
            ):
                continue
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
            LOGGER.info(
                "Saved %s seed %s at iteration %s, %s nodes, %.1f active seconds",
                checkpoint_id,
                seed,
                checkpoint["iteration"],
                checkpoint["nodes_touched"],
                active_seconds,
            )
        if len(captured) == len(schedule):
            active_solver.target_nodes_touched = int(active_solver.nodes_touched)

    try:
        raw_curves = solver.solve(post_checkpoint_callback=capture)
        if len(captured) != len(schedule):
            missing = [
                row["checkpoint_id"]
                for row in schedule
                if row["checkpoint_id"] not in captured
            ]
            raise RuntimeError(f"Safety cap reached with missing checkpoints: {missing}")
        records = [captured[str(target["checkpoint_id"])] for target in schedule]
        validate_records(
            records,
            candidate_id=CANDIDATE_ID,
            seed=seed,
            schedule=schedule,
        )
        for record in records:
            validate_playable_snapshot(
                UCV_POLICY_LOADER_ID, worker_dir / record["relative_path"]
            )
        curves = _curve_rows(raw_curves, seed=seed)
        _assert_selected_config(config, curves)
        diagnostics_dir = worker_dir / "diagnostics"
        write_csv(worker_dir / "checkpoint_curves.csv", curves)
        write_csv(
            diagnostics_dir / "information_action_diagnostics.csv",
            solver.information_action_rows(),
        )
        write_csv(diagnostics_dir / "beta_histogram.csv", solver.beta_histogram_rows())
        write_csv(
            diagnostics_dir / "critic_error_subsequent_local_regret.csv",
            solver.critic_subsequent_regret_rows(),
        )
        result = {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "candidate_id": CANDIDATE_ID,
            "candidate_label": CANDIDATE_LABEL,
            "seed": int(seed),
            "smoke": bool(smoke),
            "checkpoint_schedule": list(schedule),
            "repository_commit": commit,
            "config": config,
            "snapshots": records,
            "peak_rss_mb": _peak_rss_mb(),
            "artifacts": {
                "checkpoint_curves": "checkpoint_curves.csv",
                "information_action_diagnostics": "diagnostics/information_action_diagnostics.csv",
                "beta_histogram": "diagnostics/beta_histogram.csv",
                "critic_error_subsequent_local_regret": "diagnostics/critic_error_subsequent_local_regret.csv",
            },
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


__all__ = ["run_worker"]
