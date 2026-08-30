"""Run and validate one Experiment 22 variant/seed trajectory."""

from __future__ import annotations

from copy import deepcopy
import gc
import math
from pathlib import Path
import pickle
import resource
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    sha256,
    validate_playable_snapshot,
    write_csv,
    write_json,
)
from vr_deep_cfr.logger import Logger

from .config import (
    EXPERIMENT_NAME,
    FIXED_BETA_ONE,
    TWO_CROSS_FITTED_CRITICS,
    VARIANTS,
    variant_config,
)
from .diagnostics import DiagnosticUCVSolver


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
    "policy_weighted_advantage_abs_mean",
    "full_support_sampling_min_probability",
    "calibration_target_version",
    "q_ensemble_target_version_min",
    "q_ensemble_target_version_max",
    "prediction_gate_player_0",
    "prediction_gate_player_1",
    "prediction_gate_next_player_0",
    "prediction_gate_next_player_1",
    "predictor_relative_skill_player_0",
    "predictor_relative_skill_player_1",
    "predictor_holdout_mse_player_0",
    "predictor_holdout_mse_player_1",
    "predictor_zero_mse_player_0",
    "predictor_zero_mse_player_1",
    "q_fold_0_replay_size",
    "q_fold_1_replay_size",
    "q_fold_2_replay_size",
)


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


def _parse_float(value) -> float:
    return math.nan if value in {None, ""} else float(value)


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
            "max_num_iterations": 3,
            "evaluation_frequency": 1,
            "evaluate_initial_policy": False,
            "early_evaluation_node_thresholds": (),
        }
    )


def _make_solver(seed: int, config: Mapping[str, Any]) -> DiagnosticUCVSolver:
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
    solver = DiagnosticUCVSolver(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def _curve_rows(raw_checkpoints, *, variant_id: str, seed: int) -> list[dict]:
    rows = []
    for checkpoint_index, raw in enumerate(raw_checkpoints):
        row = {
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["variant_label"],
            "seed": int(seed),
            "checkpoint_index": int(checkpoint_index),
            "iteration": int(raw["iteration"]),
            "episode": int(raw["episode"]),
            "nodes_touched": int(raw["nodes_touched"]),
            "wall_clock_seconds": float(raw["wall_clock_seconds"]),
            "exploitability": float(raw["exp"]),
            "average_policy_value": float(raw["average_policy_value"]),
            "average_policy_loss": _parse_float(raw.get("average_policy_loss")),
            "regret_loss_player_0": _parse_float(raw.get("regret_loss_0")),
            "regret_loss_player_1": _parse_float(raw.get("regret_loss_1")),
            "baseline_loss_player_0": _parse_float(raw.get("baseline_loss_0")),
            "baseline_loss_player_1": _parse_float(raw.get("baseline_loss_1")),
            "checkpoint_kind": str(raw.get("checkpoint_kind", "outer_iteration")),
        }
        for field in DIAGNOSTIC_FIELDS:
            row[field] = _parse_float(raw.get(field))
        rows.append(row)
    return rows


def _normalised_auc(rows: list[dict]) -> float:
    selected = [row for row in rows if row["checkpoint_kind"] != "initial_untrained_policy"]
    if len(selected) < 2:
        return math.nan
    x = np.asarray([row["nodes_touched"] for row in selected], dtype=float)
    y = np.asarray([row["exploitability"] for row in selected], dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    return float(np.trapz(y, x) / span) if span > 0.0 else float(y[-1])


def _augment_policy_snapshot(
    path: Path,
    *,
    variant_id: str,
    seed: int,
    target_nodes: int,
    solver,
    config: Mapping[str, Any],
    commit: str,
) -> dict:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    payload.update(
        {
            "experiment_22_schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["variant_label"],
            "repository_commit": commit,
            "target_nodes_touched": int(target_nodes),
            "observed_nodes_touched": int(solver.nodes_touched),
            "completed_iteration": int(solver.num_iteration),
            "frozen_config": dict(config),
        }
    )
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    return {
        "relative_path": str(path.relative_to(path.parent.parent)),
        "filename": path.name,
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
        "variant_id": variant_id,
        "seed": int(seed),
        "nodes_touched": int(solver.nodes_touched),
        "completed_iteration": int(solver.num_iteration),
        "repository_commit": commit,
    }


def _assert_variant(variant_id: str, config: Mapping[str, Any], curves: list[dict]) -> None:
    measured = [row for row in curves if row["unbiased_estimator_sample_count"] > 0.0]
    if variant_id == FIXED_BETA_ONE:
        if not measured or any(
            not np.isclose(row["control_variate_beta_min"], 1.0)
            or not np.isclose(row["control_variate_beta_max"], 1.0)
            for row in measured
        ):
            raise RuntimeError("Fixed-beta arm did not use beta=1 throughout")
    if variant_id == TWO_CROSS_FITTED_CRITICS and int(config["q_ensemble_size"]) != 2:
        raise RuntimeError("Two-critic arm did not instantiate two critic folds")


def run_worker(
    *,
    variant_id: str,
    seed: int,
    target_nodes: int,
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(variant_config(variant_id))
    if smoke:
        _smoke_overrides(config)
    commit = _repository_commit()
    solver = _make_solver(seed, config)
    solver.target_nodes_touched = int(target_nodes)
    try:
        raw_checkpoints = solver.solve()
        if int(solver.nodes_touched) < int(target_nodes):
            raise RuntimeError(
                f"Safety cap reached at {solver.nodes_touched} before {target_nodes} nodes"
            )
        curves = _curve_rows(raw_checkpoints, variant_id=variant_id, seed=seed)
        _assert_variant(variant_id, config, curves)
        final = curves[-1]
        final_window = curves[-min(5, len(curves)) :]
        information_action_rows = solver.information_action_rows()
        beta_histogram_rows = solver.beta_histogram_rows()
        critic_lag_rows = solver.critic_subsequent_regret_rows()

        diagnostics_dir = worker_dir / "diagnostics"
        write_csv(diagnostics_dir / "information_action_diagnostics.csv", information_action_rows)
        write_csv(diagnostics_dir / "beta_histogram.csv", beta_histogram_rows)
        write_csv(
            diagnostics_dir / "critic_error_subsequent_local_regret.csv",
            critic_lag_rows,
        )
        write_csv(worker_dir / "checkpoint_curves.csv", curves)

        snapshot_path = worker_dir / "snapshots" / f"{variant_id}_seed_{seed}_node_15m.pkl"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        save_torch_policy_snapshot(
            solver,
            snapshot_path,
            seed=int(seed),
            iteration=int(solver.num_iteration),
            arm="checkpointed",
            config=dict(config),
            stage_label=f"Experiment 22 {variant_id} at 15M nodes",
            checkpoint_target_nodes=int(target_nodes),
        )
        snapshot = _augment_policy_snapshot(
            snapshot_path,
            variant_id=variant_id,
            seed=seed,
            target_nodes=target_nodes,
            solver=solver,
            config=config,
            commit=commit,
        )
        validate_playable_snapshot(UCV_POLICY_LOADER_ID, snapshot_path)

        summary = {
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["variant_label"],
            "seed": int(seed),
            "target_nodes_touched": int(target_nodes),
            "final_nodes_touched": int(final["nodes_touched"]),
            "node_overshoot": int(final["nodes_touched"]) - int(target_nodes),
            "final_iteration": int(final["iteration"]),
            "final_exploitability": float(final["exploitability"]),
            "best_exploitability": float(min(row["exploitability"] for row in curves)),
            "final_window_mean_exploitability": float(
                np.mean([row["exploitability"] for row in final_window])
            ),
            "node_normalised_auc": _normalised_auc(curves),
            "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
            "peak_rss_mb": _peak_rss_mb(),
            "q_ensemble_size": int(config["q_ensemble_size"]),
            "fixed_control_variate_beta": config.get("fixed_control_variate_beta"),
            "num_information_action_groups": len(information_action_rows),
            "num_critic_lag_rows": len(critic_lag_rows),
        }
        result = {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "variant_id": variant_id,
            "seed": int(seed),
            "smoke": bool(smoke),
            "repository_commit": commit,
            "config": config,
            "summary": summary,
            "snapshot": snapshot,
            "artifacts": {
                "checkpoint_curves": "checkpoint_curves.csv",
                "information_action_diagnostics": (
                    "diagnostics/information_action_diagnostics.csv"
                ),
                "beta_histogram": "diagnostics/beta_histogram.csv",
                "critic_error_subsequent_local_regret": (
                    "diagnostics/critic_error_subsequent_local_regret.csv"
                ),
            },
            "status": "complete",
        }
        write_json(worker_dir / "worker_result.json", result)
        write_json(worker_dir / "SUCCESS.json", {"status": "complete", "snapshot": snapshot})
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["run_worker"]
