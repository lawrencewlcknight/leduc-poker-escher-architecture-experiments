"""Train one Experiment 29 promoted-candidate trajectory."""

from __future__ import annotations

from copy import deepcopy
import gc
import logging
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    build_training_state,
    read_training_state,
    restore_training_state,
    save_training_state,
)
from experiments.leduc_poker.ucv_residual_target_factorial.worker import (
    DIAGNOSTIC_FIELDS,
    DiagnosticFactorialUCVSolver,
)
from unbiased_escher.policy_distillation import (
    SOFT_TARGET_CROSS_ENTROPY,
    SoftTargetCrossEntropyAvePolicyTrainer,
)
from vr_deep_cfr.logger import Logger

from .common import (
    augment_snapshot,
    sha256,
    validate_playable_snapshot,
    validate_records,
    write_csv,
    write_json,
)
from .config import (
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    CANDIDATE_LABEL,
    EXPERIMENT_NAME,
    training_state_checkpoint_ids,
)


LOGGER = logging.getLogger(__name__)
UCV_POLICY_LOADER_ID = "unbiased_control_variate_escher"
TRAINING_STATE_TYPE = "experiment_29_full_training_state"


class DiagnosticPromotedUCVSolver(DiagnosticFactorialUCVSolver):
    """Averaged-target UCV with reset soft-target CE policy distillation."""

    def __init__(
        self,
        *args,
        average_policy_loss: str = SOFT_TARGET_CROSS_ENTROPY,
        average_policy_reset_each_fit: bool = True,
        **kwargs,
    ):
        if str(average_policy_loss) != SOFT_TARGET_CROSS_ENTROPY:
            raise ValueError("Experiment 29 requires soft-target cross-entropy")
        if not bool(average_policy_reset_each_fit):
            raise ValueError("Experiment 29 requires reset average-policy fits")
        self.average_policy_loss = str(average_policy_loss)
        self.average_policy_reset_each_fit = True
        super().__init__(*args, **kwargs)

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = SoftTargetCrossEntropyAvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.device,
            self.gamma,
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


def _make_solver(seed: int, config: Mapping[str, Any]) -> DiagnosticPromotedUCVSolver:
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
    solver = DiagnosticPromotedUCVSolver(**kwargs)
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


def _save_policy(
    *,
    solver,
    seed: int,
    target: Mapping,
    checkpoint: Mapping,
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
        iteration=int(checkpoint["iteration"]),
        arm="checkpointed",
        config=dict(config),
        stage_label=f"Experiment 29 {CANDIDATE_ID} {checkpoint_id}",
        checkpoint_target_nodes=(
            int(target["target_nodes"])
            if target.get("target_nodes") is not None
            else int(checkpoint["nodes_touched"])
        ),
    )
    return augment_snapshot(
        path,
        candidate_id=CANDIDATE_ID,
        seed=seed,
        checkpoint=target,
        nodes_touched=int(checkpoint["nodes_touched"]),
        active_seconds=float(checkpoint["active_seconds"]),
        completed_iteration=int(checkpoint["iteration"]),
        repository_commit=commit,
        config=config,
    )


def _training_state_paths(worker_dir: Path, *, seed: int, smoke: bool) -> list[Path]:
    return [
        worker_dir / "training_states" / f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pt"
        for checkpoint_id in reversed(training_state_checkpoint_ids(smoke=smoke))
    ]


def _sync_remote_resume_point(worker_dir: Path) -> None:
    remote = os.environ.get("EXP29_REMOTE_TASK_URI")
    if not remote:
        return
    subprocess.run(
        ["gcloud", "storage", "rsync", "--recursive", str(worker_dir.resolve()), remote.rstrip("/")],
        check=True,
    )


def _restore_latest(
    solver,
    *,
    worker_dir: Path,
    seed: int,
    smoke: bool,
    commit: str,
    config: Mapping,
) -> dict[str, dict]:
    for path in _training_state_paths(worker_dir, seed=seed, smoke=smoke):
        if not path.is_file():
            continue
        payload = read_training_state(path)
        records = restore_training_state(
            solver,
            payload,
            variant_id=CANDIDATE_ID,
            seed=seed,
            repository_commit=commit,
            config=config,
            expected_state_type=TRAINING_STATE_TYPE,
        )
        for record in records:
            snapshot = worker_dir / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Restored snapshot is missing or corrupt: {snapshot}")
        LOGGER.info(
            "Resuming %s seed %s from %s at iteration %s and %.2f active hours",
            CANDIDATE_ID,
            seed,
            path.name,
            solver.num_iteration,
            solver._resume_elapsed_seconds / 3600.0,
        )
        return {str(row["checkpoint_id"]): dict(row) for row in records}
    return {}


def _assert_candidate(config: Mapping, curves: Sequence[Mapping]) -> None:
    required = {
        "regret_network_type": "mlp",
        "critic_target_average_window": 4,
        "fixed_control_variate_beta": 1.0,
        "q_ensemble_size": 2,
        "use_instantaneous_predictor": False,
        "use_residual_calibration": True,
        "average_policy_loss": SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Experiment 29 requires {key}={expected!r}")
    measured = [row for row in curves if row["unbiased_estimator_sample_count"] > 0]
    if measured and any(
        not np.isclose(row["control_variate_beta_min"], 1.0)
        or not np.isclose(row["control_variate_beta_max"], 1.0)
        for row in measured
    ):
        raise RuntimeError("Observed control-variate beta differed from one")


def _validate_resume_round_trip(
    *, worker_dir: Path, seed: int, commit: str, config: Mapping
) -> None:
    path = next(
        (candidate for candidate in _training_state_paths(worker_dir, seed=seed, smoke=True) if candidate.is_file()),
        None,
    )
    if path is None:
        raise RuntimeError("Smoke did not produce a resumable training state")
    verifier = _make_solver(seed, config)
    payload = read_training_state(path)
    records = restore_training_state(
        verifier,
        payload,
        variant_id=CANDIDATE_ID,
        seed=seed,
        repository_commit=commit,
        config=config,
        expected_state_type=TRAINING_STATE_TYPE,
    )
    if not records or verifier.num_iteration != int(payload["solver"]["num_iteration"]):
        raise RuntimeError("Smoke continuation state did not restore its progress")
    verifier._solve_start_time = time.perf_counter() - verifier._resume_elapsed_seconds
    verifier._post_checkpoint_callback = None
    verifier.iteration()
    del verifier


def run_worker(
    *,
    seed: int,
    schedule: Sequence[Mapping],
    worker_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(CANDIDATE_CONFIG)
    if smoke:
        _smoke_overrides(config)
    commit = _repository_commit()
    solver = _make_solver(seed, config)
    solver.target_nodes_touched = 2**62
    captured = (
        _restore_latest(
            solver,
            worker_dir=worker_dir,
            seed=seed,
            smoke=smoke,
            commit=commit,
            config=config,
        )
        if resume
        else {}
    )
    excluded_snapshot_seconds = 0.0
    diagnostic_seconds_offset = float(solver._cumulative_factorial_diagnostic_seconds)
    state_ids = set(training_state_checkpoint_ids(smoke=smoke))

    def capture(active_solver, raw_checkpoint):
        nonlocal excluded_snapshot_seconds
        active_seconds = max(
            0.0,
            float(raw_checkpoint["wall_clock_seconds"])
            - excluded_snapshot_seconds
            - (
                float(active_solver._cumulative_factorial_diagnostic_seconds)
                - diagnostic_seconds_offset
            ),
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
            captured[checkpoint_id] = _save_policy(
                solver=active_solver,
                seed=seed,
                target=target,
                checkpoint=checkpoint,
                config=config,
                worker_dir=worker_dir,
                commit=commit,
            )
            if checkpoint_id in state_ids:
                state_path = worker_dir / "training_states" / (
                    f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pt"
                )
                payload = build_training_state(
                    active_solver,
                    variant_id=CANDIDATE_ID,
                    seed=seed,
                    checkpoint_id=checkpoint_id,
                    active_seconds=active_seconds,
                    repository_commit=commit,
                    config=config,
                    captured_snapshots=[
                        captured[str(row["checkpoint_id"])]
                        for row in schedule
                        if str(row["checkpoint_id"]) in captured
                    ],
                    state_type=TRAINING_STATE_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                save_training_state(state_path, payload)
                _sync_remote_resume_point(worker_dir)
                LOGGER.info("Saved resumable state %s", state_path.name)
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
        raw_curves = (
            list(solver.checkpoint_rows)
            if len(captured) == len(schedule)
            else solver.solve(post_checkpoint_callback=capture)
        )
        if len(captured) != len(schedule):
            missing = [
                row["checkpoint_id"]
                for row in schedule
                if row["checkpoint_id"] not in captured
            ]
            raise RuntimeError(f"Safety cap reached with missing checkpoints: {missing}")
        records = [captured[str(target["checkpoint_id"])] for target in schedule]
        validate_records(records, candidate_id=CANDIDATE_ID, seed=seed, schedule=schedule)
        for record in records:
            validate_playable_snapshot(
                UCV_POLICY_LOADER_ID, worker_dir / record["relative_path"]
            )
        curves = _curve_rows(raw_curves, seed=seed)
        _assert_candidate(config, curves)
        diagnostics_dir = worker_dir / "diagnostics"
        write_csv(worker_dir / "checkpoint_curves.csv", curves)
        write_csv(diagnostics_dir / "information_action_diagnostics.csv", solver.information_action_rows())
        write_csv(diagnostics_dir / "beta_histogram.csv", solver.beta_histogram_rows())
        write_csv(
            diagnostics_dir / "critic_error_subsequent_local_regret.csv",
            solver.critic_subsequent_regret_rows(),
        )
        write_csv(diagnostics_dir / "exact_q_oracle_diagnostics.csv", solver.q_oracle_diagnostic_rows)
        training_states = []
        for path in _training_state_paths(worker_dir, seed=seed, smoke=smoke):
            if path.is_file():
                training_states.append(
                    {
                        "filename": path.name,
                        "relative_path": str(path.relative_to(worker_dir)),
                        "sha256": sha256(path),
                        "size_bytes": int(path.stat().st_size),
                    }
                )
        if smoke:
            _validate_resume_round_trip(
                worker_dir=worker_dir, seed=seed, commit=commit, config=config
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
            "training_states": training_states,
            "resumed_from_training_state": bool(solver._resume_elapsed_seconds > 0.0),
            "resume_round_trip_validated": bool(smoke),
            "peak_rss_mb": _peak_rss_mb(),
            "artifacts": {
                "checkpoint_curves": "checkpoint_curves.csv",
                "information_action_diagnostics": "diagnostics/information_action_diagnostics.csv",
                "beta_histogram": "diagnostics/beta_histogram.csv",
                "critic_error_subsequent_local_regret": "diagnostics/critic_error_subsequent_local_regret.csv",
                "exact_q_oracle_diagnostics": "diagnostics/exact_q_oracle_diagnostics.csv",
            },
            "status": "complete",
        }
        write_json(worker_dir / "worker_result.json", result)
        write_json(
            worker_dir / "SUCCESS.json",
            {"status": "complete", "snapshots": records, "training_states": training_states},
        )
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["DiagnosticPromotedUCVSolver", "run_worker"]
