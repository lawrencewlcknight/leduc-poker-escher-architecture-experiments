"""Run one Experiment 25 factorial arm and paired seed."""

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
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    ExactLeducOracle,
    ExactWeightedAverageStrategy,
    aggregate_q_rows,
    build_policy_table,
    exact_exploitability,
)
from experiments.leduc_poker.ucv_three_arm_15m_simplification.diagnostics import (
    DiagnosticUCVSolver,
)
from unbiased_escher.factorial import FactorialUnbiasedControlVariateEscher
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
    EXPERIMENT_NAME,
    VARIANTS,
    training_state_checkpoint_ids,
    variant_config,
)
from .training_state import (
    build_training_state,
    read_training_state,
    restore_training_state,
    save_training_state,
)


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
    "regret_policy_learning_rate",
    "exact_average_exploitability",
    "neural_average_exploitability",
    "average_policy_distillation_gap",
    "current_strategy_exploitability",
    "q_oracle_mae",
    "q_oracle_rmse",
    "q_target_update_l2_mean",
    "q_target_history_size_min",
    "q_target_history_size_max",
    "q_ensemble_target_version_min",
    "q_ensemble_target_version_max",
    "factorial_diagnostic_wall_clock_seconds",
)


class DiagnosticFactorialUCVSolver(
    DiagnosticUCVSolver, FactorialUnbiasedControlVariateEscher
):
    """Factorial solver composed with the established passive diagnostics."""

    def __init__(self, *args, **kwargs):
        self.q_oracle_diagnostic_rows: list[dict] = []
        self._cumulative_factorial_diagnostic_seconds = 0.0
        super().__init__(*args, **kwargs)
        self.exact_average_strategy = ExactWeightedAverageStrategy(
            self.game, gamma=self.gamma
        )

    def current_strategy(self, state) -> np.ndarray:
        player = int(state.current_player())
        return self.regret_trainers[player].get_policy(
            state, max(1, int(self.num_iteration))
        )

    def iteration(self):
        diagnostic_start = time.perf_counter()
        next_iteration = int(self.num_iteration) + 1
        self.exact_average_strategy.observe_iteration(
            next_iteration,
            lambda state: self.regret_trainers[int(state.current_player())].get_policy(
                state, next_iteration
            ),
        )
        self._cumulative_factorial_diagnostic_seconds += (
            time.perf_counter() - diagnostic_start
        )
        return super().iteration()

    def _q_oracle_rows(self) -> list[dict]:
        current_table = build_policy_table(self.game, self.current_strategy)
        ensemble = self.q_value_trainer
        original_fold = int(ensemble.active_fold)
        rows = []
        try:
            for heldout_fold in range(int(ensemble.ensemble_size)):
                ensemble.active_fold = heldout_fold
                oracle = ExactLeducOracle(self, current_table)
                rows.extend(
                    {
                        "checkpoint_index": len(self.checkpoint_rows),
                        "iteration": int(self.num_iteration),
                        "nodes_touched": int(self.nodes_touched),
                        "heldout_fold": heldout_fold,
                        **row,
                    }
                    for row in oracle.q_oracle_rows()
                )
        finally:
            ensemble.active_fold = original_fold
        return rows

    def evaluate(self, **kwargs):
        diagnostic_start = time.perf_counter()
        exact_average = self.exact_average_strategy.exploitability()
        neural_table = build_policy_table(
            self.game,
            lambda state: self.ave_policy_trainer.action_probabilities(
                state, probs_as_dict=False
            ),
        )
        neural_average = exact_exploitability(self.game, neural_table)
        current_table = build_policy_table(self.game, self.current_strategy)
        current_exploitability = exact_exploitability(self.game, current_table)
        q_rows = self._q_oracle_rows()
        self.q_oracle_diagnostic_rows.extend(q_rows)
        q_summary = aggregate_q_rows(q_rows)
        updates = [
            float(member.last_target_update_l2)
            for member in self.q_value_trainer.members
        ]
        histories = [
            len(member.target_history) for member in self.q_value_trainer.members
        ]
        self.logger.record("exact_average_exploitability", exact_average)
        self.logger.record("neural_average_exploitability", neural_average)
        self.logger.record(
            "average_policy_distillation_gap", neural_average - exact_average
        )
        self.logger.record("current_strategy_exploitability", current_exploitability)
        self.logger.record("q_oracle_mae", q_summary["q_oracle_mae"])
        self.logger.record("q_oracle_rmse", q_summary["q_oracle_rmse"])
        self.logger.record("q_target_update_l2_mean", float(np.mean(updates)))
        self.logger.record("q_target_history_size_min", int(min(histories)))
        self.logger.record("q_target_history_size_max", int(max(histories)))
        self._cumulative_factorial_diagnostic_seconds += (
            time.perf_counter() - diagnostic_start
        )
        self.logger.record(
            "factorial_diagnostic_wall_clock_seconds",
            self._cumulative_factorial_diagnostic_seconds,
        )
        return super().evaluate(**kwargs)


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


def _make_solver(seed: int, config: Mapping[str, Any]) -> DiagnosticFactorialUCVSolver:
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
    solver = DiagnosticFactorialUCVSolver(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def _parse_float(value) -> float:
    return math.nan if value in {None, ""} else float(value)


def _curve_rows(raw_rows, *, variant_id: str, seed: int) -> list[dict]:
    rows = []
    for index, raw in enumerate(raw_rows):
        row = {
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["variant_label"],
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
    variant_id: str,
    seed: int,
    target: Mapping,
    checkpoint: Mapping,
    config: Mapping,
    worker_dir: Path,
    commit: str,
) -> dict:
    checkpoint_id = str(target["checkpoint_id"])
    path = worker_dir / "snapshots" / f"{variant_id}_seed_{seed}_{checkpoint_id}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_policy_snapshot(
        solver,
        path,
        seed=int(seed),
        iteration=int(checkpoint["iteration"]),
        arm="checkpointed",
        config=dict(config),
        stage_label=f"Experiment 25 {variant_id} {checkpoint_id}",
        checkpoint_target_nodes=(
            int(target["target_nodes"])
            if target.get("target_nodes") is not None
            else int(checkpoint["nodes_touched"])
        ),
    )
    return augment_snapshot(
        path,
        variant_id=variant_id,
        seed=seed,
        checkpoint=target,
        nodes_touched=int(checkpoint["nodes_touched"]),
        active_seconds=float(checkpoint["active_seconds"]),
        completed_iteration=int(checkpoint["iteration"]),
        repository_commit=commit,
        config=config,
    )


def _candidate_training_state_paths(
    worker_dir: Path, *, variant_id: str, seed: int, smoke: bool
) -> list[Path]:
    return [
        worker_dir / "training_states" / f"{variant_id}_seed_{seed}_{checkpoint_id}.pt"
        for checkpoint_id in reversed(training_state_checkpoint_ids(smoke=smoke))
    ]


def _sync_remote_resume_point(worker_dir: Path) -> None:
    """Durably upload a newly written continuation state when running on GCP."""
    remote = os.environ.get("EXP25_REMOTE_TASK_URI")
    if not remote:
        return
    subprocess.run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            str(Path(worker_dir).resolve()),
            remote.rstrip("/"),
        ],
        check=True,
    )


def _restore_latest(
    solver,
    *,
    worker_dir: Path,
    variant_id: str,
    seed: int,
    smoke: bool,
    commit: str,
    config: Mapping,
) -> dict[str, dict]:
    for path in _candidate_training_state_paths(
        worker_dir, variant_id=variant_id, seed=seed, smoke=smoke
    ):
        if not path.is_file():
            continue
        payload = read_training_state(path)
        records = restore_training_state(
            solver,
            payload,
            variant_id=variant_id,
            seed=seed,
            repository_commit=commit,
            config=config,
        )
        for record in records:
            snapshot = worker_dir / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Restored snapshot is missing or corrupt: {snapshot}")
        LOGGER.info(
            "Resuming %s seed %s from %s at iteration %s and %.2f active hours",
            variant_id,
            seed,
            path.name,
            solver.num_iteration,
            solver._resume_elapsed_seconds / 3600.0,
        )
        return {str(row["checkpoint_id"]): dict(row) for row in records}
    return {}


def _assert_variant(variant_id: str, config: Mapping, curves: Sequence[Mapping]) -> None:
    treatment = VARIANTS[variant_id]
    expected_network = (
        "residual_layer_norm" if treatment["residual_regret"] else "mlp"
    )
    expected_window = 4 if treatment["averaged_critic_target"] else 1
    if config.get("regret_network_type") != expected_network:
        raise RuntimeError(f"{variant_id} used the wrong regret architecture")
    if int(config.get("critic_target_average_window", -1)) != expected_window:
        raise RuntimeError(f"{variant_id} used the wrong critic target window")
    if config.get("fixed_control_variate_beta") != 1.0:
        raise RuntimeError("Experiment 25 must retain fixed beta=1")
    if int(config.get("q_ensemble_size", -1)) != 2:
        raise RuntimeError("Experiment 25 must retain two cross-fitted critics")
    if config.get("use_instantaneous_predictor") is not False:
        raise RuntimeError("Experiment 25 must retain the non-predictive core")
    measured = [row for row in curves if row["unbiased_estimator_sample_count"] > 0]
    if measured and any(
        not np.isclose(row["control_variate_beta_min"], 1.0)
        or not np.isclose(row["control_variate_beta_max"], 1.0)
        for row in measured
    ):
        raise RuntimeError("Observed control-variate beta differed from one")


def _validate_resume_round_trip(
    *,
    worker_dir: Path,
    variant_id: str,
    seed: int,
    commit: str,
    config: Mapping,
) -> None:
    """Smoke-only restore plus one complete optimisation iteration."""
    candidates = _candidate_training_state_paths(
        worker_dir, variant_id=variant_id, seed=seed, smoke=True
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError("Smoke did not produce a resumable training state")
    verifier = _make_solver(seed, config)
    payload = read_training_state(path)
    records = restore_training_state(
        verifier,
        payload,
        variant_id=variant_id,
        seed=seed,
        repository_commit=commit,
        config=config,
    )
    if not records or verifier.num_iteration != int(payload["solver"]["num_iteration"]):
        raise RuntimeError("Smoke continuation state did not restore its progress")
    verifier._solve_start_time = time.perf_counter() - verifier._resume_elapsed_seconds
    verifier._post_checkpoint_callback = None
    verifier.iteration()
    del verifier


def run_worker(
    *,
    variant_id: str,
    seed: int,
    schedule: Sequence[Mapping],
    worker_dir: Path,
    smoke: bool,
    resume: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(variant_config(variant_id))
    if smoke:
        _smoke_overrides(config)
    commit = _repository_commit()
    solver = _make_solver(seed, config)
    solver.target_nodes_touched = 2**62
    captured = (
        _restore_latest(
            solver,
            worker_dir=worker_dir,
            variant_id=variant_id,
            seed=seed,
            smoke=smoke,
            commit=commit,
            config=config,
        )
        if resume
        else {}
    )
    excluded_snapshot_seconds = 0.0
    diagnostic_seconds_offset = float(
        solver._cumulative_factorial_diagnostic_seconds
    )
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
                variant_id=variant_id,
                seed=seed,
                target=target,
                checkpoint=checkpoint,
                config=config,
                worker_dir=worker_dir,
                commit=commit,
            )
            if checkpoint_id in state_ids:
                state_path = (
                    worker_dir
                    / "training_states"
                    / f"{variant_id}_seed_{seed}_{checkpoint_id}.pt"
                )
                payload = build_training_state(
                    active_solver,
                    variant_id=variant_id,
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
                )
                save_training_state(state_path, payload)
                _sync_remote_resume_point(worker_dir)
                LOGGER.info("Saved resumable state %s", state_path.name)
            excluded_snapshot_seconds += time.perf_counter() - save_start
            LOGGER.info(
                "Saved %s %s seed %s at iteration %s, %s nodes, %.1f active seconds",
                variant_id,
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
        validate_records(records, variant_id=variant_id, seed=seed, schedule=schedule)
        for record in records:
            validate_playable_snapshot(
                UCV_POLICY_LOADER_ID, worker_dir / record["relative_path"]
            )
        curves = _curve_rows(raw_curves, variant_id=variant_id, seed=seed)
        _assert_variant(variant_id, config, curves)
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
        write_csv(
            diagnostics_dir / "exact_q_oracle_diagnostics.csv",
            solver.q_oracle_diagnostic_rows,
        )
        training_states = []
        for path in _candidate_training_state_paths(
            worker_dir, variant_id=variant_id, seed=seed, smoke=smoke
        ):
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
                worker_dir=worker_dir,
                variant_id=variant_id,
                seed=seed,
                commit=commit,
                config=config,
            )
        result = {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["variant_label"],
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
            {
                "status": "complete",
                "snapshots": records,
                "training_states": training_states,
            },
        )
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["DiagnosticFactorialUCVSolver", "run_worker"]
