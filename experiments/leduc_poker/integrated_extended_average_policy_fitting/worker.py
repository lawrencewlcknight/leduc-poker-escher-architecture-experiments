"""Train one Experiment 43 trajectory and refine each saved policy checkpoint."""

from __future__ import annotations

from copy import deepcopy
import csv
import gc
import logging
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from open_spiel.python.algorithms import expected_game_score

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
    table_action_probabilities,
    tabular_policy_from_callable,
)
from experiments.leduc_poker.average_policy_optimization_horizon.fitting import (
    fit_continuously,
    grouped_replay_dataset,
    objective_value,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    _policy_errors,
    empirical_table,
)
from experiments.leduc_poker.policy_post_training.core import model_table
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import (
    augment_snapshot,
    sha256,
    validate_playable_snapshot,
    validate_records,
    write_csv,
    write_json,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.worker import (
    DiagnosticPromotedUCVSolver,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    build_training_state,
    read_training_state,
    restore_training_state,
    save_training_state,
)
from experiments.leduc_poker.ucv_residual_target_factorial.worker import (
    DIAGNOSTIC_FIELDS,
)
from unbiased_escher.policy_distillation import SOFT_TARGET_CROSS_ENTROPY
from vr_deep_cfr.logger import Logger

from .config import (
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    CANDIDATE_LABEL,
    EXPERIMENT_NAME,
    FINAL_AUDIT_UPDATES,
    ORDINARY_POLICY_ID,
    ORDINARY_POLICY_LABEL,
    REFINEMENT_GRADIENT_CLIP_NORM,
    REFINEMENT_LEARNING_RATE,
    REFINEMENT_UPDATES,
    SMOKE_FINAL_AUDIT_UPDATES,
    final_active_checkpoint_id,
    refinement_schedule,
    training_state_checkpoint_ids,
)


LOGGER = logging.getLogger(__name__)
UCV_POLICY_LOADER_ID = "unbiased_control_variate_escher"
TRAINING_STATE_TYPE = "experiment_43_full_training_state"
REMOTE_SYNC_ATTEMPTS = 5


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        capture_output=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _smoke_overrides(config: dict) -> None:
    for key in (
        "advantage_network_train_steps", "ave_policy_network_train_steps",
        "baseline_network_train_steps", "calibration_train_steps",
    ):
        config[key] = 1
    for key in (
        "advantage_batch_size", "ave_policy_batch_size", "baseline_batch_size",
        "calibration_batch_size",
    ):
        config[key] = 2
    for key in (
        "advantage_buffer_size", "ave_policy_buffer_size", "baseline_buffer_size",
        "calibration_buffer_size",
    ):
        config[key] = 128
    config.update(
        num_traversals=4,
        max_num_iterations=8,
        evaluation_frequency=1,
        evaluate_initial_policy=False,
        early_evaluation_node_thresholds=(),
        integrated_refinement_updates=2,
    )


def _make_solver(seed: int, config: Mapping[str, Any]) -> DiagnosticPromotedUCVSolver:
    control_fields = {
        "max_num_iterations", "preserve_evaluation_rng", "evaluate_initial_policy",
        "early_evaluation_node_thresholds", "integrated_refinement_updates",
        "integrated_refinement_learning_rate",
        "integrated_refinement_gradient_clip_norm",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=2 * int(config["num_traversals"]) * int(config["max_num_iterations"]),
        seed=int(seed), logger=Logger(verbose=False),
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


def _buffer_source(solver) -> SimpleNamespace:
    buffer = solver.ave_policy_trainer.buffer
    size = min(int(buffer.cur_id), int(buffer.buffer_size))
    if size <= 0:
        raise ValueError("Cannot refine an empty average-policy reservoir")
    return SimpleNamespace(
        infostates=np.asarray(buffer.infostate_buf[:size], dtype=np.float32),
        policies=np.asarray(buffer.q_value_buf[:size], dtype=np.float64),
        legal_masks=np.asarray(buffer.q_value_mask_buf[:size], dtype=np.float32),
        iterations=np.asarray(buffer.iteration_buf[:size], dtype=np.float64).reshape(-1),
        gamma=float(solver.gamma),
    )


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_solver_policy(
    *, solver, model, policy_id: str, seed: int, target: Mapping,
    checkpoint: Mapping, config: Mapping, worker_dir: Path, commit: str,
    suffix: str = "",
) -> dict:
    checkpoint_id = str(target["checkpoint_id"])
    suffix_text = f"_{suffix}" if suffix else ""
    path = worker_dir / "snapshots" / (
        f"{policy_id}_seed_{seed}_{checkpoint_id}{suffix_text}.pkl"
    )
    original = solver.ave_policy_trainer.model
    try:
        solver.ave_policy_trainer.model = model
        save_torch_policy_snapshot(
            solver, path, seed=int(seed), iteration=int(checkpoint["iteration"]),
            arm="checkpointed", config=dict(config),
            stage_label=f"Experiment 43 {policy_id} {checkpoint_id}{suffix_text}",
            checkpoint_target_nodes=(
                int(target["target_nodes"])
                if target.get("target_nodes") is not None
                else int(checkpoint["nodes_touched"])
            ),
        )
    finally:
        solver.ave_policy_trainer.model = original
    record = augment_snapshot(
        path, candidate_id=policy_id, seed=seed, checkpoint=target,
        nodes_touched=int(checkpoint["nodes_touched"]),
        active_seconds=float(checkpoint["active_seconds"]),
        completed_iteration=int(checkpoint["iteration"]),
        repository_commit=commit, config=config,
    )
    if suffix:
        record["refinement_update"] = int(suffix.replace("update_", ""))
    return record


def _seat_averaged_ev(game, candidate_table, ordinary_table) -> float:
    candidate = tabular_policy_from_callable(
        game, lambda state: table_action_probabilities(candidate_table, state)
    )
    ordinary = tabular_policy_from_callable(
        game, lambda state: table_action_probabilities(ordinary_table, state)
    )
    as_player_0 = float(
        expected_game_score.policy_value(game.new_initial_state(), [candidate, ordinary])[0]
    )
    as_player_1 = float(
        expected_game_score.policy_value(game.new_initial_state(), [ordinary, candidate])[1]
    )
    return 0.5 * (as_player_0 + as_player_1)


def _training_state_paths(worker_dir: Path, *, seed: int, smoke: bool) -> list[Path]:
    return [
        worker_dir / "training_states" / f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pt"
        for checkpoint_id in reversed(training_state_checkpoint_ids(smoke=smoke))
    ]


def _sync_remote_resume_point(worker_dir: Path) -> None:
    remote = os.environ.get("EXP43_REMOTE_TASK_URI")
    if not remote:
        return
    if not remote.startswith("gs://") or remote != remote.strip() or any(
        ord(character) < 32 for character in remote
    ):
        raise ValueError(f"Invalid Experiment 43 remote task URI: {remote!r}")
    command = [
        "gcloud", "storage", "rsync", "--recursive",
        str(worker_dir.resolve()), remote.rstrip("/"),
    ]
    last = None
    for attempt in range(REMOTE_SYNC_ATTEMPTS):
        last = subprocess.run(command, check=False)
        if last.returncode == 0:
            return
        if attempt + 1 < REMOTE_SYNC_ATTEMPTS:
            time.sleep(min(2.0 ** attempt, 30.0))
    assert last is not None
    raise subprocess.CalledProcessError(last.returncode, command)


def _read_existing_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _restore_latest(
    solver, *, worker_dir: Path, seed: int, smoke: bool, commit: str,
    config: Mapping,
) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    for path in _training_state_paths(worker_dir, seed=seed, smoke=smoke):
        if not path.is_file():
            continue
        payload = read_training_state(path)
        records = restore_training_state(
            solver, payload, variant_id=CANDIDATE_ID, seed=seed,
            repository_commit=commit, config=config,
            expected_state_type=TRAINING_STATE_TYPE,
        )
        captured = {}
        for record in records:
            snapshot = worker_dir / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Restored snapshot is missing or corrupt: {snapshot}")
            captured[(str(record["candidate_id"]), str(record["checkpoint_id"]))] = dict(record)
        LOGGER.info(
            "Resuming Experiment 43 seed %s from %s at %.2f active hours",
            seed, path.name, solver._resume_elapsed_seconds / 3600.0,
        )
        return captured, _read_existing_metrics(worker_dir / "refinement_metrics.csv")
    return {}, []


def _assert_candidate(config: Mapping, curves: Sequence[Mapping], *, smoke: bool) -> None:
    required = {
        "regret_network_type": "mlp",
        "critic_target_average_window": 4,
        "fixed_control_variate_beta": 1.0,
        "q_ensemble_size": 2,
        "use_instantaneous_predictor": False,
        "average_policy_loss": SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
        "ave_policy_network_train_steps": 1 if smoke else 5_000,
        "integrated_refinement_updates": 2 if smoke else REFINEMENT_UPDATES,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Experiment 43 requires {key}={expected!r}")
    measured = [row for row in curves if row["unbiased_estimator_sample_count"] > 0]
    if measured and any(
        not np.isclose(row["control_variate_beta_min"], 1.0)
        or not np.isclose(row["control_variate_beta_max"], 1.0)
        for row in measured
    ):
        raise RuntimeError("Observed control-variate beta differed from one")


def _validate_resume_round_trip(
    *, worker_dir: Path, seed: int, commit: str, config: Mapping,
) -> None:
    path = next(
        (item for item in _training_state_paths(worker_dir, seed=seed, smoke=True) if item.is_file()),
        None,
    )
    if path is None:
        raise RuntimeError("Smoke did not produce a resumable training state")
    verifier = _make_solver(seed, config)
    payload = read_training_state(path)
    records = restore_training_state(
        verifier, payload, variant_id=CANDIDATE_ID, seed=seed,
        repository_commit=commit, config=config,
        expected_state_type=TRAINING_STATE_TYPE,
    )
    if not records or verifier.num_iteration != int(payload["solver"]["num_iteration"]):
        raise RuntimeError("Smoke continuation state did not restore its progress")
    verifier._solve_start_time = time.perf_counter() - verifier._resume_elapsed_seconds
    verifier._post_checkpoint_callback = None
    verifier.iteration()


def run_worker(
    *, seed: int, schedule: Sequence[Mapping], worker_dir: Path,
    smoke: bool, resume: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = deepcopy(CANDIDATE_CONFIG)
    if smoke:
        _smoke_overrides(config)
    commit = _repository_commit()
    solver = _make_solver(seed, config)
    solver.target_nodes_touched = 2**62
    captured, refinement_rows = (
        _restore_latest(
            solver, worker_dir=worker_dir, seed=seed, smoke=smoke,
            commit=commit, config=config,
        ) if resume else ({}, [])
    )
    excluded_seconds = 0.0
    diagnostic_offset = float(solver._cumulative_factorial_diagnostic_seconds)
    state_ids = set(training_state_checkpoint_ids(smoke=smoke))
    final_checkpoint = final_active_checkpoint_id(smoke=smoke)
    intermediate_records: list[dict] = []

    def capture(active_solver, raw_checkpoint):
        nonlocal excluded_seconds, refinement_rows
        active_seconds = max(
            0.0,
            float(raw_checkpoint["wall_clock_seconds"])
            - excluded_seconds
            - (float(active_solver._cumulative_factorial_diagnostic_seconds) - diagnostic_offset),
        )
        checkpoint = dict(raw_checkpoint)
        checkpoint["active_seconds"] = active_seconds
        checkpoint["checkpoint_row_index"] = len(active_solver.checkpoint_rows) - 1
        for target in schedule:
            checkpoint_id = str(target["checkpoint_id"])
            ordinary_key = (ORDINARY_POLICY_ID, checkpoint_id)
            refined_key = (CANDIDATE_ID, checkpoint_id)
            if refined_key in captured or not _target_reached(
                target, active_seconds=active_seconds,
                nodes=int(checkpoint["nodes_touched"]),
            ):
                continue
            started = time.perf_counter()
            saved_rng = _rng_state()
            ordinary_model = active_solver.ave_policy_trainer.model
            ordinary_table = model_table(active_solver.game, ordinary_model)
            ordinary_exploitability = exact_exploitability(active_solver.game, ordinary_table)
            exact_table = active_solver.exact_average_strategy.table()
            exact_average = exact_exploitability(active_solver.game, exact_table)
            source = _buffer_source(active_solver)
            dataset = grouped_replay_dataset(source)
            empirical_policy_table, empirical_exploitability = empirical_table(
                source, active_solver.game
            )
            captured[ordinary_key] = _save_solver_policy(
                solver=active_solver, model=ordinary_model,
                policy_id=ORDINARY_POLICY_ID, seed=seed, target=target,
                checkpoint=checkpoint, config=config, worker_dir=worker_dir,
                commit=commit,
            )
            refined_model = deepcopy(ordinary_model)
            updates = refinement_schedule(
                smoke=smoke, final_checkpoint=checkpoint_id == final_checkpoint
            )
            update_rows = []
            fit_started = time.perf_counter()

            def record(update: int, loss_before: float, fitted_model) -> None:
                fitted_table = model_table(active_solver.game, fitted_model)
                exploitability = exact_exploitability(active_solver.game, fitted_table)
                errors = _policy_errors(
                    exact_table, fitted_table,
                    active_solver.exact_average_strategy.denominators,
                )
                update_rows.append(
                    {
                        "seed": int(seed),
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_type": str(target["checkpoint_type"]),
                        "checkpoint_target_active_hours": target.get("target_active_hours"),
                        "actual_active_hours": active_seconds / 3600.0,
                        "nodes_touched": int(checkpoint["nodes_touched"]),
                        "completed_iteration": int(checkpoint["iteration"]),
                        "refinement_update": int(update),
                        "policy_id": ORDINARY_POLICY_ID if update == 0 else CANDIDATE_ID,
                        "exploitability": float(exploitability),
                        "ordinary_exploitability": float(ordinary_exploitability),
                        "refined_minus_ordinary_exploitability": float(exploitability - ordinary_exploitability),
                        "empirical_reservoir_exploitability": float(empirical_exploitability),
                        "exact_tabular_average_exploitability": float(exact_average),
                        "neural_minus_empirical_gap": float(exploitability - empirical_exploitability),
                        "neural_minus_exact_gap": float(exploitability - exact_average),
                        "full_batch_cross_entropy": objective_value(fitted_model, dataset),
                        "training_loss_before_update": float(loss_before),
                        "fit_elapsed_seconds": time.perf_counter() - fit_started,
                        "replay_rows": int(dataset.replay_rows),
                        "unique_information_states": int(len(dataset.features)),
                        **{key: value for key, value in errors.items() if key != "rows"},
                    }
                )
                maximum = int(config["integrated_refinement_updates"])
                if update == maximum or (
                    checkpoint_id == final_checkpoint
                    and update in (SMOKE_FINAL_AUDIT_UPDATES if smoke else FINAL_AUDIT_UPDATES)
                ):
                    suffix = "" if update == maximum else f"update_{update}"
                    record_row = _save_solver_policy(
                        solver=active_solver, model=fitted_model,
                        policy_id=CANDIDATE_ID, seed=seed, target=target,
                        checkpoint=checkpoint, config=config, worker_dir=worker_dir,
                        commit=commit, suffix=suffix,
                    )
                    record_row["refinement_update"] = int(update)
                    if update == maximum:
                        captured[refined_key] = record_row
                    else:
                        intermediate_records.append(record_row)

            try:
                fit_continuously(
                    model=refined_model, dataset=dataset,
                    learning_rate=REFINEMENT_LEARNING_RATE,
                    gradient_clip_norm=REFINEMENT_GRADIENT_CLIP_NORM,
                    evaluation_updates=updates, callback=record,
                )
                final_table = model_table(active_solver.game, refined_model)
                final_row = update_rows[-1]
                final_row["refined_vs_ordinary_seat_averaged_ev"] = _seat_averaged_ev(
                    active_solver.game, final_table, ordinary_table
                )
                for row in update_rows[:-1]:
                    row["refined_vs_ordinary_seat_averaged_ev"] = math.nan
                refinement_rows.extend(update_rows)
                write_csv(worker_dir / "refinement_metrics.csv", refinement_rows)
            finally:
                active_solver.ave_policy_trainer.model = ordinary_model
                _restore_rng(saved_rng)

            if checkpoint_id in state_ids:
                state_path = worker_dir / "training_states" / (
                    f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pt"
                )
                ordered_records = [
                    captured[key]
                    for row in schedule
                    for key in (
                        (ORDINARY_POLICY_ID, str(row["checkpoint_id"])),
                        (CANDIDATE_ID, str(row["checkpoint_id"])),
                    )
                    if key in captured
                ]
                payload = build_training_state(
                    active_solver, variant_id=CANDIDATE_ID, seed=seed,
                    checkpoint_id=checkpoint_id, active_seconds=active_seconds,
                    repository_commit=commit, config=config,
                    captured_snapshots=ordered_records,
                    state_type=TRAINING_STATE_TYPE, experiment_name=EXPERIMENT_NAME,
                )
                save_training_state(state_path, payload)
                _sync_remote_resume_point(worker_dir)
            excluded_seconds += time.perf_counter() - started
            LOGGER.info(
                "Saved and refined %s seed %s at iteration %s, %s nodes, %.2f active hours",
                checkpoint_id, seed, checkpoint["iteration"],
                checkpoint["nodes_touched"], active_seconds / 3600.0,
            )
        if sum(key[0] == CANDIDATE_ID for key in captured) == len(schedule):
            active_solver.target_nodes_touched = int(active_solver.nodes_touched)

    try:
        raw_curves = (
            list(solver.checkpoint_rows)
            if sum(key[0] == CANDIDATE_ID for key in captured) == len(schedule)
            else solver.solve(post_checkpoint_callback=capture)
        )
        ordinary_records = [
            captured[(ORDINARY_POLICY_ID, str(target["checkpoint_id"]))]
            for target in schedule
        ]
        refined_records = [
            captured[(CANDIDATE_ID, str(target["checkpoint_id"]))]
            for target in schedule
        ]
        validate_records(
            ordinary_records, candidate_id=ORDINARY_POLICY_ID,
            seed=seed, schedule=schedule,
        )
        validate_records(
            refined_records, candidate_id=CANDIDATE_ID,
            seed=seed, schedule=schedule,
        )
        for record in ordinary_records + refined_records + intermediate_records:
            validate_playable_snapshot(
                UCV_POLICY_LOADER_ID, worker_dir / record["relative_path"]
            )
        curves = _curve_rows(raw_curves, seed=seed)
        _assert_candidate(config, curves, smoke=smoke)
        write_csv(worker_dir / "checkpoint_curves.csv", curves)
        diagnostics = worker_dir / "diagnostics"
        write_csv(diagnostics / "information_action_diagnostics.csv", solver.information_action_rows())
        write_csv(diagnostics / "beta_histogram.csv", solver.beta_histogram_rows())
        write_csv(diagnostics / "critic_error_subsequent_local_regret.csv", solver.critic_subsequent_regret_rows())
        write_csv(diagnostics / "exact_q_oracle_diagnostics.csv", solver.q_oracle_diagnostic_rows)
        states = []
        for path in _training_state_paths(worker_dir, seed=seed, smoke=smoke):
            if path.is_file():
                states.append(
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
            "ordinary_policy_id": ORDINARY_POLICY_ID,
            "ordinary_policy_label": ORDINARY_POLICY_LABEL,
            "seed": int(seed),
            "smoke": bool(smoke),
            "checkpoint_schedule": list(schedule),
            "repository_commit": commit,
            "config": config,
            "ordinary_snapshots": ordinary_records,
            "refined_snapshots": refined_records,
            "final_audit_snapshots": intermediate_records,
            "training_states": states,
            "resumed_from_training_state": bool(solver._resume_elapsed_seconds > 0.0),
            "resume_round_trip_validated": bool(smoke),
            "peak_rss_mb": _peak_rss_mb(),
            "artifacts": {
                "checkpoint_curves": "checkpoint_curves.csv",
                "refinement_metrics": "refinement_metrics.csv",
            },
            "status": "complete",
        }
        write_json(worker_dir / "worker_result.json", result)
        write_json(
            worker_dir / "SUCCESS.json",
            {"status": "complete", "seed": int(seed), "snapshots": ordinary_records + refined_records},
        )
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["run_worker"]
