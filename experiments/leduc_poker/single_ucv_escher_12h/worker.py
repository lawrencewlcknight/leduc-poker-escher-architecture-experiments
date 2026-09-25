"""Train and evaluate one Experiment 44 Single UCV-ESCHER trajectory."""

from __future__ import annotations

from copy import deepcopy
import csv
import gc
import logging
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

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_optimization_horizon.fitting import (
    fit_continuously,
    grouped_replay_dataset,
)
from experiments.leduc_poker.average_policy_redistillation.distill import empirical_table
from experiments.leduc_poker.policy_post_training.core import model_table
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import (
    augment_snapshot,
    sha256,
    validate_playable_snapshot,
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
from vr_deep_cfr.logger import Logger

from .config import (
    BOUNDED_CAPACITIES,
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    CANDIDATE_LABEL,
    EMPIRICAL_ID,
    EMPIRICAL_LABEL,
    EXACT_ID,
    EXACT_LABEL,
    EXPERIMENT_NAME,
    FULL_MIXTURE_TOLERANCE,
    INCUMBENT_NEURAL_ID,
    INCUMBENT_NEURAL_LABEL,
    ORDINARY_NEURAL_ID,
    ORDINARY_NEURAL_LABEL,
    REFINEMENT_GRADIENT_CLIP_NORM,
    REFINEMENT_LEARNING_RATE,
    REFINEMENT_UPDATES,
    SMOKE_REFINEMENT_UPDATES,
    training_state_checkpoint_ids,
)
from .historical_policy import (
    HistoricalNetworkArchive,
    bounded_mixture_table,
    load_snapshot,
    table_max_abs_difference,
    weighted_average_table,
)


LOGGER = logging.getLogger(__name__)
UCV_POLICY_LOADER_ID = "unbiased_control_variate_escher"
TRAINING_STATE_TYPE = "experiment_44_single_ucv_escher_training_state"
REMOTE_SYNC_ATTEMPTS = 5


class HistoricalPromotedUCVSolver(DiagnosticPromotedUCVSolver):
    """The promoted learner with a pre-update regret-network archive."""

    historical_archive: HistoricalNetworkArchive | None = None

    def iteration(self):
        iteration = int(self.num_iteration) + 1
        if self.historical_archive is None:
            raise RuntimeError("Historical archive must be attached before training")
        started = time.perf_counter()
        self.historical_archive.capture(self, iteration)
        # Historical persistence is evaluation/deployment bookkeeping, not learning.
        self._cumulative_factorial_diagnostic_seconds += time.perf_counter() - started
        return super().iteration()


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
    )


def _make_solver(seed: int, config: Mapping[str, Any]) -> HistoricalPromotedUCVSolver:
    control_fields = {
        "max_num_iterations", "preserve_evaluation_rng", "evaluate_initial_policy",
        "early_evaluation_node_thresholds", "integrated_refinement_updates",
        "integrated_refinement_learning_rate", "integrated_refinement_gradient_clip_norm",
        "historical_policy_representation", "historical_policy_weighting",
        "historical_bounded_capacities",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=2 * int(config["num_traversals"]) * int(config["max_num_iterations"]),
        seed=int(seed), logger=Logger(verbose=False),
    )
    solver = HistoricalPromotedUCVSolver(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def _buffer_source(solver) -> SimpleNamespace:
    buffer = solver.ave_policy_trainer.buffer
    size = min(int(buffer.cur_id), int(buffer.buffer_size))
    if size <= 0:
        raise ValueError("Cannot evaluate an empty average-policy reservoir")
    return SimpleNamespace(
        infostates=np.asarray(buffer.infostate_buf[:size], dtype=np.float32),
        policies=np.asarray(buffer.q_value_buf[:size], dtype=np.float64),
        legal_masks=np.asarray(buffer.q_value_mask_buf[:size], dtype=np.float32),
        iterations=np.asarray(buffer.iteration_buf[:size], dtype=np.float64).reshape(-1),
        gamma=float(solver.gamma),
    )


def _rng_state() -> dict:
    result = {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def _restore_rng(state: Mapping) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _target_reached(target: Mapping, *, active_seconds: float, nodes: int) -> bool:
    if target["checkpoint_type"] == "active_time":
        return active_seconds >= float(target["target_active_seconds"])
    return nodes >= int(target["target_nodes"])


def _save_incumbent(
    *, solver, model, seed: int, target: Mapping, checkpoint: Mapping,
    config: Mapping, worker_dir: Path, commit: str,
) -> dict:
    path = worker_dir / "snapshots" / (
        f"{INCUMBENT_NEURAL_ID}_seed_{seed}_{target['checkpoint_id']}.pkl"
    )
    original = solver.ave_policy_trainer.model
    try:
        solver.ave_policy_trainer.model = model
        save_torch_policy_snapshot(
            solver, path, seed=int(seed), iteration=int(checkpoint["iteration"]),
            arm="checkpointed", config=dict(config),
            stage_label=f"Experiment 44 incumbent {target['checkpoint_id']}",
            checkpoint_target_nodes=int(checkpoint["nodes_touched"]),
        )
    finally:
        solver.ave_policy_trainer.model = original
    return augment_snapshot(
        path, candidate_id=INCUMBENT_NEURAL_ID, seed=seed, checkpoint=target,
        nodes_touched=int(checkpoint["nodes_touched"]),
        active_seconds=float(checkpoint["active_seconds"]),
        completed_iteration=int(checkpoint["iteration"]),
        repository_commit=commit, config=config,
    )


def _training_state_paths(worker_dir: Path, *, seed: int, smoke: bool) -> list[Path]:
    return [
        worker_dir / "training_states" / f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pt"
        for checkpoint_id in reversed(training_state_checkpoint_ids(smoke=smoke))
    ]


def _sync_remote_resume_point(worker_dir: Path) -> None:
    remote = os.environ.get("EXP44_REMOTE_TASK_URI")
    if not remote:
        return
    if not remote.startswith("gs://") or remote != remote.strip() or any(
        ord(character) < 32 for character in remote
    ):
        raise ValueError(f"Invalid Experiment 44 remote task URI: {remote!r}")
    command = ["gcloud", "storage", "rsync", "--recursive", str(worker_dir), remote.rstrip("/")]
    last = None
    for attempt in range(REMOTE_SYNC_ATTEMPTS):
        last = subprocess.run(command, check=False)
        if last.returncode == 0:
            return
        if attempt + 1 < REMOTE_SYNC_ATTEMPTS:
            time.sleep(min(2.0**attempt, 30.0))
    assert last is not None
    raise subprocess.CalledProcessError(last.returncode, command)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _restore_latest(
    solver, *, worker_dir: Path, seed: int, smoke: bool, commit: str,
    config: Mapping,
) -> tuple[dict[str, dict], list[dict], list[dict]]:
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
            captured[str(record["checkpoint_id"])] = dict(record)
        LOGGER.info("Resuming seed %s from %s", seed, path.name)
        return (
            captured,
            _read_csv(worker_dir / "policy_metrics.csv"),
            list((read_json(worker_dir / "reservoir_selections.json") if
                  (worker_dir / "reservoir_selections.json").is_file() else {}).get("rows", [])),
        )
    return {}, [], []


def read_json(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_historical_snapshot(worker_dir: Path, archive: HistoricalNetworkArchive) -> None:
    if not archive.entries:
        raise RuntimeError("Smoke produced no historical regret-network snapshots")
    for entry in (archive.entries[0], archive.entries[-1]):
        payload = load_snapshot(entry.path)
        if int(payload["iteration"]) != entry.iteration:
            raise RuntimeError("Historical snapshot round trip changed iteration metadata")
        if not entry.path.is_relative_to(worker_dir):
            raise RuntimeError("Historical archive escaped the worker directory")


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
    archive = HistoricalNetworkArchive(worker_dir / "historical_networks")
    solver.historical_archive = archive
    solver.target_nodes_touched = 2**62
    captured, metric_rows, selection_rows = (
        _restore_latest(
            solver, worker_dir=worker_dir, seed=seed, smoke=smoke,
            commit=commit, config=config,
        ) if resume else ({}, [], [])
    )
    if archive.entries and len(archive.entries) != int(solver.num_iteration):
        raise ValueError(
            "Historical archive length does not match restored solver iteration"
        )
    excluded_seconds = 0.0
    diagnostic_offset = float(solver._cumulative_factorial_diagnostic_seconds)
    state_ids = set(training_state_checkpoint_ids(smoke=smoke))

    def capture(active_solver, raw_checkpoint):
        nonlocal excluded_seconds, metric_rows, selection_rows
        active_seconds = max(
            0.0,
            float(raw_checkpoint["wall_clock_seconds"]) - excluded_seconds
            - (float(active_solver._cumulative_factorial_diagnostic_seconds) - diagnostic_offset),
        )
        checkpoint = dict(raw_checkpoint)
        checkpoint["active_seconds"] = active_seconds
        for target in schedule:
            checkpoint_id = str(target["checkpoint_id"])
            if checkpoint_id in captured or not _target_reached(
                target, active_seconds=active_seconds,
                nodes=int(checkpoint["nodes_touched"]),
            ):
                continue
            started = time.perf_counter()
            saved_rng = _rng_state()
            try:
                entries = archive.through(int(checkpoint["iteration"]))
                exact_table = active_solver.exact_average_strategy.table()
                exact_exp = exact_exploitability(active_solver.game, exact_table)
                full_table = weighted_average_table(
                    active_solver.game,
                    [(entry.policy_table, entry.iteration_weight) for entry in entries],
                )
                full_difference = table_max_abs_difference(full_table, exact_table)
                full_exp = exact_exploitability(active_solver.game, full_table)
                if full_difference > FULL_MIXTURE_TOLERANCE or abs(full_exp - exact_exp) > FULL_MIXTURE_TOLERANCE:
                    raise RuntimeError(
                        "Full historical mixture did not reproduce the exact average: "
                        f"table={full_difference}, exploitability={full_exp-exact_exp}"
                    )

                ordinary_model = active_solver.ave_policy_trainer.model
                ordinary_exp = exact_exploitability(
                    active_solver.game, model_table(active_solver.game, ordinary_model)
                )
                source = _buffer_source(active_solver)
                dataset = grouped_replay_dataset(source)
                _, empirical_exp = empirical_table(source, active_solver.game)
                incumbent_model = deepcopy(ordinary_model)
                updates = SMOKE_REFINEMENT_UPDATES if smoke else REFINEMENT_UPDATES
                fit_continuously(
                    model=incumbent_model, dataset=dataset,
                    learning_rate=REFINEMENT_LEARNING_RATE,
                    gradient_clip_norm=REFINEMENT_GRADIENT_CLIP_NORM,
                    evaluation_updates=(0, int(updates)),
                    callback=lambda *_: None,
                )
                incumbent_exp = exact_exploitability(
                    active_solver.game, model_table(active_solver.game, incumbent_model)
                )
                common = {
                    "seed": int(seed), "checkpoint_id": checkpoint_id,
                    "checkpoint_type": str(target["checkpoint_type"]),
                    "checkpoint_target_active_hours": target.get("target_active_hours"),
                    "actual_active_hours": active_seconds / 3600.0,
                    "nodes_touched": int(checkpoint["nodes_touched"]),
                    "completed_iteration": int(checkpoint["iteration"]),
                    "historical_snapshot_count": len(entries),
                    "historical_storage_bytes": sum(entry.size_bytes for entry in entries),
                    "full_exact_table_max_abs": full_difference,
                }
                arms = [
                    (CANDIDATE_ID, CANDIDATE_LABEL, full_exp, len(entries), len(entries)),
                    (ORDINARY_NEURAL_ID, ORDINARY_NEURAL_LABEL, ordinary_exp, None, None),
                    (INCUMBENT_NEURAL_ID, INCUMBENT_NEURAL_LABEL, incumbent_exp, None, None),
                    (EMPIRICAL_ID, EMPIRICAL_LABEL, empirical_exp, None, None),
                    (EXACT_ID, EXACT_LABEL, exact_exp, None, None),
                ]
                bounded_selections = {}
                for capacity in BOUNDED_CAPACITIES:
                    bounded_table, selected = bounded_mixture_table(
                        active_solver.game, entries, int(capacity),
                        seed=int(seed) * 10_000 + int(capacity),
                    )
                    bounded_exp = exact_exploitability(active_solver.game, bounded_table)
                    arm_id = f"historical_reservoir_{capacity}"
                    arms.append((
                        arm_id, f"Bounded historical mixture (K={capacity})",
                        bounded_exp, int(capacity), len(set(selected)),
                    ))
                    selection_rows.append({
                        "seed": int(seed), "checkpoint_id": checkpoint_id,
                        "capacity": int(capacity), "selected_iterations": selected,
                        "unique_selected_iterations": len(set(selected)),
                    })
                    bounded_selections[str(capacity)] = selected
                for arm_id, arm_label, exploitability, capacity, unique in arms:
                    metric_rows.append({
                        **common, "arm_id": arm_id, "arm_label": arm_label,
                        "exploitability": float(exploitability),
                        "arm_minus_incumbent": float(exploitability - incumbent_exp),
                        "arm_minus_exact": float(exploitability - exact_exp),
                        "bounded_capacity": capacity,
                        "unique_retained_snapshots": unique,
                    })
                write_csv(worker_dir / "policy_metrics.csv", metric_rows)
                write_json(worker_dir / "reservoir_selections.json", {"rows": selection_rows})
                historical_checkpoint = worker_dir / "historical_policy_checkpoints" / (
                    f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.json"
                )
                write_json(historical_checkpoint, {
                    "schema_version": 1,
                    "policy_type": "weighted_historical_regret_network_mixture",
                    "seed": int(seed), "checkpoint_id": checkpoint_id,
                    "completed_iteration": int(checkpoint["iteration"]),
                    "sampling_rule": "sample one component independently per player per hand",
                    "components": [
                        {
                            "iteration": entry.iteration,
                            "iteration_weight": entry.iteration_weight,
                            "relative_path": str(entry.path.relative_to(worker_dir)),
                            "sha256": entry.sha256,
                        }
                        for entry in entries
                    ],
                    "bounded_slot_iterations": bounded_selections,
                })
                captured[checkpoint_id] = _save_incumbent(
                    solver=active_solver, model=incumbent_model, seed=seed,
                    target=target, checkpoint=checkpoint, config=config,
                    worker_dir=worker_dir, commit=commit,
                )
            finally:
                _restore_rng(saved_rng)

            if checkpoint_id in state_ids:
                state_path = worker_dir / "training_states" / (
                    f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id}.pt"
                )
                ordered = [
                    captured[str(row["checkpoint_id"])] for row in schedule
                    if str(row["checkpoint_id"]) in captured
                ]
                payload = build_training_state(
                    active_solver, variant_id=CANDIDATE_ID, seed=seed,
                    checkpoint_id=checkpoint_id, active_seconds=active_seconds,
                    repository_commit=commit, config=config,
                    captured_snapshots=ordered, state_type=TRAINING_STATE_TYPE,
                    experiment_name=EXPERIMENT_NAME,
                )
                save_training_state(state_path, payload)
                _sync_remote_resume_point(worker_dir)
            excluded_seconds += time.perf_counter() - started
            LOGGER.info(
                "Evaluated %s seed %s at iteration %s and %.2f active hours",
                checkpoint_id, seed, checkpoint["iteration"], active_seconds / 3600.0,
            )
        if len(captured) == len(schedule):
            active_solver.target_nodes_touched = int(active_solver.nodes_touched)

    try:
        solver.solve(post_checkpoint_callback=capture) if len(captured) < len(schedule) else None
        if set(captured) != {str(row["checkpoint_id"]) for row in schedule}:
            raise RuntimeError("Worker stopped before all scheduled checkpoints were captured")
        for record in captured.values():
            path = worker_dir / record["relative_path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise RuntimeError(f"Missing or corrupt incumbent snapshot: {path}")
            validate_playable_snapshot(UCV_POLICY_LOADER_ID, path)
        archive_rows = archive.manifest_rows(worker_dir)
        write_json(worker_dir / "historical_manifest.json", {"snapshots": archive_rows})
        historical_checkpoints = []
        for target in schedule:
            path = worker_dir / "historical_policy_checkpoints" / (
                f"{CANDIDATE_ID}_seed_{seed}_{target['checkpoint_id']}.json"
            )
            if not path.is_file():
                raise RuntimeError(f"Missing historical policy checkpoint: {path}")
            historical_checkpoints.append({
                "checkpoint_id": str(target["checkpoint_id"]),
                "relative_path": str(path.relative_to(worker_dir)),
                "sha256": sha256(path), "size_bytes": int(path.stat().st_size),
            })
        if smoke:
            _validate_historical_snapshot(worker_dir, archive)
        result = {
            "schema_version": 1, "experiment_name": EXPERIMENT_NAME,
            "candidate_id": CANDIDATE_ID, "candidate_label": CANDIDATE_LABEL,
            "seed": int(seed), "smoke": bool(smoke),
            "checkpoint_schedule": list(schedule), "repository_commit": commit,
            "config": config,
            "incumbent_snapshots": [captured[str(row["checkpoint_id"])] for row in schedule],
            "historical_policy_checkpoints": historical_checkpoints,
            "historical_snapshot_count": len(archive.entries),
            "historical_storage_bytes": sum(row.size_bytes for row in archive.entries),
            "peak_rss_mb": _peak_rss_mb(),
            "artifacts": {
                "policy_metrics": "policy_metrics.csv",
                "reservoir_selections": "reservoir_selections.json",
                "historical_manifest": "historical_manifest.json",
            },
            "status": "complete",
        }
        write_json(worker_dir / "worker_result.json", result)
        write_json(worker_dir / "SUCCESS.json", {"status": "complete", "seed": int(seed)})
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["run_worker"]
