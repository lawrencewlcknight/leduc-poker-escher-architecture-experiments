"""Experiment 35 learner plus opt-in caching, passive timers and lean outputs."""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from experiments.leduc_poker.grouped_wide_policy_confirmation import worker as base
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import (
    sha256, validate_records, validate_playable_snapshot, write_csv, write_json,
)
from .config import (CANDIDATE_CONFIG, CANDIDATE_ID, EXPERIMENT_NAME,
                     THREADS, checkpoint_schedule, contract, load_reference)


class CachedGroupedSolver(base.DiagnosticGroupedWideUCVSolver):
    """Do not override target algebra, sampling, traversal or fitting frequency."""
    def __init__(self, *args, cache_frozen_critic_targets=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.component_seconds = {}
        self.critic_cache_rows = 0
        self.critic_cache_peak_bytes = 0
        self.critic_fit_calls = 0
        self._wrap(self.ave_policy_trainer, "train_model", "candidate_policy_fit")
        self._wrap(self.paired_legacy_policy_trainer, "train_model", "legacy_policy_fit")
        self._wrap(self.calibration_trainer, "train_model", "calibration_fit")
        for trainer in self.regret_trainers:
            self._wrap(trainer, "train_model", "regret_fit")
        for trainer in self.q_value_trainer.members:
            trainer.cache_frozen_targets = bool(cache_frozen_critic_targets)
            self._wrap(trainer, "train_model", "critic_fit", critic=True)

    def _wrap(self, trainer, name, label, critic=False):
        original = getattr(trainer, name)
        def timed(*args, **kwargs):
            started = time.perf_counter()
            eligible = (not critic or (len(trainer.buffer) > 0 and trainer.train_steps > 0
                        and (trainer.batch_size < 0 or len(trainer.buffer) >= trainer.batch_size)))
            try:
                return original(*args, **kwargs)
            finally:
                self.component_seconds[label] = self.component_seconds.get(label, 0.) + time.perf_counter() - started
                if critic:
                    self.critic_fit_calls += 1
                    stats = getattr(trainer, "last_target_cache_stats", {})
                    if trainer.cache_frozen_targets and eligible:
                        if stats.get("cache_lifetime") != "this_fit_only":
                            raise RuntimeError("Critic cache was not rebuilt for this fitting block")
                        self.component_seconds["cache_build_subset_of_critic_fit"] = (
                            self.component_seconds.get("cache_build_subset_of_critic_fit", 0.)
                            + float(stats["cache_build_seconds"]))
                        self.critic_cache_rows += int(stats["cached_rows"])
                        self.critic_cache_peak_bytes = max(self.critic_cache_peak_bytes, int(stats["cache_bytes"]))
        setattr(trainer, name, timed)

    def collect_training_data(self, player):
        # Collection may trigger an early policy checkpoint: avoid double-counting it.
        started, excluded_before = time.perf_counter(), getattr(self, "checkpoint_seconds", 0.)
        try:
            return super().collect_training_data(player)
        finally:
            elapsed = time.perf_counter() - started - (getattr(self, "checkpoint_seconds", 0.) - excluded_before)
            self.component_seconds["collection_excluding_nested_checkpoints"] = (
                self.component_seconds.get("collection_excluding_nested_checkpoints", 0.) + elapsed)

    def _run_checkpoint(self, **kwargs):
        started = time.perf_counter()
        try:
            return super()._run_checkpoint(**kwargs)
        finally:
            self.checkpoint_seconds = getattr(self, "checkpoint_seconds", 0.) + time.perf_counter() - started

    def evaluate(self, **kwargs):
        started = time.perf_counter()
        try:
            return super().evaluate(**kwargs)
        finally:
            self.component_seconds["evaluation_inclusive"] = self.component_seconds.get("evaluation_inclusive", 0.) + time.perf_counter() - started


def make_solver(seed, config):
    return base._make_solver(seed, config, solver_class=CachedGroupedSolver)


def runtime():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "threads": torch.get_num_threads()}


def configuration(smoke):
    c = deepcopy(CANDIDATE_CONFIG)
    if smoke:
        base._smoke_overrides(c)
    return c


def worker_directory(root, index, seed):
    return Path(root) / "workers" / f"task_{index:03d}_{CANDIDATE_ID}_seed_{seed}"


def sync_worker(directory):
    uri = os.environ.get("EXP48_REMOTE_WORKER")
    if not uri:
        return
    if not uri.startswith("gs://") or any(c.isspace() for c in uri):
        raise ValueError("Invalid EXP48_REMOTE_WORKER")
    command = ["gcloud", "storage", "rsync", "--recursive", "--exclude", r".*(\.pt|\.tmp)$",
               str(directory), uri]
    for attempt in range(3):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def verify_completed(directory, expected_manifest):
    marker = directory / "run_manifest.json"
    if not marker.exists():
        if any(directory.iterdir()):
            raise ValueError("Non-empty worker directory without run manifest; use a new RUN_ID")
        return None
    if json.loads(marker.read_text()) != expected_manifest:
        raise ValueError("Existing worker has a different source/code/runtime contract; use a new RUN_ID")
    if not (directory / "SUCCESS.json").exists():
        raise ValueError("Interrupted run has no full training state; use a new RUN_ID, not a silent restart")
    success = json.loads((directory / "SUCCESS.json").read_text())
    for name, digest in success["files"].items():
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing/corrupt completed output: {name}")
    return json.loads((directory / "worker_result.json").read_text())


def run_worker(*, seed, directory, smoke=False):
    torch.set_num_threads(THREADS if not smoke else 1)
    c = configuration(smoke)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    commit = base._repository_commit()
    manifest = {"contract": contract(smoke), "seed": seed, "config": c,
                "smoke": smoke, "commit": commit, "runtime": runtime()}
    # JSON canonicalisation normalises tuples before comparison with saved metadata.
    manifest = json.loads(json.dumps(manifest))
    existing = verify_completed(directory, manifest)
    if existing is not None:
        return existing
    write_json(directory / "run_manifest.json", manifest)
    sync_worker(directory)  # Publish partial-run marker BEFORE expensive training.
    started = time.perf_counter()
    solver = make_solver(seed, c)
    setup_seconds = time.perf_counter() - started
    solver.target_nodes_touched = 2**62
    schedule = checkpoint_schedule(smoke)
    captured, milestones, timing_rows = {}, {}, []
    persistence_seconds = 0.
    reference = ({"iteration": 1, "nodes": 1, "actual_hours": 0.001} if smoke
                 else load_reference()["milestones"][str(seed)])

    def persist():
        curves = base._curve_rows(solver.checkpoint_rows, seed=seed)
        for row, raw in zip(curves, solver.checkpoint_rows):
            row["recorded_active_seconds"] = raw.get("recorded_active_seconds", "")
        write_csv(directory / "checkpoint_curves.csv", curves)
        write_csv(directory / "component_timings.csv", timing_rows)
        write_json(directory / "checkpoint_manifest.json", {"snapshots": list(captured.values())})
        write_json(directory / "matched_workload.json", milestones)
        sync_worker(directory)

    def capture(current, raw):
        nonlocal persistence_seconds
        # Intentionally EXACTLY the Experiment 35 clock, not a new time definition.
        active = max(0., float(raw["wall_clock_seconds"]) - persistence_seconds
                     - current._cumulative_factorial_diagnostic_seconds
                     - current._cumulative_confirmation_diagnostic_seconds)
        current.checkpoint_rows[-1]["recorded_active_seconds"] = active
        checkpoint = dict(raw, active_seconds=active, checkpoint_row_index=len(current.checkpoint_rows)-1)
        timing_rows.append({"seed": seed, "iteration": current.num_iteration,
            "nodes_touched": current.nodes_touched, "recorded_active_seconds": active,
            "raw_solve_seconds": raw["wall_clock_seconds"],
            "persistence_seconds": persistence_seconds, "setup_seconds": setup_seconds,
            "excluded_factorial_diagnostics": current._cumulative_factorial_diagnostic_seconds,
            "excluded_confirmation_diagnostics": current._cumulative_confirmation_diagnostic_seconds,
            "critic_fit_calls": current.critic_fit_calls, "critic_cache_rows": current.critic_cache_rows,
            "critic_cache_peak_bytes": current.critic_cache_peak_bytes, **current.component_seconds})
        milestones_changed = False
        for field, value in (("iteration", current.num_iteration), ("nodes", current.nodes_touched)):
            if field not in milestones and value >= reference[field]:
                milestones[field] = {"seed": seed, "criterion": field, "reference": reference,
                    "observed_iteration": current.num_iteration, "observed_nodes": current.nodes_touched,
                    "active_hours": active / 3600.,
                    "hours_saved": reference["actual_hours"] - active / 3600.,
                    "exploitability": float(raw["neural_average_exploitability"])}
                milestones_changed = True
        new_targets = [t for t in schedule if t["checkpoint_id"] not in captured
                       and base._target_reached(t, active_seconds=active, nodes=current.nodes_touched)]
        if new_targets or milestones_changed:
            saving = time.perf_counter()
            for target in new_targets:
                captured[target["checkpoint_id"]] = base._save_policy(
                    solver=current, seed=seed, target=target, checkpoint=checkpoint, config=c,
                    worker_dir=directory, commit=commit, experiment_id=48, arm="cached_followup")
            persist()
            persistence_seconds += time.perf_counter() - saving
        if len(captured) == len(schedule):
            current.target_nodes_touched = int(current.nodes_touched)

    try:
        solver.solve(post_checkpoint_callback=capture)
        if set(captured) != {t["checkpoint_id"] for t in schedule}:
            raise RuntimeError("Safety iteration cap reached before the complete 36-hour schedule")
        records = [captured[t["checkpoint_id"]] for t in schedule]
        validate_records(records, candidate_id=CANDIDATE_ID, seed=seed, schedule=schedule)
        for record in records:
            validate_playable_snapshot(base.UCV_POLICY_LOADER_ID, directory / record["relative_path"])
        curves = base._curve_rows(solver.checkpoint_rows, seed=seed)
        base._assert_candidate(c, curves, smoke=smoke)
        if solver.critic_fit_calls != 2 * solver.num_iteration or solver.critic_cache_rows <= 0:
            raise RuntimeError("Cached critic fitting was not exercised each iteration")
        for name, rows in (
            ("information_action_diagnostics", solver.information_action_rows()),
            ("beta_histogram", solver.beta_histogram_rows()),
            ("critic_error_subsequent_local_regret", solver.critic_subsequent_regret_rows()),
            ("exact_q_oracle_diagnostics", solver.q_oracle_diagnostic_rows)):
            write_csv(directory / "diagnostics" / f"{name}.csv", rows)
        persist()
        result = {"status": "complete", "experiment_name": EXPERIMENT_NAME, "experiment_id": 48,
            "candidate_id": CANDIDATE_ID, "candidate_label": "UCV-ESCHER (cached grouped)",
            "seed": seed, "smoke": smoke, "repository_commit": commit, "config": c,
            "checkpoint_schedule": list(schedule), "snapshots": records, "training_states": [],
            "peak_rss_mb": base._peak_rss_mb(), "runtime": runtime(),
            "matched_workload": milestones, "worker_elapsed_seconds": time.perf_counter()-started,
            "artifacts": {"checkpoint_curves": "checkpoint_curves.csv",
                          "component_timings": "component_timings.csv"}}
        write_json(directory / "worker_result.json", result)
        files = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*") if p.is_file()}
        write_json(directory / "SUCCESS.json", {"status": "complete", "files": files})
        sync_worker(directory)
        return result
    finally:
        del solver
        gc.collect()
