"""Run one source-trajectory worker for Experiment 27."""

from __future__ import annotations

import gc
from pathlib import Path
import resource
import subprocess
import sys
from typing import Mapping

import numpy as np
import pyspiel
import torch
from vr_deep_cfr.solver import MLP

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    read_training_state,
)

from .config import (
    ARMS,
    EXPERIMENT_NAME,
    ORACLE_CROSS_ENTROPY,
    RESET_CROSS_ENTROPY,
    RESET_MSE,
    SOURCE_CHECKPOINTS,
    WARM_CROSS_ENTROPY,
    WARM_MSE,
    fit_replicates,
)
from .distill import (
    empirical_table,
    evaluate_model_state,
    extract_source,
    fit_oracle_policy,
    fit_reservoir_policy,
)


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _source_path(source_root: Path, *, seed: int, checkpoint_id: str, smoke: bool) -> Path:
    if smoke:
        role = "smoke_time_02" if checkpoint_id == "time_24h" else "smoke_time_03"
    else:
        role = checkpoint_id
    filename = f"averaged_critic_target_seed_{seed}_{role}.pt"
    matches = list(Path(source_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source state named {filename}, found {len(matches)}")
    return matches[0]


def _save_fit(
    worker_dir: Path,
    *,
    arm_id: str,
    checkpoint_id: str,
    seed: int,
    fit_replicate: int,
    fit_seed: int,
    source,
    state: Mapping,
) -> dict:
    path = worker_dir / "policies" / f"{arm_id}_{checkpoint_id}_fit_{fit_replicate}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "arm_id": arm_id,
            "checkpoint_id": checkpoint_id,
            "source_seed": int(seed),
            "fit_replicate": int(fit_replicate),
            "fit_seed": int(fit_seed),
            "network_layers": list(source.network_layers),
            "input_size": int(source.infostates.shape[1]),
            "output_size": int(source.policies.shape[1]),
            "model": {name: value.detach().cpu() for name, value in state["model"].items()},
        },
        path,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if (
        loaded.get("experiment_name") != EXPERIMENT_NAME
        or loaded.get("arm_id") != arm_id
        or int(loaded.get("source_seed", -1)) != int(seed)
    ):
        raise ValueError(f"Saved policy failed metadata reload validation: {path}")
    verifier = MLP(
        int(loaded["input_size"]),
        list(loaded["network_layers"]),
        int(loaded["output_size"]),
    )
    verifier.load_state_dict(loaded["model"])
    return {
        "arm_id": arm_id,
        "checkpoint_id": checkpoint_id,
        "source_seed": int(seed),
        "fit_replicate": int(fit_replicate),
        "relative_path": str(path.relative_to(worker_dir)),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def run_worker(*, seed: int, source_root: Path, worker_dir: Path, smoke: bool) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game("leduc_poker")
    train_steps_override = 3 if smoke else None
    validation_samples = 16 if smoke else 50_000
    metric_rows, information_rows, decomposition_rows, policies = [], [], [], []
    warm_states = {}
    source_records = []

    for checkpoint_id in SOURCE_CHECKPOINTS:
        path = _source_path(source_root, seed=seed, checkpoint_id=checkpoint_id, smoke=smoke)
        source_result_paths = list(path.parents[1].glob("worker_result.json"))
        expected_sha = None
        if source_result_paths:
            source_result = read_json(source_result_paths[0])
            expected = [
                row for row in source_result.get("training_states", ())
                if row.get("filename") == path.name
            ]
            if len(expected) != 1:
                raise ValueError(f"Experiment 25 source manifest does not identify {path.name}")
            expected_sha = expected[0]["sha256"]
        observed_sha = sha256(path)
        if expected_sha is not None and observed_sha != expected_sha:
            raise ValueError(f"Experiment 25 source checksum differs for {path.name}")
        payload = read_training_state(path)
        source = extract_source(payload, seed=seed, checkpoint_id=(payload["checkpoint_id"]))
        del payload
        gc.collect()
        source_records.append(
            {
                "checkpoint_id": checkpoint_id,
                "source_checkpoint_id": source.checkpoint_id,
                "path": str(path.resolve()),
                "sha256": observed_sha,
                "size_bytes": int(path.stat().st_size),
                "reservoir_size": int(len(source.infostates)),
                "iteration": int(source.iteration),
            }
        )
        exact_value = exact_exploitability(game, source.exact_table)
        empirical, empirical_value = empirical_table(source, game)
        source_value, _ = evaluate_model_state(source, game, source.source_model_state)
        decomposition_rows.extend(
            [
                {"source_seed": seed, "checkpoint_id": checkpoint_id, "policy": "exact_tabular", "exploitability": exact_value},
                {"source_seed": seed, "checkpoint_id": checkpoint_id, "policy": "empirical_tabular_reservoir", "exploitability": empirical_value},
                {"source_seed": seed, "checkpoint_id": checkpoint_id, "policy": "archived_neural_policy", "exploitability": source_value},
            ]
        )

        steps = train_steps_override or source.train_steps
        for replicate in fit_replicates(smoke=smoke):
            for arm_id in (RESET_MSE, RESET_CROSS_ENTROPY):
                treatment = ARMS[arm_id]
                # Common random numbers isolate the loss and warm-start
                # interventions from initialisation and minibatch variation.
                fit_seed = int(seed) * 100 + int(replicate)
                metrics, state, details = fit_reservoir_policy(
                    source,
                    game,
                    loss_name=treatment["loss"],
                    fit_seed=fit_seed,
                    train_steps=steps,
                    validation_samples=validation_samples,
                )
                row = {
                    "source_seed": seed,
                    "checkpoint_id": checkpoint_id,
                    "fit_replicate": replicate,
                    "fit_seed": fit_seed,
                    "arm_id": arm_id,
                    "arm_label": treatment["label"],
                    "loss": treatment["loss"],
                    "warm_start": False,
                    "oracle_targets": False,
                    "exact_exploitability": exact_value,
                    "empirical_tabular_exploitability": empirical_value,
                    "archived_neural_exploitability": source_value,
                    "distillation_gap": metrics["exploitability"] - exact_value,
                    **metrics,
                }
                metric_rows.append(row)
                information_rows.extend({**detail, **{key: row[key] for key in ("source_seed", "checkpoint_id", "fit_replicate", "arm_id")}} for detail in details)
                policies.append(
                    _save_fit(
                        worker_dir,
                        arm_id=arm_id,
                        checkpoint_id=checkpoint_id,
                        seed=seed,
                        fit_replicate=replicate,
                        fit_seed=fit_seed,
                        source=source,
                        state=state,
                    )
                )
                if checkpoint_id == "time_24h":
                    warm_states[(treatment["loss"], replicate)] = state

            if checkpoint_id == "time_36h":
                for arm_id, parent_loss in (
                    (WARM_MSE, "mse"),
                    (WARM_CROSS_ENTROPY, "cross_entropy"),
                ):
                    treatment = ARMS[arm_id]
                    fit_seed = int(seed) * 100 + int(replicate)
                    metrics, state, details = fit_reservoir_policy(
                        source,
                        game,
                        loss_name=parent_loss,
                        fit_seed=fit_seed,
                        train_steps=steps,
                        validation_samples=validation_samples,
                        initial_state=warm_states[(parent_loss, replicate)],
                    )
                    row = {
                        "source_seed": seed,
                        "checkpoint_id": checkpoint_id,
                        "fit_replicate": replicate,
                        "fit_seed": fit_seed,
                        "arm_id": arm_id,
                        "arm_label": treatment["label"],
                        "loss": parent_loss,
                        "warm_start": True,
                        "oracle_targets": False,
                        "exact_exploitability": exact_value,
                        "empirical_tabular_exploitability": empirical_value,
                        "archived_neural_exploitability": source_value,
                        "distillation_gap": metrics["exploitability"] - exact_value,
                        **metrics,
                    }
                    metric_rows.append(row)
                    information_rows.extend({**detail, **{key: row[key] for key in ("source_seed", "checkpoint_id", "fit_replicate", "arm_id")}} for detail in details)
                    policies.append(
                        _save_fit(worker_dir, arm_id=arm_id, checkpoint_id=checkpoint_id, seed=seed, fit_replicate=replicate, fit_seed=fit_seed, source=source, state=state)
                    )

                fit_seed = int(seed) * 100 + int(replicate) + 50
                metrics, state, details = fit_oracle_policy(
                    source, game, fit_seed=fit_seed, train_steps=steps
                )
                treatment = ARMS[ORACLE_CROSS_ENTROPY]
                row = {
                    "source_seed": seed,
                    "checkpoint_id": checkpoint_id,
                    "fit_replicate": replicate,
                    "fit_seed": fit_seed,
                    "arm_id": ORACLE_CROSS_ENTROPY,
                    "arm_label": treatment["label"],
                    "loss": "cross_entropy",
                    "warm_start": False,
                    "oracle_targets": True,
                    "exact_exploitability": exact_value,
                    "empirical_tabular_exploitability": empirical_value,
                    "archived_neural_exploitability": source_value,
                    "distillation_gap": metrics["exploitability"] - exact_value,
                    **metrics,
                }
                metric_rows.append(row)
                information_rows.extend({**detail, **{key: row[key] for key in ("source_seed", "checkpoint_id", "fit_replicate", "arm_id")}} for detail in details)
                policies.append(
                    _save_fit(worker_dir, arm_id=ORACLE_CROSS_ENTROPY, checkpoint_id=checkpoint_id, seed=seed, fit_replicate=replicate, fit_seed=fit_seed, source=source, state=state)
                )
        del source, empirical
        gc.collect()

    write_csv(worker_dir / "fit_metrics.csv", metric_rows)
    write_csv(worker_dir / "information_set_errors.csv", information_rows)
    write_csv(worker_dir / "source_decomposition.csv", decomposition_rows)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": _repository_commit(),
        "source_states": source_records,
        "policies": policies,
        "peak_rss_mb": _peak_rss_mb(),
        "artifacts": {
            "fit_metrics": "fit_metrics.csv",
            "information_set_errors": "information_set_errors.csv",
            "source_decomposition": "source_decomposition.csv",
        },
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(worker_dir / "SUCCESS.json", {"status": "complete", "source_seed": seed})
    return result


__all__ = ["run_worker"]
