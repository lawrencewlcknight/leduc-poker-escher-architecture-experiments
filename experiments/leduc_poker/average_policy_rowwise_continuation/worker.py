"""Per-reservoir worker for Experiment 42."""

from __future__ import annotations

from copy import deepcopy
import gc
from pathlib import Path
import resource
import subprocess
import sys
import time

import pyspiel
import torch

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import _policy_errors
from experiments.leduc_poker.causal_rare_state_audit.audit import extract_promoted_source
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.policy_post_training.core import make_model, model_table
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    read_training_state,
)

from .config import (
    ARM_IDS,
    EQUAL_EXAMPLE_ARM,
    EQUAL_UPDATE_ARM,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    GAME_NAME,
    SOURCE_CHECKPOINT,
    SOURCE_SHA256,
    contract_manifest,
    runtime_config,
)
from .fitting import diagnostic_row_objective, fit_rowwise_continuously


def repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def source_path(source_root: Path, *, seed: int, smoke: bool):
    role = "smoke_time_03" if smoke else SOURCE_CHECKPOINT
    filename = f"promoted_ucv_cross_entropy_seed_{seed}_{role}.pt"
    matches = list(Path(source_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source {filename}; found {len(matches)}")
    return matches[0], role


def _save_policy(path: Path, *, source, seed: int, arm_id: str,
                 progress: int, examples_seen: int, optimizer_steps: int,
                 model) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "type": "rowwise_average_policy_continuation",
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": 29,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_seed": int(seed),
        "arm_id": arm_id,
        "equivalent_update": int(progress),
        "examples_seen": int(examples_seen),
        "optimizer_steps": int(optimizer_steps),
        "input_size": int(source.infostates.shape[1]),
        "output_size": int(source.policies.shape[1]),
        "network_layers": list(source.network_layers),
        "model": deepcopy(
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
        ),
    }
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    verifier = make_model(source, state=loaded["model"])
    if loaded["arm_id"] != arm_id or int(loaded["equivalent_update"]) != progress:
        raise ValueError(f"Saved policy failed metadata validation: {path}")
    del verifier
    return {
        "arm_id": arm_id,
        "equivalent_update": int(progress),
        "examples_seen": int(examples_seen),
        "optimizer_steps": int(optimizer_steps),
        "relative_path": str(path),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def run_worker(*, seed: int, source_root: Path, worker_dir: Path,
               smoke: bool) -> dict:
    started = time.perf_counter()
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = runtime_config(seed=seed, smoke=smoke)
    path, source_role = source_path(source_root, seed=seed, smoke=smoke)
    source_digest = sha256(path)
    if not smoke and source_digest != SOURCE_SHA256[int(seed)]:
        raise ValueError(
            f"Experiment 29 source digest differs for seed {seed}: {source_digest}"
        )
    payload = read_training_state(path)
    source_repository_commit = str(payload["repository_commit"])
    source = extract_promoted_source(
        payload, seed=seed, checkpoint_id=SOURCE_CHECKPOINT,
        source_checkpoint_id=source_role,
    )
    del payload
    gc.collect()

    game = pyspiel.load_game(GAME_NAME)
    archived_model = make_model(source)
    archived_exploitability = exact_exploitability(
        game, model_table(game, archived_model)
    )
    exact_average_exploitability = exact_exploitability(game, source.exact_table)

    metric_rows = []
    snapshots = []
    arm_runtime = {}
    required = set(int(value) for value in config["required_checkpoint_updates"])
    for arm_index, arm_id in enumerate(ARM_IDS):
        model = make_model(source)
        arm_started = time.perf_counter()

        def record(progress, examples_seen, optimizer_steps, training_loss,
                   fitted_model, *, _arm=arm_id):
            fitted_table = model_table(game, fitted_model)
            exploitability = exact_exploitability(game, fitted_table)
            errors = _policy_errors(
                source.exact_table, fitted_table, source.exact_denominators
            )
            metric_rows.append(
                {
                    "source_seed": int(seed),
                    "arm_id": _arm,
                    "equivalent_update": int(progress),
                    "examples_seen": int(examples_seen),
                    "optimizer_steps": int(optimizer_steps),
                    "exploitability": float(exploitability),
                    "change_from_archived_policy": (
                        float(exploitability) - archived_exploitability
                    ),
                    "gap_to_exact_tabular_average": (
                        float(exploitability) - exact_average_exploitability
                    ),
                    "last_minibatch_loss": float(training_loss),
                    "diagnostic_row_cross_entropy": diagnostic_row_objective(
                        fitted_model, source
                    ),
                    "fit_elapsed_seconds": time.perf_counter() - arm_started,
                    **{key: value for key, value in errors.items() if key != "rows"},
                }
            )
            if int(progress) in required:
                snapshot_path = (
                    worker_dir / "policies"
                    / f"{_arm}_seed_{seed}_equivalent_{int(progress):04d}.pt"
                )
                snapshot = _save_policy(
                    snapshot_path, source=source, seed=seed, arm_id=_arm,
                    progress=int(progress), examples_seen=int(examples_seen),
                    optimizer_steps=int(optimizer_steps), model=fitted_model,
                )
                snapshot["relative_path"] = str(
                    snapshot_path.relative_to(worker_dir)
                )
                snapshots.append(snapshot)

        fit_rowwise_continuously(
            model=model,
            source=source,
            arm_id=arm_id,
            unique_information_states=config["unique_information_states"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            gradient_clip_norm=config["gradient_clip_norm"],
            diagnostic_equivalent_updates=config[
                "diagnostic_equivalent_updates"
            ],
            sampling_seed=(
                int(seed) * 1009 + (arm_index + 1) * 104729
            ),
            callback=record,
        )
        arm_runtime[arm_id] = time.perf_counter() - arm_started
        del model
        gc.collect()

    write_csv(worker_dir / "fit_metrics.csv", metric_rows)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": repository_commit(),
        "contract": contract_manifest(),
        "runtime_schedule": {
            "diagnostic_equivalent_updates": list(
                config["diagnostic_equivalent_updates"]
            ),
            "required_checkpoint_updates": list(
                config["required_checkpoint_updates"]
            ),
            "unique_information_states": int(
                config["unique_information_states"]
            ),
            "batch_size": int(config["batch_size"]),
        },
        "source": {
            "path": str(path.resolve()),
            "sha256": source_digest,
            "size_bytes": int(path.stat().st_size),
            "repository_commit": source_repository_commit,
        },
        "dataset": {
            "replay_rows": int(len(source.infostates)),
            "grouping_performed_in_training_path": False,
            "training_requires_game_tree": False,
        },
        "baselines": {
            "archived_neural_policy": archived_exploitability,
            "exact_tabular_average": exact_average_exploitability,
        },
        "arm_runtime_seconds": arm_runtime,
        "snapshots": snapshots,
        "peak_rss_mb": peak_rss_mb(),
        "wall_clock_seconds": time.perf_counter() - started,
        "artifacts": {"metrics": "fit_metrics.csv"},
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(
        worker_dir / "SUCCESS.json",
        {
            "status": "complete",
            "experiment_name": EXPERIMENT_NAME,
            "source_seed": int(seed),
            "repository_commit": result["repository_commit"],
        },
    )
    return result


__all__ = ["peak_rss_mb", "repository_commit", "run_worker", "source_path"]
