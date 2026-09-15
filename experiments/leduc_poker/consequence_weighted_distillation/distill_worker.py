"""Run Experiment 33 redistillation arms for one source trajectory."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping

import pyspiel
import torch
from vr_deep_cfr.solver import MLP

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    empirical_table,
    evaluate_model_state,
)
from experiments.leduc_poker.causal_rare_state_audit.audit import extract_promoted_source
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    read_training_state,
)

from .common import (
    peak_rss_mb,
    read_csv,
    repository_commit,
    source_path,
    validate_source_manifest,
)
from .config import (
    ARM_LABELS,
    ARM_ORDER,
    EXPERIMENT_NAME,
    FINE_TUNE_STEPS,
    GAME_NAME,
    SMOKE_FINE_TUNE_STEPS,
    SMOKE_TRAIN_STEPS,
    SMOKE_VALIDATION_SAMPLES,
    SOURCE_CHECKPOINTS,
    TRAIN_STEPS,
    VALIDATION_SAMPLES,
    fit_replicates,
)
from .distill import fit_consequence_policy


def _save_policy(
    worker_dir: Path,
    *,
    arm_id: str,
    checkpoint_id: str,
    seed: int,
    replicate: int,
    fit_seed: int,
    source,
    state: Mapping,
) -> dict:
    path = worker_dir / "policies" / f"{arm_id}_{checkpoint_id}_fit_{replicate}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "experiment_name": EXPERIMENT_NAME,
            "arm_id": arm_id,
            "checkpoint_id": checkpoint_id,
            "source_seed": int(seed),
            "fit_replicate": int(replicate),
            "fit_seed": int(fit_seed),
            "network_layers": list(source.network_layers),
            "input_size": int(source.infostates.shape[1]),
            "output_size": int(source.policies.shape[1]),
            "model": {name: value.detach().cpu() for name, value in state["model"].items()},
        },
        path,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    verifier = MLP(loaded["input_size"], loaded["network_layers"], loaded["output_size"])
    verifier.load_state_dict(loaded["model"])
    return {
        "arm_id": arm_id,
        "checkpoint_id": checkpoint_id,
        "source_seed": int(seed),
        "fit_replicate": int(replicate),
        "relative_path": str(path.relative_to(worker_dir)),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def run_distill_worker(
    *,
    seed: int,
    source_root: Path,
    proxy_root: Path,
    selection_path: Path,
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    selection = read_json(selection_path)
    if selection.get("status") != "complete":
        raise ValueError("Proxy selection is incomplete")
    selected_proxy_id = str(selection["selected_proxy_id"])
    proxy_paths = list(Path(proxy_root).rglob("proxy_information_sets.csv"))
    matching = []
    for path in proxy_paths:
        rows = read_csv(path)
        if rows and int(rows[0]["source_seed"]) == int(seed):
            matching.append((path, rows))
    if len(matching) != 1:
        raise FileNotFoundError(f"Expected one proxy table for seed {seed}, found {len(matching)}")
    proxy_path, proxy_rows = matching[0]
    game = pyspiel.load_game(GAME_NAME)
    train_steps = SMOKE_TRAIN_STEPS if smoke else TRAIN_STEPS
    fine_tune_steps = SMOKE_FINE_TUNE_STEPS if smoke else FINE_TUNE_STEPS
    validation_samples = SMOKE_VALIDATION_SAMPLES if smoke else VALIDATION_SAMPLES
    metrics_rows, information_rows, sampling_rows, decomposition_rows = [], [], [], []
    policies, source_records = [], []

    for checkpoint_id in SOURCE_CHECKPOINTS:
        path, role = source_path(
            source_root, seed=seed, checkpoint_id=checkpoint_id, smoke=smoke
        )
        provenance = validate_source_manifest(path)
        payload = read_training_state(path)
        source = extract_promoted_source(
            payload,
            seed=seed,
            checkpoint_id=checkpoint_id,
            source_checkpoint_id=role,
        )
        exact_value = exact_exploitability(game, source.exact_table)
        _, empirical_value = empirical_table(source, game)
        archived_value, _ = evaluate_model_state(source, game, source.source_model_state)
        decomposition_rows.extend(
            {
                "source_seed": int(seed),
                "checkpoint_id": checkpoint_id,
                "policy": policy,
                "exploitability": value,
            }
            for policy, value in (
                ("exact_tabular", exact_value),
                ("empirical_reservoir", empirical_value),
                ("archived_neural", archived_value),
            )
        )
        for replicate in fit_replicates(smoke=smoke):
            fit_seed = int(seed) * 100 + int(replicate)
            for arm_id in ARM_ORDER:
                fit_metrics, state, details, sampling = fit_consequence_policy(
                    source,
                    game,
                    arm_id=arm_id,
                    selected_proxy_id=selected_proxy_id,
                    proxy_rows=proxy_rows,
                    fit_seed=fit_seed,
                    train_steps=train_steps,
                    fine_tune_steps=fine_tune_steps,
                    validation_samples=validation_samples,
                )
                row = {
                    "source_seed": int(seed),
                    "checkpoint_id": checkpoint_id,
                    "fit_replicate": int(replicate),
                    "fit_seed": fit_seed,
                    "arm_id": arm_id,
                    "arm_label": ARM_LABELS[arm_id],
                    "selected_proxy_id": selected_proxy_id,
                    "exact_exploitability": exact_value,
                    "empirical_exploitability": empirical_value,
                    "archived_neural_exploitability": archived_value,
                    "distillation_gap": fit_metrics["exploitability"] - exact_value,
                    **fit_metrics,
                }
                metrics_rows.append(row)
                information_rows.extend(
                    {
                        **detail,
                        "source_seed": int(seed),
                        "checkpoint_id": checkpoint_id,
                        "fit_replicate": int(replicate),
                        "arm_id": arm_id,
                    }
                    for detail in details
                )
                if replicate == 0:
                    sampling_rows.append(
                        {
                            "source_seed": int(seed),
                            "checkpoint_id": checkpoint_id,
                            "arm_id": arm_id,
                            "selected_proxy_id": selected_proxy_id,
                            **sampling,
                        }
                    )
                policies.append(
                    _save_policy(
                        worker_dir,
                        arm_id=arm_id,
                        checkpoint_id=checkpoint_id,
                        seed=seed,
                        replicate=replicate,
                        fit_seed=fit_seed,
                        source=source,
                        state=state,
                    )
                )
        source_records.append(
            {
                "checkpoint_id": checkpoint_id,
                "source_checkpoint_id": role,
                **provenance,
                "size_bytes": int(path.stat().st_size),
            }
        )
        del payload, source
        gc.collect()

    write_csv(worker_dir / "fit_metrics.csv", metrics_rows)
    write_csv(worker_dir / "information_set_errors.csv", information_rows)
    write_csv(worker_dir / "sampling_diagnostics.csv", sampling_rows)
    write_csv(worker_dir / "source_decomposition.csv", decomposition_rows)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "stage": "distill",
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": repository_commit(),
        "selected_proxy_id": selected_proxy_id,
        "selection_sha256": sha256(selection_path),
        "proxy_sha256": sha256(proxy_path),
        "source_states": source_records,
        "policies": policies,
        "peak_rss_mb": peak_rss_mb(),
        "artifacts": {
            "fit_metrics": "fit_metrics.csv",
            "information_set_errors": "information_set_errors.csv",
            "sampling_diagnostics": "sampling_diagnostics.csv",
            "source_decomposition": "source_decomposition.csv",
        },
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(worker_dir / "SUCCESS.json", {"status": "complete", "source_seed": seed})
    return result
