"""Per-source workers for Experiment 34 development and validation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping

import numpy as np
import pyspiel
import torch

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    empirical_table,
    evaluate_model_state,
)
from experiments.leduc_poker.causal_rare_state_audit.audit import extract_promoted_source
from experiments.leduc_poker.consequence_weighted_distillation.common import (
    peak_rss_mb,
    repository_commit,
    source_path,
    validate_source_manifest,
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
    ARCHITECTURE_LABELS,
    ARCHITECTURE_ORDER,
    ARM_METADATA,
    ARM_ORDER,
    DEVELOPMENT_CHECKPOINT,
    EXPERIMENT_NAME,
    GAME_NAME,
    GROUPED_ARM_BY_ARCHITECTURE,
    GROUPED_TARGETS,
    ROW_TARGETS,
    VALIDATION_CHECKPOINT,
    fit_replicates,
    learning_rates,
    training_budgets,
    validation_samples,
)
from .distill import (
    build_catalog,
    build_model,
    ensemble_table,
    exact_teacher_dataset,
    fit_schedule,
    grouped_dataset,
    objective_equivalence,
    parameter_count,
    row_dataset,
)


def _source(
    *, seed: int, checkpoint_id: str, source_root: Path, smoke: bool
):
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
    return source, path, role, provenance, payload


def _decomposition(source, game) -> list[dict]:
    exact_value = exact_exploitability(game, source.exact_table)
    _, empirical_value = empirical_table(source, game)
    archived_value, _ = evaluate_model_state(
        source, game, source.source_model_state
    )
    return [
        {
            "source_seed": int(source.seed),
            "checkpoint_id": source.checkpoint_id,
            "policy": policy,
            "exploitability": value,
        }
        for policy, value in (
            ("exact_tabular", exact_value),
            ("empirical_reservoir", empirical_value),
            ("archived_neural", archived_value),
        )
    ]


def _metric_row(
    result: Mapping,
    *,
    source,
    arm_id: str,
    architecture_id: str,
    target_id: str,
    replicate: int,
    fit_seed: int,
    teacher_type: str,
) -> dict:
    exact_value = exact_exploitability(pyspiel.load_game(GAME_NAME), source.exact_table)
    return {
        "source_seed": int(source.seed),
        "checkpoint_id": source.checkpoint_id,
        "fit_replicate": int(replicate),
        "fit_seed": int(fit_seed),
        "arm_id": arm_id,
        "arm_label": (
            ARM_METADATA[arm_id]["arm_label"]
            if arm_id in ARM_METADATA
            else f"Exact teacher: {ARCHITECTURE_LABELS[architecture_id]}"
        ),
        "architecture_id": architecture_id,
        "target_id": target_id,
        "teacher_type": teacher_type,
        "exact_exploitability": exact_value,
        "distillation_gap": float(result["exploitability"]) - exact_value,
        **{
            key: value
            for key, value in result.items()
            if key
            not in {
                "model_state",
                "policy_table",
                "gradient_rows",
                "group_rows",
                "error_rows",
            }
        },
    }


def _diagnostic_rows(
    result: Mapping,
    *,
    source,
    arm_id: str,
    architecture_id: str,
    target_id: str,
    replicate: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    common = {
        "source_seed": int(source.seed),
        "checkpoint_id": source.checkpoint_id,
        "fit_replicate": int(replicate),
        "arm_id": arm_id,
        "architecture_id": architecture_id,
        "target_id": target_id,
        "learning_rate": float(result["learning_rate"]),
        "training_budget": int(result["training_budget"]),
    }
    gradient = [{**common, **row} for row in result["gradient_rows"]]
    groups = [{**common, **row} for row in result["group_rows"]]
    errors = [{**common, **row} for row in result["error_rows"]]
    return gradient, groups, errors


def _save_policy(
    worker_dir: Path,
    *,
    source,
    result: Mapping,
    arm_id: str,
    architecture_id: str,
    target_id: str,
    replicate: int,
    fit_seed: int,
) -> dict:
    path = worker_dir / "policies" / f"{arm_id}_fit_{replicate}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "source_seed": int(source.seed),
        "checkpoint_id": source.checkpoint_id,
        "arm_id": arm_id,
        "architecture_id": architecture_id,
        "target_id": target_id,
        "fit_replicate": int(replicate),
        "fit_seed": int(fit_seed),
        "input_size": int(source.infostates.shape[1]),
        "output_size": int(source.policies.shape[1]),
        "learning_rate": float(result["learning_rate"]),
        "training_budget": int(result["training_budget"]),
        "parameter_count": int(result["parameter_count"]),
        "model": {
            name: value.detach().cpu().clone()
            for name, value in result["model_state"].items()
        },
    }
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    verifier = build_model(
        architecture_id, loaded["input_size"], loaded["output_size"]
    )
    verifier.load_state_dict(loaded["model"])
    if parameter_count(verifier) != loaded["parameter_count"]:
        raise ValueError("Saved Experiment 34 policy parameter count differs")
    return {
        "source_seed": int(source.seed),
        "checkpoint_id": source.checkpoint_id,
        "arm_id": arm_id,
        "architecture_id": architecture_id,
        "target_id": target_id,
        "fit_replicate": int(replicate),
        "relative_path": str(path.relative_to(worker_dir)),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _fit_seed(seed: int, replicate: int) -> int:
    return int(seed) * 100 + int(replicate)


def run_development_worker(
    *, seed: int, source_root: Path, worker_dir: Path, smoke: bool
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game(GAME_NAME)
    source, path, role, provenance, payload = _source(
        seed=seed,
        checkpoint_id=DEVELOPMENT_CHECKPOINT,
        source_root=source_root,
        smoke=smoke,
    )
    catalog = build_catalog(game)
    datasets = {
        ROW_TARGETS: row_dataset(source, catalog),
        GROUPED_TARGETS: grouped_dataset(source, catalog),
    }
    uniform = datasets[ROW_TARGETS].masks / np.maximum(
        datasets[ROW_TARGETS].masks.sum(axis=1, keepdims=True), 1.0
    )
    row_objective, grouped_objective = objective_equivalence(
        source, catalog, uniform
    )
    if not np.isclose(row_objective, grouped_objective, rtol=1e-6, atol=1e-7):
        raise ValueError("Grouped targets do not preserve the row-wise CE objective")

    metrics, gradients, group_rows, errors = [], [], [], []
    for arm in ARM_ORDER:
        metadata = ARM_METADATA[arm]
        architecture_id = metadata["architecture_id"]
        target_id = metadata["target_id"]
        for learning_rate in learning_rates(smoke=smoke):
            for replicate in fit_replicates(smoke=smoke):
                fit_seed = _fit_seed(seed, replicate)
                snapshots = fit_schedule(
                    source,
                    game,
                    catalog,
                    architecture_id=architecture_id,
                    dataset=datasets[target_id],
                    fit_seed=fit_seed,
                    learning_rate=learning_rate,
                    training_budgets=training_budgets(smoke=smoke),
                    validation_samples=validation_samples(smoke=smoke),
                    raw_diagnostics=datasets[ROW_TARGETS],
                    grouped_diagnostics=datasets[GROUPED_TARGETS],
                )
                for snapshot in snapshots:
                    metrics.append(
                        _metric_row(
                            snapshot,
                            source=source,
                            arm_id=arm,
                            architecture_id=architecture_id,
                            target_id=target_id,
                            replicate=replicate,
                            fit_seed=fit_seed,
                            teacher_type="empirical_reservoir",
                        )
                    )
                    grad, groups, detail = _diagnostic_rows(
                        snapshot,
                        source=source,
                        arm_id=arm,
                        architecture_id=architecture_id,
                        target_id=target_id,
                        replicate=replicate,
                    )
                    gradients.extend(grad)
                    group_rows.extend(groups)
                    errors.extend(detail)

    decomposition = _decomposition(source, game)
    write_csv(worker_dir / "development_metrics.csv", metrics)
    write_csv(worker_dir / "gradient_cosines.csv", gradients)
    write_csv(worker_dir / "group_metrics.csv", group_rows)
    write_csv(worker_dir / "information_set_errors.csv", errors)
    write_csv(worker_dir / "source_decomposition.csv", decomposition)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "stage": "development",
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": repository_commit(),
        "source_state": {
            "checkpoint_id": DEVELOPMENT_CHECKPOINT,
            "source_checkpoint_id": role,
            **provenance,
            "size_bytes": int(path.stat().st_size),
        },
        "objective_equivalence": {
            "row_cross_entropy": row_objective,
            "grouped_cross_entropy": grouped_objective,
            "absolute_difference": abs(row_objective - grouped_objective),
        },
        "dataset": {
            "num_replay_rows": len(datasets[ROW_TARGETS].features),
            "num_unique_information_sets": len(datasets[GROUPED_TARGETS].features),
            "reduction_ratio": (
                len(datasets[GROUPED_TARGETS].features)
                / float(len(datasets[ROW_TARGETS].features))
            ),
        },
        "peak_rss_mb": peak_rss_mb(),
        "artifacts": {
            "metrics": "development_metrics.csv",
            "gradient_cosines": "gradient_cosines.csv",
            "group_metrics": "group_metrics.csv",
            "information_set_errors": "information_set_errors.csv",
            "source_decomposition": "source_decomposition.csv",
        },
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(worker_dir / "SUCCESS.json", {"status": "complete", "source_seed": seed})
    del payload, source
    gc.collect()
    return result


def run_validation_worker(
    *,
    seed: int,
    source_root: Path,
    selection_path: Path,
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    selection = read_json(selection_path)
    if selection.get("status") != "complete" or bool(selection.get("smoke")) != bool(smoke):
        raise ValueError("Experiment 34 selection is incomplete or has the wrong mode")
    selected = selection["selected_configs"]
    if set(selected) != set(ARM_ORDER):
        raise ValueError("Experiment 34 selection does not contain all six arms")

    game = pyspiel.load_game(GAME_NAME)
    source, path, role, provenance, payload = _source(
        seed=seed,
        checkpoint_id=VALIDATION_CHECKPOINT,
        source_root=source_root,
        smoke=smoke,
    )
    catalog = build_catalog(game)
    datasets = {
        ROW_TARGETS: row_dataset(source, catalog),
        GROUPED_TARGETS: grouped_dataset(source, catalog),
    }
    exact_dataset = exact_teacher_dataset(source, catalog)
    metrics, gradients, group_rows, errors, policies, ensembles = [], [], [], [], [], []
    tables_by_arm: dict[str, list[Mapping]] = {}

    for arm in ARM_ORDER:
        metadata = ARM_METADATA[arm]
        architecture_id = metadata["architecture_id"]
        target_id = metadata["target_id"]
        chosen = selected[arm]
        tables_by_arm[arm] = []
        for replicate in fit_replicates(smoke=smoke):
            fit_seed = _fit_seed(seed, replicate)
            snapshot = fit_schedule(
                source,
                game,
                catalog,
                architecture_id=architecture_id,
                dataset=datasets[target_id],
                fit_seed=fit_seed,
                learning_rate=float(chosen["learning_rate"]),
                training_budgets=(int(chosen["training_budget"]),),
                validation_samples=validation_samples(smoke=smoke),
                raw_diagnostics=datasets[ROW_TARGETS],
                grouped_diagnostics=datasets[GROUPED_TARGETS],
            )[0]
            metrics.append(
                _metric_row(
                    snapshot,
                    source=source,
                    arm_id=arm,
                    architecture_id=architecture_id,
                    target_id=target_id,
                    replicate=replicate,
                    fit_seed=fit_seed,
                    teacher_type="empirical_reservoir",
                )
            )
            grad, groups, detail = _diagnostic_rows(
                snapshot,
                source=source,
                arm_id=arm,
                architecture_id=architecture_id,
                target_id=target_id,
                replicate=replicate,
            )
            gradients.extend(grad)
            group_rows.extend(groups)
            errors.extend(detail)
            tables_by_arm[arm].append(snapshot["policy_table"])
            policies.append(
                _save_policy(
                    worker_dir,
                    source=source,
                    result=snapshot,
                    arm_id=arm,
                    architecture_id=architecture_id,
                    target_id=target_id,
                    replicate=replicate,
                    fit_seed=fit_seed,
                )
            )
        ensemble = ensemble_table(tables_by_arm[arm])
        ensembles.append(
            {
                "source_seed": int(seed),
                "checkpoint_id": VALIDATION_CHECKPOINT,
                "arm_id": arm,
                "arm_label": metadata["arm_label"],
                "architecture_id": architecture_id,
                "target_id": target_id,
                "num_members": len(tables_by_arm[arm]),
                "exploitability": exact_exploitability(game, ensemble),
            }
        )

    # Exact-table teachers are deliberately diagnostic and use the hyperparameters
    # selected for the corresponding grouped empirical arm.
    for architecture_id in ARCHITECTURE_ORDER:
        grouped_arm = GROUPED_ARM_BY_ARCHITECTURE[architecture_id]
        chosen = selected[grouped_arm]
        diagnostic_arm = f"exact_teacher__{architecture_id}"
        diagnostic_tables = []
        for replicate in fit_replicates(smoke=smoke):
            fit_seed = _fit_seed(seed, replicate)
            snapshot = fit_schedule(
                source,
                game,
                catalog,
                architecture_id=architecture_id,
                dataset=exact_dataset,
                fit_seed=fit_seed,
                learning_rate=float(chosen["learning_rate"]),
                training_budgets=(int(chosen["training_budget"]),),
                validation_samples=validation_samples(smoke=smoke),
                raw_diagnostics=datasets[ROW_TARGETS],
                grouped_diagnostics=datasets[GROUPED_TARGETS],
            )[0]
            metrics.append(
                _metric_row(
                    snapshot,
                    source=source,
                    arm_id=diagnostic_arm,
                    architecture_id=architecture_id,
                    target_id="exact_tabular_teacher",
                    replicate=replicate,
                    fit_seed=fit_seed,
                    teacher_type="exact_tabular_diagnostic",
                )
            )
            diagnostic_tables.append(snapshot["policy_table"])
            policies.append(
                _save_policy(
                    worker_dir,
                    source=source,
                    result=snapshot,
                    arm_id=diagnostic_arm,
                    architecture_id=architecture_id,
                    target_id="exact_tabular_teacher",
                    replicate=replicate,
                    fit_seed=fit_seed,
                )
            )
        ensembles.append(
            {
                "source_seed": int(seed),
                "checkpoint_id": VALIDATION_CHECKPOINT,
                "arm_id": diagnostic_arm,
                "arm_label": f"Exact teacher: {ARCHITECTURE_LABELS[architecture_id]}",
                "architecture_id": architecture_id,
                "target_id": "exact_tabular_teacher",
                "num_members": len(diagnostic_tables),
                "exploitability": exact_exploitability(
                    game, ensemble_table(diagnostic_tables)
                ),
            }
        )

    decomposition = _decomposition(source, game)
    write_csv(worker_dir / "validation_metrics.csv", metrics)
    write_csv(worker_dir / "gradient_cosines.csv", gradients)
    write_csv(worker_dir / "group_metrics.csv", group_rows)
    write_csv(worker_dir / "information_set_errors.csv", errors)
    write_csv(worker_dir / "ensemble_metrics.csv", ensembles)
    write_csv(worker_dir / "source_decomposition.csv", decomposition)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "stage": "validation",
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": repository_commit(),
        "selection_sha256": sha256(selection_path),
        "source_state": {
            "checkpoint_id": VALIDATION_CHECKPOINT,
            "source_checkpoint_id": role,
            **provenance,
            "size_bytes": int(path.stat().st_size),
        },
        "policies": policies,
        "peak_rss_mb": peak_rss_mb(),
        "artifacts": {
            "metrics": "validation_metrics.csv",
            "gradient_cosines": "gradient_cosines.csv",
            "group_metrics": "group_metrics.csv",
            "information_set_errors": "information_set_errors.csv",
            "ensemble_metrics": "ensemble_metrics.csv",
            "source_decomposition": "source_decomposition.csv",
        },
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(worker_dir / "SUCCESS.json", {"status": "complete", "source_seed": seed})
    del payload, source
    gc.collect()
    return result


__all__ = ["run_development_worker", "run_validation_worker"]
