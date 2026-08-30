"""Run exact tabular implementation validation of UCV-ESCHER."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str((Path("outputs") / ".matplotlib_cache").resolve())
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from unbiased_escher import UnbiasedControlVariateEscher  # noqa: E402
from vr_deep_cfr.logger import Logger  # noqa: E402

from .config import (  # noqa: E402
    BASELINE_FREE,
    BASE_CONFIG,
    BATCH_TIMEOUT_SECONDS,
    CONDITIONAL_BIAS_TOLERANCE,
    DEFAULT_SEEDS,
    EXPECTED_SEQUENTIAL_HOURS,
    EXPERIMENT_NAME,
    FULL_ADAPTIVE,
    MEASURED_HOURS_PER_SEED,
    TARGET_NODES,
    VARIANTS,
    VARIANT_ORDER,
    checkpoint_contract,
)
from .diagnostics import (  # noqa: E402
    ExactUCVOracle,
    frozen_state_fingerprint,
    policy_table_for_mode,
    predictability_audit,
)


LOGGER = logging.getLogger(EXPERIMENT_NAME)
MODULE = "experiments.leduc_poker.ucv_exact_tabular_validation.run"
SMOKE_SEED = 99991


def _parse_seeds(value: str | None) -> List[int]:
    if value is None:
        return list(DEFAULT_SEEDS)
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)):
        raise ValueError("Seeds must be non-empty and distinct")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _append_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("Diagnostic rows do not share a stable schema")
    exists = path.is_file() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _apply_overrides(args, config: Dict[str, Any]) -> None:
    overrides = {
        "num_traversals": args.traversals,
        "max_num_iterations": args.max_iterations,
        "advantage_network_train_steps": args.advantage_train_steps,
        "ave_policy_network_train_steps": args.policy_train_steps,
        "baseline_network_train_steps": args.q_train_steps,
        "calibration_train_steps": args.calibration_train_steps,
        "advantage_batch_size": args.batch_size,
        "ave_policy_batch_size": args.batch_size,
        "baseline_batch_size": args.batch_size,
        "calibration_batch_size": args.batch_size,
        "advantage_buffer_size": args.buffer_size,
        "ave_policy_buffer_size": args.buffer_size,
        "baseline_buffer_size": args.buffer_size,
        "calibration_buffer_size": args.buffer_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value


def _build_solver(seed: int, config: Mapping[str, Any]):
    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    solver = UnbiasedControlVariateEscher(**kwargs)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    return solver


def _cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _save_diagnostic_snapshot(
    solver,
    path: Path,
    *,
    seed: int,
    checkpoint_id: str,
    checkpoint_target_nodes: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    regret = []
    for trainer in solver.regret_trainers:
        item = {
            "model": _cpu_state_dict(trainer.model),
            "prediction_gate": float(getattr(trainer, "prediction_gate", 0.0)),
        }
        if hasattr(trainer, "imm_model"):
            item["immediate_model"] = _cpu_state_dict(trainer.imm_model)
        regret.append(item)
    payload = {
        "schema_version": 1,
        "type": "ucv_exact_diagnostic_snapshot",
        "experiment_name": EXPERIMENT_NAME,
        "repository_commit": _repository_commit(),
        "seed": int(seed),
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_target_nodes": int(checkpoint_target_nodes),
        "completed_iteration": int(solver.num_iteration),
        "nodes_touched": int(solver.nodes_touched),
        "config": dict(config),
        "regret_trainers": regret,
        "q_target_models": [
            _cpu_state_dict(member.target_model)
            for member in solver.q_value_trainer.members
        ],
        "q_target_versions": [
            int(member.target_version) for member in solver.q_value_trainer.members
        ],
        "calibration_target_model": _cpu_state_dict(
            solver.calibration_trainer.target_model
        ),
        "calibration_target_version": int(
            solver.calibration_trainer.target_version
        ),
        "next_prediction_gates": [
            float(solver.gate_controller.value(player))
            for player in range(solver.num_players)
        ],
        "variant_contract": VARIANTS,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "snapshot_path": str(path.name),
        "snapshot_sha256": _sha256(path),
        "snapshot_size_bytes": int(path.stat().st_size),
    }


def _weighted_average(rows, field: str) -> float:
    weights = np.asarray([float(row["sampling_reach_mass"]) for row in rows])
    values = np.asarray([float(row[field]) for row in rows])
    if float(np.sum(weights)) <= 0.0:
        return float(np.mean(values))
    return float(np.dot(weights, values) / np.sum(weights))


def _checkpoint_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    checkpoint_id: str,
    checkpoint_target_nodes: int,
    iteration: int,
    nodes_touched: int,
    diagnostic_seconds: float,
    frozen_state_unchanged: bool,
    snapshot_record: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    summaries = []
    for variant_id in VARIANT_ORDER:
        selected = [row for row in rows if row["variant_id"] == variant_id]
        max_q_bias = max(abs(float(row["action_value_bias"])) for row in selected)
        max_advantage_bias = max(
            abs(float(row["advantage_bias"])) for row in selected
        )
        summary = {
            "seed": int(seed),
            "checkpoint_id": checkpoint_id,
            "checkpoint_target_nodes": int(checkpoint_target_nodes),
            "completed_iteration": int(iteration),
            "nodes_touched": int(nodes_touched),
            "variant_id": variant_id,
            "variant_label": VARIANTS[variant_id]["label"],
            "policy_mode": VARIANTS[variant_id]["policy_mode"],
            "num_exact_rows": len(selected),
            "num_folds": len({int(row["fold"]) for row in selected}),
            "max_abs_action_value_bias": max_q_bias,
            "max_abs_advantage_bias": max_advantage_bias,
            "reach_weighted_action_value_variance": _weighted_average(
                selected, "action_value_variance"
            ),
            "reach_weighted_advantage_variance": _weighted_average(
                selected, "advantage_variance"
            ),
            "reach_weighted_advantage_mse": _weighted_average(
                selected, "advantage_mse"
            ),
            "minimum_sampling_probability": min(
                float(row["sampling_probability"]) for row in selected
            ),
            "conditional_unbiasedness_pass": (
                max(max_q_bias, max_advantage_bias)
                <= CONDITIONAL_BIAS_TOLERANCE
            ),
            "conditional_bias_tolerance": CONDITIONAL_BIAS_TOLERANCE,
            "frozen_state_unchanged": bool(frozen_state_unchanged),
            "diagnostic_seconds": float(diagnostic_seconds),
            "advantage_variance_ratio_vs_baseline_free": np.nan,
            "fraction_pairs_lower_variance_than_baseline_free": np.nan,
            **snapshot_record,
        }
        summaries.append(summary)

    baseline = {
        (
            int(row["fold"]),
            int(row["player"]),
            str(row["information_state"]),
            int(row["action"]),
        ): row
        for row in rows
        if row["variant_id"] == BASELINE_FREE
    }
    full_rows = [row for row in rows if row["variant_id"] == FULL_ADAPTIVE]
    paired = []
    for row in full_rows:
        key = (
            int(row["fold"]),
            int(row["player"]),
            str(row["information_state"]),
            int(row["action"]),
        )
        if key in baseline:
            paired.append((row, baseline[key]))
    full_summary = next(
        summary for summary in summaries if summary["variant_id"] == FULL_ADAPTIVE
    )
    full_total = sum(
        float(row["sampling_reach_mass"]) * float(row["advantage_variance"])
        for row, _ in paired
    )
    baseline_total = sum(
        float(reference["sampling_reach_mass"])
        * float(reference["advantage_variance"])
        for _, reference in paired
    )
    full_summary["advantage_variance_ratio_vs_baseline_free"] = (
        full_total / baseline_total if baseline_total > 0.0 else np.nan
    )
    full_summary["fraction_pairs_lower_variance_than_baseline_free"] = float(
        np.mean(
            [
                float(row["advantage_variance"])
                < float(reference["advantage_variance"])
                for row, reference in paired
            ]
        )
    )
    return summaries


def validate_checkpoint(
    solver,
    *,
    seed: int,
    checkpoint_id: str,
    checkpoint_target_nodes: int,
    config: Mapping[str, Any],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    start = time.perf_counter()
    before = frozen_state_fingerprint(solver)
    policy_tables = {
        mode: policy_table_for_mode(solver, mode)
        for mode in sorted({spec["policy_mode"] for spec in VARIANTS.values()})
    }
    rows = []
    for variant_id in VARIANT_ORDER:
        spec = VARIANTS[variant_id]
        for fold in range(int(solver.q_value_trainer.ensemble_size)):
            oracle = ExactUCVOracle(
                solver,
                policy_table=policy_tables[spec["policy_mode"]],
                variant_id=variant_id,
                fold=fold,
            )
            variant_rows = oracle.rows()
            for row in variant_rows:
                row.update(
                    {
                        "seed": int(seed),
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_target_nodes": int(checkpoint_target_nodes),
                        "completed_iteration": int(solver.num_iteration),
                        "nodes_touched": int(solver.nodes_touched),
                    }
                )
            rows.extend(variant_rows)
    after = frozen_state_fingerprint(solver)
    frozen_state_unchanged = before == after
    if not frozen_state_unchanged:
        changed = {
            key: {"before": before.get(key), "after": after.get(key)}
            for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)
        }
        raise RuntimeError(
            "Exact validation mutated a frozen model, control, fold or RNG: "
            f"{changed}"
        )
    snapshot_path = output_dir / "snapshots" / f"{checkpoint_id}.pt"
    snapshot_record = _save_diagnostic_snapshot(
        solver,
        snapshot_path,
        seed=seed,
        checkpoint_id=checkpoint_id,
        checkpoint_target_nodes=checkpoint_target_nodes,
        config=config,
    )
    snapshot_record["snapshot_path"] = str(snapshot_path.relative_to(output_dir))
    diagnostic_seconds = time.perf_counter() - start
    _append_csv(output_dir / "estimator_diagnostics.csv", rows)
    summaries = _checkpoint_summaries(
        rows,
        seed=seed,
        checkpoint_id=checkpoint_id,
        checkpoint_target_nodes=checkpoint_target_nodes,
        iteration=int(solver.num_iteration),
        nodes_touched=int(solver.nodes_touched),
        diagnostic_seconds=diagnostic_seconds,
        frozen_state_unchanged=frozen_state_unchanged,
        snapshot_record=snapshot_record,
    )
    _append_csv(output_dir / "checkpoint_summary.csv", summaries)
    if not all(bool(row["conditional_unbiasedness_pass"]) for row in summaries):
        LOGGER.error(
            "Conditional unbiasedness exceeded tolerance at %s seed %s",
            checkpoint_id,
            seed,
        )
    return summaries


def run_seed(
    *,
    seed: int,
    config: Mapping[str, Any],
    target_nodes: int,
    checkpoints,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    solver = _build_solver(seed, config)
    solver.target_nodes_touched = int(target_nodes)
    pending = list(checkpoints)
    summary_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()

    def capture(active_solver, raw_checkpoint):
        if str(raw_checkpoint.get("checkpoint_kind")) != "outer_iteration":
            return
        while pending and int(active_solver.nodes_touched) >= int(pending[0][1]):
            checkpoint_id, checkpoint_target = pending.pop(0)
            LOGGER.info(
                "Validating seed %s %s at iteration %s and %s nodes",
                seed,
                checkpoint_id,
                active_solver.num_iteration,
                active_solver.nodes_touched,
            )
            summary_rows.extend(
                validate_checkpoint(
                    active_solver,
                    seed=seed,
                    checkpoint_id=str(checkpoint_id),
                    checkpoint_target_nodes=int(checkpoint_target),
                    config=config,
                    output_dir=output_dir,
                )
            )

    try:
        curves = solver.solve(post_checkpoint_callback=capture)
        if pending:
            raise RuntimeError(f"Missing checkpoint diagnostics: {pending}")
        elapsed = time.perf_counter() - start
        with open(output_dir / "estimator_diagnostics.csv", encoding="utf-8") as handle:
            num_exact_rows = sum(1 for _ in handle) - 1
        result = {
            "status": "complete",
            "seed": int(seed),
            "target_nodes": int(target_nodes),
            "final_nodes_touched": int(solver.nodes_touched),
            "final_iteration": int(solver.num_iteration),
            "elapsed_seconds": float(elapsed),
            "checkpoint_ids": [str(name) for name, _ in checkpoints],
            "num_checkpoint_summary_rows": len(summary_rows),
            "num_exact_rows": num_exact_rows,
            "all_conditional_unbiasedness_checks_pass": all(
                bool(row["conditional_unbiasedness_pass"]) for row in summary_rows
            ),
            "curves": curves,
        }
        shared._write_json(output_dir / "seed_result.json", result)
        shared._write_json(
            output_dir / "SUCCESS.json",
            {
                "status": "complete",
                "seed": int(seed),
                "all_conditional_unbiasedness_checks_pass": result[
                    "all_conditional_unbiasedness_checks_pass"
                ],
            },
        )
        return result
    finally:
        del solver
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_sequential(seeds: Sequence[int], run_one: Callable[[int], Any]) -> List[Any]:
    """Execute exactly one seed at a time in the declared order."""

    results = []
    for seed in seeds:
        results.append(run_one(int(seed)))
    return results


def _run_worker(input_path: Path, output_dir: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    run_seed(
        seed=int(payload["seed"]),
        config=payload["config"],
        target_nodes=int(payload["target_nodes"]),
        checkpoints=[
            (str(item["checkpoint_id"]), int(item["target_nodes"]))
            for item in payload["checkpoints"]
        ],
        output_dir=output_dir,
    )
    return 0


def _run_seed_subprocess(
    *,
    run_dir: Path,
    seed: int,
    config: Mapping[str, Any],
    target_nodes: int,
    checkpoints,
) -> Dict[str, Any]:
    input_path = run_dir / "worker_inputs" / f"seed_{seed}.json"
    output_dir = run_dir / "worker_results" / f"seed_{seed}"
    log_path = run_dir / "worker_logs" / f"seed_{seed}.log"
    payload = {
        "seed": int(seed),
        "config": dict(config),
        "target_nodes": int(target_nodes),
        "checkpoints": [
            {"checkpoint_id": str(name), "target_nodes": int(nodes)}
            for name, nodes in checkpoints
        ],
    }
    shared._write_json(input_path, payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--worker-input-json",
        str(input_path),
        "--worker-output-dir",
        str(output_dir),
    ]
    LOGGER.info("Starting sequential seed %s", seed)
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[3],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed; see {log_path}")
    with open(output_dir / "seed_result.json", encoding="utf-8") as handle:
        return json.load(handle)


def _combine_csv(run_dir: Path, filename: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted((run_dir / "worker_results").glob(f"seed_*/{filename}")):
        rows.extend(_read_csv(path))
    shared._write_csv(run_dir / filename, rows)
    return rows


def _plot_variance(run_dir: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    checkpoints = [name for name, _ in checkpoint_contract()]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(checkpoints))
    for variant_id in VARIANT_ORDER:
        means = []
        for checkpoint in checkpoints:
            values = [
                float(row["reach_weighted_advantage_variance"])
                for row in summaries
                if row["variant_id"] == variant_id
                and row["checkpoint_id"] == checkpoint
            ]
            means.append(float(np.mean(values)) if values else np.nan)
        ax.plot(x, means, marker="o", linewidth=2, label=VARIANTS[variant_id]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints)
    ax.set_xlabel("Frozen training checkpoint")
    ax.set_ylabel("Reach-weighted exact advantage-estimator variance")
    set_chart_title(ax, "Exact UCV estimator variance across training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "exact_estimator_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_bias(run_dir: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    labels = []
    values = []
    for variant_id in VARIANT_ORDER:
        selected = [
            max(
                float(row["max_abs_action_value_bias"]),
                float(row["max_abs_advantage_bias"]),
            )
            for row in summaries
            if row["variant_id"] == variant_id
        ]
        labels.append(VARIANTS[variant_id]["label"])
        values.append(max(selected) if selected else np.nan)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(np.arange(len(labels)), values)
    ax.axhline(
        CONDITIONAL_BIAS_TOLERANCE,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Declared numerical tolerance",
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Maximum absolute conditional bias")
    set_chart_title(ax, "Exact UCV conditional-unbiasedness check")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "maximum_conditional_bias.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _finalize(
    run_dir: Path,
    *,
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    summaries = _combine_csv(run_dir, "checkpoint_summary.csv")
    diagnostics = _combine_csv(run_dir, "estimator_diagnostics.csv")
    seed_rows = [
        {
            key: value
            for key, value in result.items()
            if key != "curves"
        }
        for result in results
    ]
    curves = [
        {"seed": int(result["seed"]), **row}
        for result in results
        for row in result["curves"]
    ]
    shared._write_csv(run_dir / "seed_summary.csv", seed_rows)
    shared._write_csv(run_dir / "training_checkpoint_curves.csv", curves)
    all_pass = all(
        str(row["conditional_unbiasedness_pass"]).lower() == "true"
        and str(row["frozen_state_unchanged"]).lower() == "true"
        for row in summaries
    )
    audit = predictability_audit()
    aggregate = {
        "status": "complete",
        "num_seeds": len(results),
        "num_frozen_checkpoints": len(results) * len(checkpoint_contract()),
        "num_exact_information_set_action_fold_rows": len(diagnostics),
        "all_conditional_unbiasedness_checks_pass": all_pass,
        "predictability_audit_status": audit["status"],
        "maximum_absolute_action_value_bias": max(
            float(row["max_abs_action_value_bias"]) for row in summaries
        ),
        "maximum_absolute_advantage_bias": max(
            float(row["max_abs_advantage_bias"]) for row in summaries
        ),
        "full_ucv_variance_ratio_vs_baseline_free_by_seed_checkpoint": [
            {
                "seed": int(row["seed"]),
                "checkpoint_id": row["checkpoint_id"],
                "ratio": float(row["advantage_variance_ratio_vs_baseline_free"]),
                "fraction_pairs_lower": float(
                    row["fraction_pairs_lower_variance_than_baseline_free"]
                ),
            }
            for row in summaries
            if row["variant_id"] == FULL_ADAPTIVE
        ],
    }
    shared._write_json(run_dir / "experiment_metadata.json", metadata)
    shared._write_json(run_dir / "predictability_audit.json", audit)
    shared._write_json(run_dir / "aggregate_summary.json", aggregate)
    shared._write_json(
        run_dir / "summary.json",
        {"aggregate": aggregate, "seed_summary": seed_rows},
    )
    _plot_variance(run_dir, summaries)
    _plot_bias(run_dir, summaries)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", default="outputs/ucv_exact_tabular_validation"
    )
    parser.add_argument("--seeds")
    parser.add_argument("--target-nodes", type=int, default=TARGET_NODES)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--calibration-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_dir:
        if not args.worker_input_json or not args.worker_output_dir:
            raise ValueError("Both internal worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_dir)

    config = deepcopy(BASE_CONFIG)
    if args.smoke:
        args.seeds = str(SMOKE_SEED)
        args.target_nodes = 300
        args.traversals = 2
        args.max_iterations = 20
        args.advantage_train_steps = 1
        args.policy_train_steps = 1
        args.q_train_steps = 1
        args.calibration_train_steps = 1
        args.batch_size = 2
        args.buffer_size = 128
    _apply_overrides(args, config)
    seeds = _parse_seeds(args.seeds)
    target_nodes = int(args.target_nodes)
    checkpoints = checkpoint_contract(target_nodes)
    if int(config["max_num_iterations"]) <= 0:
        raise ValueError("max_num_iterations must be positive")
    audit = predictability_audit()
    if audit["status"] != "pass":
        raise RuntimeError("Predictability source-order audit failed before training")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"{EXPERIMENT_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "repository_commit": _repository_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "execution_order": seeds,
        "execution_mode": "single_experiment_three_seeds_strictly_sequential",
        "target_nodes": target_nodes,
        "checkpoints": [
            {"checkpoint_id": name, "target_nodes": nodes}
            for name, nodes in checkpoints
        ],
        "variants": VARIANTS,
        "conditional_bias_tolerance": CONDITIONAL_BIAS_TOLERANCE,
        "config": config,
        "measured_hours_per_seed": MEASURED_HOURS_PER_SEED,
        "expected_sequential_hours": EXPECTED_SEQUENTIAL_HOURS,
        "batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "smoke": bool(args.smoke),
        "protocol": {
            "checkpoint_rule": (
                "First completed outer iteration crossing each frozen node target."
            ),
            "counterfactual_rule": (
                "All estimator arms reuse the same frozen learned state; no arm is retrained."
            ),
            "fold_rule": (
                "Every cross-fitting fold is conditioned on and enumerated separately."
            ),
            "evidence_scope": (
                "Validates implementation conditional moments; it does not prove the theorem."
            ),
        },
    }
    shared._write_json(run_dir / "experiment_metadata.json", metadata)
    shared._write_json(run_dir / "predictability_audit.json", audit)

    failures = []

    def run_one(seed: int):
        try:
            return _run_seed_subprocess(
                run_dir=run_dir,
                seed=seed,
                config=config,
                target_nodes=target_nodes,
                checkpoints=checkpoints,
            )
        except Exception as exc:
            failures.append(
                {
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            shared._write_json(run_dir / "failed_runs.json", failures)
            raise

    results = run_sequential(seeds, run_one)
    _finalize(run_dir, results=results, metadata=metadata)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())


__all__ = [
    "main",
    "run_seed",
    "run_sequential",
    "validate_checkpoint",
]
