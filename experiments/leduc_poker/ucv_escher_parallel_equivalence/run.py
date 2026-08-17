"""Run Experiment 18: sequential versus parallel Experiment 7 UCV-ESCHER."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path("outputs") / ".matplotlib_cache").resolve()),
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run import (  # noqa: E402
    DIAGNOSTIC_FIELDS,
)
from unbiased_escher.parallel_utils import equivalence_summary  # noqa: E402

from .config import (  # noqa: E402
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    EXPECTED_FULL_EXPERIMENT_HOURS,
    EXPECTED_PARALLEL_HOURS_PER_SEED,
    EXPERIMENT_ID,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    MEASURED_SEQUENTIAL_HOURS_PER_SEED,
    PARALLEL_NUM_WORKERS,
    PARALLEL_RAY_OBJECT_STORE_MEMORY,
    PARALLEL_VARIANT_ID,
    RECOMMENDED_BATCH_TIMEOUT_MINUTES,
    SEQUENTIAL_VARIANT_ID,
    TARGET_NODES,
    VARIANTS,
)


LOGGER = logging.getLogger("ucv_escher_parallel_equivalence")
VARIANT_BY_ID = {str(variant["variant_id"]): variant for variant in VARIANTS}

PARALLEL_DIAGNOSTIC_FIELDS = (
    "cumulative_experience_collection_seconds",
    "cumulative_parallel_collection_seconds",
    "cumulative_worker_collection_seconds",
    "cumulative_parallel_sync_seconds",
    "cumulative_parallel_merge_seconds",
    "cumulative_parallel_learner_seconds",
    "parallel_independent_learner_threads",
    "parallel_peak_worker_result_mib",
)


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one seed is required")
    return parsed


def _parse_variant_ids(value: str | None) -> List[str]:
    if value is None:
        return [str(variant["variant_id"]) for variant in VARIANTS]
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(parsed) - set(VARIANT_BY_ID))
    if unknown:
        raise ValueError(f"Unknown variant ids: {', '.join(unknown)}")
    if not parsed:
        raise ValueError("At least one variant is required")
    return parsed


def _parse_float(value) -> float:
    return np.nan if value in {None, ""} else float(value)


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
        "evaluation_frequency": args.evaluation_frequency,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (
            int(args.early_evaluation_nodes),
        )


def _solver_kwargs(config: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    from vr_deep_cfr.logger import Logger

    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {
        key: value
        for key, value in config.items()
        if key not in control_fields
    }
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    return kwargs


def _build_solver(
    variant: Mapping[str, Any],
    seed: int,
    config: Mapping[str, Any],
    parallel_num_workers: int,
    parallel_ray_object_store_memory: int,
    parallel_learner_threads: int | None,
):
    kwargs = _solver_kwargs(config, seed)
    if variant["execution_backend"] == "sequential":
        from unbiased_escher import UnbiasedControlVariateEscher

        return UnbiasedControlVariateEscher(**kwargs)
    from unbiased_escher.parallel_solver import (
        ParallelUnbiasedControlVariateEscher,
    )

    return ParallelUnbiasedControlVariateEscher(
        **kwargs,
        parallel_num_workers=int(parallel_num_workers),
        parallel_run_seed=int(seed),
        parallel_ray_object_store_memory=int(parallel_ray_object_store_memory),
        parallel_learner_threads=parallel_learner_threads,
    )


def _run_variant(
    variant_id: str,
    seed: int,
    config: Mapping[str, Any],
    target_nodes: int,
    parallel_num_workers: int,
    parallel_ray_object_store_memory: int,
    parallel_learner_threads: int | None,
) -> Dict[str, Any]:
    import torch

    variant = VARIANT_BY_ID[str(variant_id)]
    end_to_end_start = time.perf_counter()
    init_start = time.perf_counter()
    solver = _build_solver(
        variant,
        seed,
        config,
        parallel_num_workers,
        parallel_ray_object_store_memory,
        parallel_learner_threads,
    )
    initialization_seconds = time.perf_counter() - init_start
    try:
        solver.target_nodes_touched = int(target_nodes)
        solver.max_num_iterations = int(config["max_num_iterations"])
        solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
        solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
        solver.early_evaluation_node_thresholds = tuple(
            int(value) for value in config["early_evaluation_node_thresholds"]
        )
        training_start = time.perf_counter()
        raw_checkpoints = solver.solve()
        training_seconds = time.perf_counter() - training_start

        curves = []
        for checkpoint_index, raw in enumerate(raw_checkpoints):
            value = float(raw["average_policy_value"])
            row = {
                "variant_id": str(variant_id),
                "variant_label": str(variant["variant_label"]),
                "execution_backend": str(variant["execution_backend"]),
                "parallel_num_workers": (
                    int(parallel_num_workers)
                    if variant["execution_backend"] == "ray_parallel"
                    else 1
                ),
                "seed": int(seed),
                "checkpoint_index": int(checkpoint_index),
                "iteration": int(raw["iteration"]),
                "episode": int(raw["episode"]),
                "nodes_touched": float(raw["nodes_touched"]),
                "wall_clock_seconds": float(raw["wall_clock_seconds"]),
                "exploitability": float(raw["exp"]),
                "average_policy_value": value,
                "policy_value_error": abs(value - LEDUC_GAME_VALUE_PLAYER_0),
                "average_policy_loss": _parse_float(raw.get("average_policy_loss")),
                "regret_loss_player_0": _parse_float(raw.get("regret_loss_0")),
                "regret_loss_player_1": _parse_float(raw.get("regret_loss_1")),
                "baseline_loss_player_0": _parse_float(raw.get("baseline_loss_0")),
                "baseline_loss_player_1": _parse_float(raw.get("baseline_loss_1")),
                "checkpoint_kind": str(raw.get("checkpoint_kind", "outer_iteration")),
                "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
                "is_initial_policy_evaluation": (
                    raw.get("checkpoint_kind") == "initial_untrained_policy"
                ),
            }
            for field in (*DIAGNOSTIC_FIELDS, *PARALLEL_DIAGNOSTIC_FIELDS):
                row[field] = _parse_float(raw.get(field))
            curves.append(row)

        if not curves:
            raise RuntimeError("Solver returned no checkpoints")
        final = curves[-1]
        if final["nodes_touched"] < int(target_nodes):
            raise RuntimeError(
                f"{variant_id} seed {seed} stopped before the node target"
            )
        training_curves = [
            row for row in curves if not row["is_initial_policy_evaluation"]
        ]
        exploitabilities = [row["exploitability"] for row in training_curves]
        summary = {
            "variant_id": str(variant_id),
            "variant_label": str(variant["variant_label"]),
            "execution_backend": str(variant["execution_backend"]),
            "parallel_num_workers": final["parallel_num_workers"],
            "seed": int(seed),
            "final_exploitability": float(final["exploitability"]),
            "best_exploitability": float(np.min(exploitabilities)),
            "final_window_mean_exploitability": float(
                np.mean(
                    exploitabilities[
                        -min(DEFAULT_FINAL_WINDOW, len(exploitabilities)) :
                    ]
                )
            ),
            "final_policy_value": float(final["average_policy_value"]),
            "final_policy_value_error": float(final["policy_value_error"]),
            "final_nodes_touched": float(final["nodes_touched"]),
            "num_iterations_completed": int(final["iteration"]),
            "solver_initialization_seconds": float(initialization_seconds),
            "training_seconds": float(training_seconds),
            "end_to_end_seconds": float(time.perf_counter() - end_to_end_start),
            "target_nodes_touched": int(target_nodes),
            "node_budget_delta": float(final["nodes_touched"] - target_nodes),
            "final_average_policy_buffer_size": len(solver.ave_policy_trainer.buffer),
            "final_advantage_buffer_size_player_0": len(
                solver.regret_trainers[0].buffer
            ),
            "final_advantage_buffer_size_player_1": len(
                solver.regret_trainers[1].buffer
            ),
            "final_history_value_buffer_size": int(
                sum(solver.q_value_trainer.fold_sizes())
            ),
        }
        for field in PARALLEL_DIAGNOSTIC_FIELDS:
            summary[f"final_{field}"] = float(final[field])
        parallel_span = summary.get(
            "final_cumulative_parallel_collection_seconds",
            np.nan,
        )
        worker_work = summary.get(
            "final_cumulative_worker_collection_seconds",
            np.nan,
        )
        if np.isfinite(parallel_span) and parallel_span > 0.0:
            summary["parallel_collection_work_over_span"] = (
                worker_work / parallel_span
            )
            summary["parallel_collection_worker_efficiency"] = (
                worker_work
                / (float(final["parallel_num_workers"]) * parallel_span)
            )
        else:
            summary["parallel_collection_work_over_span"] = np.nan
            summary["parallel_collection_worker_efficiency"] = np.nan
        return {"summary": summary, "curves": curves}
    finally:
        close = getattr(solver, "close", None)
        if close is not None:
            close()
        del solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_variant(
        str(payload["variant_id"]),
        int(payload["seed"]),
        payload["config"],
        int(payload["target_nodes"]),
        int(payload["parallel_num_workers"]),
        int(payload["parallel_ray_object_store_memory"]),
        payload.get("parallel_learner_threads"),
    )
    shared._write_json(output_path, result)
    return 0


def _run_subprocess(
    run_dir: Path,
    variant_id: str,
    seed: int,
    config: Mapping[str, Any],
    args,
) -> Dict[str, Any]:
    stem = f"{variant_id}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    shared._write_json(
        input_path,
        {
            "variant_id": variant_id,
            "seed": seed,
            "config": dict(config),
            "target_nodes": int(args.target_nodes),
            "parallel_num_workers": int(args.parallel_num_workers),
            "parallel_ray_object_store_memory": int(
                args.parallel_ray_object_store_memory
            ),
            "parallel_learner_threads": args.parallel_learner_threads,
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.ucv_escher_parallel_equivalence.run",
        "--worker-input-json",
        str(input_path),
        "--worker-output-json",
        str(output_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"{stem} failed; see {log_path}\n{tail}")
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)


def _paired_rows(summary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    indexed = {
        (str(row["variant_id"]), int(row["seed"])): row
        for row in summary_rows
    }
    rows = []
    for seed in sorted({int(row["seed"]) for row in summary_rows}):
        sequential = indexed.get((SEQUENTIAL_VARIANT_ID, seed))
        parallel = indexed.get((PARALLEL_VARIANT_ID, seed))
        if sequential is None or parallel is None:
            continue
        row = {
            "seed": seed,
            "exploitability_delta_parallel_minus_sequential": (
                parallel["final_exploitability"]
                - sequential["final_exploitability"]
            ),
            "policy_value_delta_parallel_minus_sequential": (
                parallel["final_policy_value"] - sequential["final_policy_value"]
            ),
            "nodes_delta_parallel_minus_sequential": (
                parallel["final_nodes_touched"]
                - sequential["final_nodes_touched"]
            ),
        }
        for field in (
            "solver_initialization_seconds",
            "training_seconds",
            "end_to_end_seconds",
            "final_cumulative_experience_collection_seconds",
        ):
            row[f"{field}_delta_parallel_minus_sequential"] = (
                parallel[field] - sequential[field]
            )
            row[f"{field}_speedup_sequential_over_parallel"] = (
                sequential[field] / parallel[field]
                if float(parallel[field]) > 0.0
                else np.nan
            )
        rows.append(row)
    return rows


def _aggregate(summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for variant_id in VARIANT_BY_ID:
        selected = [row for row in summary_rows if row["variant_id"] == variant_id]
        if not selected:
            continue
        numeric_fields = {
            key
            for row in selected
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregate[variant_id] = {
            field: shared._stats(float(row.get(field, np.nan)) for row in selected)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return aggregate


def _equivalence(paired_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "method": "paired 90% confidence interval / two one-sided tests",
        "exploitability": equivalence_summary(
            (
                row["exploitability_delta_parallel_minus_sequential"]
                for row in paired_rows
            ),
            FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
        ),
        "policy_value": equivalence_summary(
            (
                row["policy_value_delta_parallel_minus_sequential"]
                for row in paired_rows
            ),
            FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
        ),
    }


def _plot_curves(
    run_dir: Path,
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
) -> None:
    colors = {
        SEQUENTIAL_VARIANT_ID: "#9467bd",
        PARALLEL_VARIANT_ID: "#1f77b4",
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for variant_id, variant in VARIANT_BY_ID.items():
        rows = [row for row in curve_rows if row["variant_id"] == variant_id]
        if not rows:
            continue
        divisor = 3600.0 if x_key == "wall_clock_seconds" else 1.0
        for seed in sorted({int(row["seed"]) for row in rows}):
            seed_rows = sorted(
                [row for row in rows if int(row["seed"]) == seed],
                key=lambda row: row[x_key],
            )
            ax.plot(
                [row[x_key] / divisor for row in seed_rows],
                [row["exploitability"] for row in seed_rows],
                color=colors[variant_id],
                alpha=0.16,
                linewidth=1,
            )
        checkpoints = sorted({int(row["checkpoint_index"]) for row in rows})
        x_values, means, ses = [], [], []
        for checkpoint in checkpoints:
            at_checkpoint = [
                row for row in rows if int(row["checkpoint_index"]) == checkpoint
            ]
            x = np.asarray([row[x_key] / divisor for row in at_checkpoint])
            y = np.asarray([row["exploitability"] for row in at_checkpoint])
            x_values.append(float(np.mean(x)))
            stats = shared._stats(y)
            means.append(stats["mean"])
            ses.append(stats["se"])
        x = np.asarray(x_values)
        mean = np.asarray(means)
        se = np.asarray(ses)
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            color=colors[variant_id],
            label=variant["variant_label"],
        )
        ax.fill_between(x, mean - se, mean + se, color=colors[variant_id], alpha=0.14)
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel(
        "Wall-clock training time (hours)"
        if x_key == "wall_clock_seconds"
        else "Nodes touched"
    )
    ax.set_ylabel("Exploitability (NashConv / 2)")
    title_suffix = "wall-clock time" if x_key == "wall_clock_seconds" else "nodes"
    set_chart_title(
        ax,
        f"Experiment 18 UCV-ESCHER parallel equivalence by {title_suffix}",
    )
    ax.legend()
    fig.tight_layout()
    filename = (
        "exploitability_by_wall_clock.png"
        if x_key == "wall_clock_seconds"
        else "exploitability_by_nodes.png"
    )
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_quality(
    run_dir: Path,
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    labels, means, ses = [], [], []
    for variant_id, variant in VARIANT_BY_ID.items():
        values = [
            float(row["final_exploitability"])
            for row in summary_rows
            if row["variant_id"] == variant_id
        ]
        if not values:
            continue
        stats = shared._stats(values)
        labels.append(variant["variant_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 18 UCV-ESCHER final exploitability")
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_runtime(run_dir: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    labels, means, ses = [], [], []
    for variant_id, variant in VARIANT_BY_ID.items():
        values = [
            float(row["end_to_end_seconds"]) / 3600.0
            for row in summary_rows
            if row["variant_id"] == variant_id
        ]
        if not values:
            continue
        stats = shared._stats(values)
        labels.append(variant["variant_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_ylabel("End-to-end runtime (hours)")
    set_chart_title(ax, "Experiment 18 UCV-ESCHER runtime")
    fig.tight_layout()
    fig.savefig(run_dir / "end_to_end_runtime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _finalize(run_dir: Path, results, failures, metadata) -> None:
    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    paired_rows = _paired_rows(summary_rows)
    shared._write_csv(run_dir / "seed_variant_summary.csv", summary_rows)
    shared._write_csv(run_dir / "checkpoint_curves.csv", curve_rows)
    shared._write_csv(run_dir / "paired_differences_and_speedups.csv", paired_rows)
    shared._write_json(run_dir / "aggregate_summary.json", _aggregate(summary_rows))
    shared._write_json(run_dir / "paired_equivalence_summary.json", _equivalence(paired_rows))
    shared._write_json(run_dir / "experiment_metadata.json", metadata)
    shared._write_json(run_dir / "failed_runs.json", failures)
    if curve_rows:
        _plot_curves(run_dir, curve_rows, x_key="nodes_touched")
        _plot_curves(run_dir, curve_rows, x_key="wall_clock_seconds")
        _plot_final_quality(run_dir, summary_rows)
        _plot_runtime(run_dir, summary_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/ucv_escher_parallel_equivalence",
    )
    parser.add_argument("--seeds")
    parser.add_argument("--variant-ids")
    parser.add_argument("--target-nodes", type=int, default=TARGET_NODES)
    parser.add_argument("--parallel-num-workers", type=int, default=PARALLEL_NUM_WORKERS)
    parser.add_argument(
        "--parallel-ray-object-store-memory",
        type=int,
        default=PARALLEL_RAY_OBJECT_STORE_MEMORY,
    )
    parser.add_argument("--parallel-learner-threads", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--calibration-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--evaluation-frequency", type=int)
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_json)
    if args.target_nodes <= 0:
        raise ValueError("target-nodes must be positive")
    if args.parallel_num_workers < 2:
        raise ValueError("parallel-num-workers must be at least 2")

    seeds = _parse_csv_ints(args.seeds, DEFAULT_SEEDS)
    variant_ids = _parse_variant_ids(args.variant_ids)
    config = deepcopy(DEFAULT_CONFIG)
    _apply_overrides(args, config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"ucv_parallel_equivalence_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "seeds": seeds,
        "variant_ids": variant_ids,
        "variants": VARIANTS,
        "target_nodes": int(args.target_nodes),
        "config": config,
        "parallel_num_workers": int(args.parallel_num_workers),
        "parallel_ray_object_store_memory": int(args.parallel_ray_object_store_memory),
        "parallel_learner_threads": args.parallel_learner_threads,
        "final_exploitability_equivalence_margin": (
            FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN
        ),
        "final_policy_value_equivalence_margin": (
            FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN
        ),
        "measured_sequential_hours_per_seed": MEASURED_SEQUENTIAL_HOURS_PER_SEED,
        "expected_parallel_hours_per_seed": EXPECTED_PARALLEL_HOURS_PER_SEED,
        "expected_full_experiment_hours": EXPECTED_FULL_EXPERIMENT_HOURS,
        "recommended_batch_timeout_minutes": RECOMMENDED_BATCH_TIMEOUT_MINUTES,
        "batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "learner": (
                "Both arms use the exact Experiment 7 UCV-ESCHER architecture, "
                "update order, node target, replay capacities and train-step budgets."
            ),
            "parallelism": (
                "The parallel arm partitions each traverser's trajectories over "
                "three synchronous Ray actors; independent Q-fold and calibration "
                "updates run concurrently under a bounded CPU-thread budget."
            ),
            "equivalence": (
                "Primary inference uses paired parallel-minus-sequential final "
                "metric deltas and pre-declared 90% CI/TOST margins."
            ),
        },
    }

    results, failures = [], []
    for variant_id in variant_ids:
        for seed in seeds:
            try:
                LOGGER.info("Running %s seed %s", variant_id, seed)
                result = _run_subprocess(run_dir, variant_id, seed, config, args)
                results.append(result)
                shared._write_json(run_dir / "partial_results.json", results)
            except Exception as exc:  # pragma: no cover - operational path
                failures.append(
                    {
                        "variant_id": variant_id,
                        "seed": seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                shared._write_json(run_dir / "failed_runs.json", failures)
                LOGGER.error("%s seed %s failed: %s", variant_id, seed, exc)
                if not args.continue_on_error:
                    return 2
    _finalize(run_dir, results, failures, metadata)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
