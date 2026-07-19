"""Run Experiment 8: paired lean-architecture ablations of Experiment 6."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path("outputs") / ".matplotlib_cache").resolve()),
)
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes import (  # noqa: E402
    run as experiment_6,
)

from .config import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_SEEDS,
    DEFAULT_VARIANT_IDS,
    EXPECTED_PARALLEL_BY_VARIANT_RUNTIME_HOURS,
    EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_ID,
    FULL_EXPERIMENT_6,
    MEASURED_FULL_EXPERIMENT_6_HOURS_PER_SEED,
    PARALLEL_BY_VARIANT_BATCH_TIMEOUT_SECONDS,
    PER_WORKER_BATCH_TIMEOUT_SECONDS,
    SEQUENTIAL_BATCH_TIMEOUT_SECONDS,
    VARIANTS,
)


LOGGER = logging.getLogger("unbiased_control_variate_escher_lean_ablation")
RESULT_SOURCE = "experiment_8_new_run"
COLORS = {
    variant_id: plt.get_cmap("tab10")(index)
    for index, variant_id in enumerate(DEFAULT_VARIANT_IDS)
}


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    selected = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not selected:
        raise ValueError("At least one seed is required")
    return selected


def _parse_variants(value: str | None) -> List[str]:
    if value is None:
        return list(DEFAULT_VARIANT_IDS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variant ids: {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one variant id is required")
    return selected


def _variant_config(variant_id: str, base_config: Mapping[str, Any]):
    config = deepcopy(dict(base_config))
    config.update(VARIANTS[variant_id]["overrides"])
    return config


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
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (
            int(args.early_evaluation_nodes),
        )


def _assert_mechanism_invariants(
    variant_id: str,
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    rows = [
        row
        for row in result["curves"]
        if float(row.get("unbiased_estimator_sample_count", 0.0)) > 0.0
    ]
    fixed_beta = config.get("fixed_control_variate_beta")
    if fixed_beta is not None:
        for row in rows:
            if not np.isclose(row["control_variate_beta_min"], fixed_beta):
                raise RuntimeError(f"{variant_id} did not use its fixed beta")
            if not np.isclose(row["control_variate_beta_max"], fixed_beta):
                raise RuntimeError(f"{variant_id} did not use its fixed beta")
    if config.get("force_prediction_gate_zero") or not config.get(
        "use_instantaneous_predictor", True
    ):
        for row in rows:
            if row["prediction_gate_player_0"] != 0.0:
                raise RuntimeError(f"{variant_id} used the player-0 predictor")
            if row["prediction_gate_player_1"] != 0.0:
                raise RuntimeError(f"{variant_id} used the player-1 predictor")
    if not config.get("use_residual_calibration", True):
        for row in rows:
            if row["calibration_target_version"] != 0.0:
                raise RuntimeError(f"{variant_id} unexpectedly trained calibration")
    if config.get("sampling_uniform_floor_mass") == 1.0:
        minimum_probability = min(
            row["full_support_sampling_min_probability"] for row in rows
        )
        if minimum_probability < (1.0 / 3.0) - 1e-12:
            raise RuntimeError(f"{variant_id} did not retain uniform full support")


def _run_variant(
    variant_id: str,
    seed: int,
    config: Dict[str, Any],
    target_nodes: int,
) -> Dict[str, Any]:
    result = experiment_6._run_candidate(seed, config, target_nodes)
    spec = VARIANTS[variant_id]
    final_nodes = float(result["summary"]["final_nodes_touched"])
    if final_nodes < target_nodes:
        raise RuntimeError(
            f"{variant_id} seed {seed} stopped at {final_nodes:.0f} nodes "
            f"before target {target_nodes}"
        )
    architecture = {
        "q_ensemble_size": int(config["q_ensemble_size"]),
        "fixed_control_variate_beta": config.get("fixed_control_variate_beta"),
        "force_prediction_gate_zero": bool(
            config.get("force_prediction_gate_zero", False)
        ),
        "use_instantaneous_predictor": bool(
            config.get("use_instantaneous_predictor", True)
        ),
        "use_residual_calibration": bool(
            config.get("use_residual_calibration", True)
        ),
        "sampling_uniform_floor_mass": float(
            config["sampling_uniform_floor_mass"]
        ),
    }
    result["summary"].update(
        {
            "algorithm_id": variant_id,
            "algorithm_label": spec["variant_label"],
            "variant_id": variant_id,
            "variant_label": spec["variant_label"],
            "mechanism": spec["mechanism"],
            "result_source": RESULT_SOURCE,
            **architecture,
        }
    )
    for row in result["curves"]:
        row.update(
            {
                "algorithm_id": variant_id,
                "algorithm_label": spec["variant_label"],
                "variant_id": variant_id,
                "variant_label": spec["variant_label"],
                "result_source": RESULT_SOURCE,
            }
        )
    _assert_mechanism_invariants(variant_id, result, config)
    return result


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_variant(
        str(payload["variant_id"]),
        int(payload["seed"]),
        payload["config"],
        int(payload["target_nodes_touched"]),
    )
    shared._write_json(output_path, result)
    return 0


def _run_subprocess(
    run_dir: Path,
    variant_id: str,
    seed: int,
    config: Dict[str, Any],
    target_nodes: int,
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
            "config": config,
            "target_nodes_touched": target_nodes,
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run",
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
        raise RuntimeError(f"{stem} failed; see {log_path}")
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate(summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for variant_id in DEFAULT_VARIANT_IDS:
        rows = [row for row in summary_rows if row["variant_id"] == variant_id]
        if not rows:
            continue
        numeric_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregate[variant_id] = {
            field: shared._stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return aggregate


def _paired_differences(summary_rows: Sequence[Mapping[str, Any]]):
    indexed = {(row["variant_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for variant_id in DEFAULT_VARIANT_IDS:
        if variant_id == FULL_EXPERIMENT_6:
            continue
        for seed in sorted({int(row["seed"]) for row in summary_rows}):
            baseline = indexed.get((FULL_EXPERIMENT_6, seed))
            candidate = indexed.get((variant_id, seed))
            if baseline is None or candidate is None:
                continue
            rows.append(
                {
                    "variant_id": variant_id,
                    "variant_label": VARIANTS[variant_id]["variant_label"],
                    "seed": seed,
                    "final_exploitability_difference_vs_full": (
                        candidate["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "final_wall_clock_seconds_difference_vs_full": (
                        candidate["final_wall_clock_seconds"]
                        - baseline["final_wall_clock_seconds"]
                    ),
                    "wall_clock_ratio_vs_full": (
                        candidate["final_wall_clock_seconds"]
                        / baseline["final_wall_clock_seconds"]
                    ),
                    "node_difference_vs_full": (
                        candidate["final_nodes_touched"]
                        - baseline["final_nodes_touched"]
                    ),
                }
            )
    return rows


def _mean_curve(rows, variant_id: str, x_key: str):
    selected = [
        row
        for row in rows
        if row["variant_id"] == variant_id
        and not bool(row.get("is_final_policy_evaluation", False))
    ]
    checkpoints = sorted({int(row["checkpoint_index"]) for row in selected})
    xs, means, ses = [], [], []
    for checkpoint in checkpoints:
        checkpoint_rows = [
            row for row in selected if int(row["checkpoint_index"]) == checkpoint
        ]
        x = np.asarray([row[x_key] for row in checkpoint_rows], dtype=float)
        y = np.asarray([row["exploitability"] for row in checkpoint_rows], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            xs.append(float(np.mean(x[finite])))
            means.append(float(np.mean(y[finite])))
            ses.append(float(shared._stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _plot_exploitability(run_dir: Path, curve_rows, *, x_key: str) -> None:
    is_time = x_key == "wall_clock_seconds"
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for variant_id, spec in VARIANTS.items():
        selected = [row for row in curve_rows if row["variant_id"] == variant_id]
        if not selected:
            continue
        divisor = 3600.0 if is_time else 1.0
        x, mean, se = _mean_curve(curve_rows, variant_id, x_key)
        x = x / divisor
        ax.plot(
            x,
            mean,
            marker="o",
            markersize=3.5,
            linewidth=2.0,
            color=COLORS[variant_id],
            label=spec["variant_label"],
        )
        ax.fill_between(
            x,
            mean - se,
            mean + se,
            color=COLORS[variant_id],
            alpha=0.10,
        )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Wall-clock training time (hours)" if is_time else "Nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    dimension = "wall-clock time" if is_time else "nodes touched"
    set_chart_title(ax, f"Experiment 8 lean ablation by {dimension}")
    ax.legend(fontsize=8.5, ncol=2)
    fig.tight_layout()
    filename = (
        "exploitability_by_wall_clock.png"
        if is_time
        else "exploitability_by_nodes.png"
    )
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final_metrics(run_dir: Path, summary_rows) -> None:
    available = [
        variant_id
        for variant_id in DEFAULT_VARIANT_IDS
        if any(row["variant_id"] == variant_id for row in summary_rows)
    ]
    labels = [VARIANTS[variant_id]["variant_label"] for variant_id in available]
    exploitability_stats = [
        shared._stats(
            row["final_exploitability"]
            for row in summary_rows
            if row["variant_id"] == variant_id
        )
        for variant_id in available
    ]
    runtime_stats = [
        shared._stats(
            row["final_wall_clock_seconds"] / 3600.0
            for row in summary_rows
            if row["variant_id"] == variant_id
        )
        for variant_id in available
    ]
    for filename, title, ylabel, stats in (
        (
            "final_exploitability.png",
            "Experiment 8 final exploitability",
            "Final exploitability (NashConv / 2)",
            exploitability_stats,
        ),
        (
            "final_wall_clock.png",
            "Experiment 8 training cost at matched nodes",
            "Wall-clock training time (hours)",
            runtime_stats,
        ),
    ):
        fig, ax = plt.subplots(figsize=(12, 6.5))
        positions = np.arange(len(available))
        ax.bar(
            positions,
            [entry["mean"] for entry in stats],
            yerr=[entry["se"] for entry in stats],
            color=[COLORS[variant_id] for variant_id in available],
            capsize=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for variant_id, exp_stats, time_stats in zip(
        available, exploitability_stats, runtime_stats
    ):
        ax.errorbar(
            time_stats["mean"],
            exp_stats["mean"],
            xerr=time_stats["se"],
            yerr=exp_stats["se"],
            marker="o",
            markersize=8,
            linestyle="none",
            capsize=3,
            color=COLORS[variant_id],
            label=VARIANTS[variant_id]["variant_label"],
        )
    ax.set_xlabel("Wall-clock training time (hours; matched node budget)")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 8 performance-cost frontier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "performance_cost_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_aggregate_results(run_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    indexed = {}
    for run_dir in run_dirs:
        paths = sorted(run_dir.rglob("worker_results/*.json"))
        if not paths:
            raise ValueError(f"No worker results found under {run_dir}")
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            key = (
                str(result["summary"]["variant_id"]),
                int(result["summary"]["seed"]),
            )
            if key in indexed:
                raise ValueError(f"Duplicate aggregate result for {key}")
            indexed[key] = result
    return [indexed[key] for key in sorted(indexed)]


def _finalize(run_dir: Path, results, failures, metadata) -> None:
    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    paired_rows = _paired_differences(summary_rows)
    aggregate = _aggregate(summary_rows)
    shared._write_json(run_dir / "experiment_metadata.json", metadata)
    shared._write_csv(run_dir / "seed_summary.csv", summary_rows)
    shared._write_csv(run_dir / "checkpoint_curves.csv", curve_rows)
    shared._write_csv(run_dir / "paired_differences_vs_full.csv", paired_rows)
    shared._write_json(run_dir / "aggregate_summary.json", aggregate)
    shared._write_json(
        run_dir / "summary.json",
        {"seed_summary": summary_rows, "aggregate": aggregate, "failures": failures},
    )
    if curve_rows:
        _plot_exploitability(run_dir, curve_rows, x_key="nodes_touched")
        _plot_exploitability(run_dir, curve_rows, x_key="wall_clock_seconds")
        _plot_final_metrics(run_dir, summary_rows)


def _metadata(seeds, variants, targets, base_config):
    return {
        "experiment_id": EXPERIMENT_ID,
        "seeds": seeds,
        "variant_ids": variants,
        "variants": VARIANTS,
        "target_nodes_touched_by_seed": targets,
        "base_config": base_config,
        "measured_full_experiment_6_hours_per_seed": (
            MEASURED_FULL_EXPERIMENT_6_HOURS_PER_SEED
        ),
        "expected_sequential_runtime_hours": EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
        "sequential_batch_timeout_seconds": SEQUENTIAL_BATCH_TIMEOUT_SECONDS,
        "expected_parallel_by_variant_runtime_hours": (
            EXPECTED_PARALLEL_BY_VARIANT_RUNTIME_HOURS
        ),
        "parallel_by_variant_batch_timeout_seconds": (
            PARALLEL_BY_VARIANT_BATCH_TIMEOUT_SECONDS
        ),
        "per_worker_batch_timeout_seconds": PER_WORKER_BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "pairing": "Every variant uses seeds 0, 1 and 2 and the matching Experiment 6 per-seed node target.",
            "isolation": "Requested arms change exactly the named mechanism; lean_candidate composes the removals explicitly.",
            "evaluation": "Untrained, approximately 10k-node and every-outer-iteration checkpoints match Experiment 6.",
            "estimator": "All arms retain the always-unbiased importance-corrected control-variate estimator.",
            "wall_clock": "Fresh full-control runs make training-cost comparisons contemporaneous and like-for-like.",
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/unbiased_control_variate_escher_lean_ablation",
    )
    parser.add_argument("--seeds")
    parser.add_argument("--variants")
    parser.add_argument(
        "--target-nodes",
        type=int,
        help="Override the paired Experiment 6 per-seed targets with one target.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--calibration-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument(
        "--aggregate-run-dir",
        action="append",
        type=Path,
        default=[],
        help="Read worker results under this run directory; repeat as needed.",
    )
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_json)
    if args.target_nodes is not None and args.target_nodes <= 0:
        raise ValueError("target-nodes must be positive")

    seeds = _parse_csv_ints(args.seeds, DEFAULT_SEEDS)
    variants = _parse_variants(args.variants)
    unknown_seeds = sorted(set(seeds) - set(EXPERIMENT_2_NODE_TARGETS))
    if args.target_nodes is None and unknown_seeds:
        raise ValueError(
            "A --target-nodes override is required for seeds without an "
            f"Experiment 6 target: {unknown_seeds}"
        )
    targets = {
        seed: (
            int(args.target_nodes)
            if args.target_nodes is not None
            else int(EXPERIMENT_2_NODE_TARGETS[seed])
        )
        for seed in seeds
    }
    base_config = deepcopy(BASE_CONFIG)
    _apply_overrides(args, base_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root)
        / f"unbiased_control_variate_escher_lean_ablation_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = _metadata(seeds, variants, targets, base_config)

    if args.aggregate_run_dir:
        results = _load_aggregate_results(args.aggregate_run_dir)
        metadata["aggregate_source_run_dirs"] = [
            str(path) for path in args.aggregate_run_dir
        ]
        metadata["variant_ids"] = sorted(
            {str(result["summary"]["variant_id"]) for result in results}
        )
        metadata["seeds"] = sorted(
            {int(result["summary"]["seed"]) for result in results}
        )
        metadata["target_nodes_touched_by_seed"] = {
            int(result["summary"]["seed"]): int(
                result["summary"]["target_nodes_touched"]
            )
            for result in results
        }
        _finalize(run_dir, results, [], metadata)
        LOGGER.info("Aggregated outputs saved to %s", run_dir.resolve())
        return 0

    results, failures = [], []
    schedule = [(variant_id, seed) for variant_id in variants for seed in seeds]
    for variant_id, seed in schedule:
        config = _variant_config(variant_id, base_config)
        try:
            LOGGER.info(
                "Running Experiment 8 %s seed %s to %s nodes",
                variant_id,
                seed,
                targets[seed],
            )
            result = _run_subprocess(
                run_dir,
                variant_id,
                seed,
                config,
                targets[seed],
            )
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

