"""Run Experiment 5: exact Leduc forensics and mechanism ablations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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
    DEFAULT_FINAL_WINDOW,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as exp3,
)
from vr_deep_cfr.logger import Logger  # noqa: E402

from .config import (  # noqa: E402
    CONTROL_VARIANT,
    DEFAULT_SEEDS,
    EXPERIMENT_1_NODE_TARGETS,
    FORENSIC_CONFIG,
    VARIANTS,
)
from .solver import ForensicAdaptiveResidualPredictiveEscher  # noqa: E402


LOGGER = logging.getLogger("adaptive_residual_predictive_escher_forensics")

DIAGNOSTIC_FIELDS = (
    "current_predictive_exploitability",
    "current_nonpredictive_exploitability",
    "predictive_exploitability_improvement",
    "exact_average_exploitability",
    "neural_average_exploitability",
    "average_policy_distillation_gap",
    "predictor_preupdate_mse",
    "predictor_postupdate_mse",
    "predictor_preupdate_mse_player_0",
    "predictor_preupdate_mse_player_1",
    "predictor_postupdate_mse_player_0",
    "predictor_postupdate_mse_player_1",
    "q_oracle_mae",
    "q_oracle_rmse",
    "estimator_bias_abs_mean",
    "estimator_variance_mean",
    "estimator_mse_mean",
    "forensic_diagnostic_wall_clock_seconds",
    "adaptive_lambda_schedule_floor",
    "adaptive_lambda_mean",
    "adaptive_lambda_min",
    "adaptive_lambda_max",
    "q_residual_abs_mean",
    "residual_correction_abs_mean",
    "policy_weighted_advantage_abs_mean",
    "adaptive_estimator_sample_count",
    "full_support_traverser_sampling_min_probability",
    "q_target_version",
)


def _stats(values: Iterable[float]) -> Dict[str, float | int]:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "n_finite": 0}
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {
        "mean": float(np.mean(finite)),
        "std": std,
        "se": std / math.sqrt(finite.size),
        "n_finite": int(finite.size),
    }


def _parse_float(value) -> float:
    if value is None or value == "":
        return np.nan
    return float(value)


def _run_variant(
    variant_id: str,
    seed: int,
    config: Dict[str, Any],
    target_nodes: int,
):
    variant = VARIANTS[variant_id]
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
        diagnostic_variant_id=variant_id,
        lambda_mode=variant["lambda_mode"],
        use_predictive_accumulator=variant["use_predictive_accumulator"],
        q_mode=variant["q_mode"],
    )
    solver = ForensicAdaptiveResidualPredictiveEscher(**kwargs)
    solver.target_nodes_touched = int(target_nodes)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )
    raw_checkpoints = solver.solve()

    curves = []
    for checkpoint_index, raw in enumerate(raw_checkpoints):
        value = float(raw["average_policy_value"])
        row = {
            "variant_id": variant_id,
            "variant_label": variant["label"],
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
            "is_final_policy_evaluation": False,
        }
        for field in DIAGNOSTIC_FIELDS:
            row[field] = _parse_float(raw.get(field))
        curves.append(row)

    final = curves[-1]
    training_curves = [
        row for row in curves if not row["is_initial_policy_evaluation"]
    ]
    neural_exploitabilities = [row["exploitability"] for row in training_curves]
    node_delta = float(final["nodes_touched"] - target_nodes)
    summary = {
        "variant_id": variant_id,
        "variant_label": variant["label"],
        "seed": int(seed),
        "final_exploitability": float(final["exploitability"]),
        "best_exploitability": float(np.min(neural_exploitabilities)),
        "final_window_mean_exploitability": float(
            np.mean(
                neural_exploitabilities[
                    -min(DEFAULT_FINAL_WINDOW, len(neural_exploitabilities)) :
                ]
            )
        ),
        "final_nodes_touched": float(final["nodes_touched"]),
        "target_nodes_touched": float(target_nodes),
        "node_budget_delta": node_delta,
        "node_budget_relative_delta": node_delta / float(target_nodes),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "num_iterations_completed": int(final["iteration"]),
        "num_checkpoints": len(curves),
    }
    for field in DIAGNOSTIC_FIELDS:
        summary[f"final_{field}"] = float(final[field])

    def enrich(rows):
        return [
            {
                "variant_id": variant_id,
                "variant_label": variant["label"],
                "seed": int(seed),
                **row,
            }
            for row in rows
        ]

    return {
        "variant_id": variant_id,
        "seed": int(seed),
        "summary": summary,
        "curves": curves,
        "strategy_diagnostics": enrich(solver.strategy_diagnostic_rows),
        "q_oracle_diagnostics": enrich(solver.q_oracle_diagnostic_rows),
        "estimator_diagnostics": enrich(solver.estimator_diagnostic_rows),
    }


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_variant(
        str(payload["variant_id"]),
        int(payload["seed"]),
        payload["config"],
        int(payload["target_nodes_touched"]),
    )
    exp3._write_json(output_path, result)
    return 0


def _run_subprocess(run_dir, variant_id, seed, config, target_nodes):
    stem = f"{variant_id}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    exp3._write_json(
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
        "experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run",
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


def _mean_curve(rows, variant_id, metric):
    selected = [row for row in rows if row["variant_id"] == variant_id]
    checkpoints = sorted({int(row["checkpoint_index"]) for row in selected})
    xs, means, ses = [], [], []
    for checkpoint in checkpoints:
        at_checkpoint = [
            row for row in selected if int(row["checkpoint_index"]) == checkpoint
        ]
        x = np.asarray([row["nodes_touched"] for row in at_checkpoint], dtype=float)
        y = np.asarray([row.get(metric, np.nan) for row in at_checkpoint], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            xs.append(float(np.mean(x[finite])))
            means.append(float(np.mean(y[finite])))
            ses.append(float(_stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _variant_colors(variant_ids):
    palette = ["#9467bd", "#1f77b4", "#8c564b", "#ff7f0e", "#2ca02c", "#d62728"]
    return {
        variant_id: palette[index % len(palette)]
        for index, variant_id in enumerate(variant_ids)
    }


def _plot_ablation_curves(run_dir, curves, variant_ids):
    colors = _variant_colors(variant_ids)
    fig, ax = plt.subplots(figsize=(12, 7))
    for variant_id in variant_ids:
        x, mean, se = _mean_curve(curves, variant_id, "exploitability")
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            color=colors[variant_id],
            label=VARIANTS[variant_id]["label"],
        )
        ax.fill_between(x, mean - se, mean + se, color=colors[variant_id], alpha=0.12)
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Training nodes touched")
    ax.set_ylabel("Neural average-policy exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 5 mechanism ablations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "ablation_exploitability_by_nodes.png", dpi=200)
    plt.close(fig)


def _plot_strategy_decomposition(run_dir, curves):
    metrics = {
        "current_predictive_exploitability": "Current predictive strategy",
        "current_nonpredictive_exploitability": "Current cumulative-only strategy",
        "exact_average_exploitability": "Exact weighted tabular average",
        "neural_average_exploitability": "Neural average-policy approximation",
    }
    colors = ["#9467bd", "#ff7f0e", "#2ca02c", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for (metric, label), color in zip(metrics.items(), colors):
        x, mean, se = _mean_curve(curves, CONTROL_VARIANT, metric)
        ax.plot(x, mean, marker="o", linewidth=2.2, color=color, label=label)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.13)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Training nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 5 control: strategy bottleneck decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "control_strategy_decomposition_by_nodes.png", dpi=200)
    plt.close(fig)


def _plot_q_error(run_dir, curves, variant_ids):
    colors = _variant_colors(variant_ids)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for variant_id in variant_ids:
        x, mean, se = _mean_curve(curves, variant_id, "q_oracle_rmse")
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            color=colors[variant_id],
            label=VARIANTS[variant_id]["label"],
        )
        ax.fill_between(x, mean - se, mean + se, color=colors[variant_id], alpha=0.12)
    ax.set_xlabel("Training nodes touched")
    ax.set_ylabel("Exact all-action Q RMSE")
    set_chart_title(ax, "Experiment 5 Q error against the exact Leduc oracle")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "q_oracle_error_by_nodes.png", dpi=200)
    plt.close(fig)


def _plot_estimator_diagnostics(run_dir, curves, variant_ids):
    colors = _variant_colors(variant_ids)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for variant_id in variant_ids:
        for ax, metric, ylabel in (
            (axes[0], "estimator_bias_abs_mean", "Mean absolute estimator bias"),
            (axes[1], "estimator_variance_mean", "Mean estimator variance"),
        ):
            x, mean, se = _mean_curve(curves, variant_id, metric)
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=1.8,
                color=colors[variant_id],
                label=VARIANTS[variant_id]["label"],
            )
            ax.fill_between(
                x,
                mean - se,
                mean + se,
                color=colors[variant_id],
                alpha=0.1,
            )
            ax.set_xlabel("Training nodes touched")
            ax.set_ylabel(ylabel)
    set_chart_title(axes[0], "Estimator bias")
    set_chart_title(axes[1], "Estimator variance")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "estimator_bias_variance_by_nodes.png", dpi=200)
    plt.close(fig)


def _predictor_ablation_rows(curves):
    controls = {
        (int(row["seed"]), int(row["checkpoint_index"])): row
        for row in curves
        if row["variant_id"] == CONTROL_VARIANT
    }
    nonpredictive = {
        (int(row["seed"]), int(row["checkpoint_index"])): row
        for row in curves
        if row["variant_id"] == "nonpredictive_accumulator"
    }
    rows = []
    for key in sorted(controls.keys() & nonpredictive.keys()):
        control = controls[key]
        baseline = nonpredictive[key]
        preupdate_error = _parse_float(control.get("predictor_preupdate_mse"))
        postupdate_error = _parse_float(control.get("predictor_postupdate_mse"))
        control_exploitability = _parse_float(
            control.get("current_predictive_exploitability")
        )
        baseline_exploitability = _parse_float(
            baseline.get("current_nonpredictive_exploitability")
        )
        rows.append(
            {
                "seed": key[0],
                "checkpoint_index": key[1],
                "control_nodes_touched": float(control["nodes_touched"]),
                "nonpredictive_nodes_touched": float(baseline["nodes_touched"]),
                "predictor_preupdate_mse": preupdate_error,
                "predictor_postupdate_mse": postupdate_error,
                "predictive_control_current_exploitability": (
                    control_exploitability
                ),
                "nonpredictive_current_exploitability": baseline_exploitability,
                "predictive_update_exploitability_improvement": (
                    baseline_exploitability - control_exploitability
                ),
            }
        )
    return rows


def _plot_predictor_error(run_dir, rows):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    selected = [
        row
        for row in rows
        if np.isfinite(row["predictor_preupdate_mse"])
        and np.isfinite(row["predictive_update_exploitability_improvement"])
    ]
    points = ax.scatter(
        [row["predictor_preupdate_mse"] for row in selected],
        [row["predictive_update_exploitability_improvement"] for row in selected],
        c=[row["control_nodes_touched"] for row in selected],
        alpha=0.8,
        cmap="viridis",
    )
    if selected:
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("Control training nodes touched")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predictive-control pre-update predictor MSE")
    ax.set_ylabel(
        "Non-predictive arm minus predictive-control current exploitability"
    )
    set_chart_title(
        ax,
        "Does predictor accuracy translate into predictive-update improvement?",
    )
    fig.tight_layout()
    fig.savefig(run_dir / "predictor_error_vs_strategy_improvement.png", dpi=200)
    plt.close(fig)


def _plot_final(run_dir, summaries, variant_ids):
    means, ses, labels = [], [], []
    for variant_id in variant_ids:
        values = [
            row["final_exploitability"]
            for row in summaries
            if row["variant_id"] == variant_id
        ]
        stats = _stats(values)
        means.append(stats["mean"])
        ses.append(stats["se"])
        labels.append(VARIANTS[variant_id]["label"])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Final neural average-policy exploitability")
    set_chart_title(ax, "Experiment 5 final mechanism-ablation performance")
    fig.tight_layout()
    fig.savefig(run_dir / "final_ablation_exploitability.png", dpi=200)
    plt.close(fig)


def _aggregate(summaries, variant_ids):
    aggregate = {}
    for variant_id in variant_ids:
        rows = [row for row in summaries if row["variant_id"] == variant_id]
        numeric_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregate[variant_id] = {
            field: _stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return aggregate


def _parse_list(value, defaults):
    if value is None:
        return list(defaults)
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_overrides(args, config):
    overrides = {
        "num_traversals": args.traversals,
        "max_num_iterations": args.max_iterations,
        "advantage_network_train_steps": args.advantage_train_steps,
        "ave_policy_network_train_steps": args.policy_train_steps,
        "baseline_network_train_steps": args.q_train_steps,
        "advantage_batch_size": args.batch_size,
        "ave_policy_batch_size": args.batch_size,
        "baseline_batch_size": args.batch_size,
        "advantage_buffer_size": args.buffer_size,
        "ave_policy_buffer_size": args.buffer_size,
        "baseline_buffer_size": args.buffer_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (args.early_evaluation_nodes,)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/adaptive_residual_predictive_escher_forensics",
    )
    parser.add_argument("--variants")
    parser.add_argument("--seeds")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--target-nodes", type=int)
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
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

    variant_ids = _parse_list(args.variants, VARIANTS)
    unknown = [variant_id for variant_id in variant_ids if variant_id not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    seeds = [int(seed) for seed in _parse_list(args.seeds, DEFAULT_SEEDS)]
    if any(seed not in EXPERIMENT_1_NODE_TARGETS for seed in seeds):
        raise ValueError("Experiment 5 supports paired seeds 0, 1 and 2")
    config = deepcopy(FORENSIC_CONFIG)
    _apply_overrides(args, config)
    targets = {
        seed: int(args.target_nodes or EXPERIMENT_1_NODE_TARGETS[seed])
        for seed in seeds
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"adaptive_escher_forensics_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    metadata = {
        "experiment_id": 5,
        "seeds": seeds,
        "variant_ids": variant_ids,
        "variants": {variant_id: VARIANTS[variant_id] for variant_id in variant_ids},
        "training_config": config,
        "paired_node_targets": targets,
        "diagnostic_contract": {
            "training_nodes": "Exact tree diagnostics are excluded from nodes_touched.",
            "current_strategy": "Exact exploitability of predictive regret matching.",
            "nonpredictive_strategy": (
                "Exact exploitability using cumulative advantages only."
            ),
            "exact_average": (
                "Exact tabular own-reach and iteration^gamma weighted average."
            ),
            "neural_average": "The normal learned average-policy approximation.",
            "q_oracle": (
                "All legal Q actions compared with exact Leduc continuation values."
            ),
            "estimator": (
                "Exact frozen-controller recursive estimator moments grouped by "
                "information set and action."
            ),
            "predictor_error": (
                "Instantaneous predictor MSE on the newly collected pre-update batch."
            ),
        },
    }
    exp3._write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    for variant_id in variant_ids:
        for seed in seeds:
            try:
                LOGGER.info("Running %s seed %s", variant_id, seed)
                result = _run_subprocess(
                    run_dir, variant_id, seed, config, targets[seed]
                )
                results.append(result)
                exp3._write_json(
                    run_dir / "partial_results.json",
                    [
                        {
                            "variant_id": completed["variant_id"],
                            "seed": completed["seed"],
                            "summary": completed["summary"],
                        }
                        for completed in results
                    ],
                )
            except Exception as exc:  # pragma: no cover - operational path
                failure = {
                    "variant_id": variant_id,
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                exp3._write_json(run_dir / "failed_runs.json", failures)
                LOGGER.error("%s seed %s failed: %s", variant_id, seed, exc)
                if not args.continue_on_error:
                    return 2

    summaries = [result["summary"] for result in results]
    curves = [row for result in results for row in result["curves"]]
    strategy_rows = [
        row for result in results for row in result["strategy_diagnostics"]
    ]
    q_rows = [row for result in results for row in result["q_oracle_diagnostics"]]
    estimator_rows = [
        row for result in results for row in result["estimator_diagnostics"]
    ]
    predictor_ablation_rows = _predictor_ablation_rows(curves)
    aggregate = _aggregate(summaries, variant_ids)

    exp3._write_csv(run_dir / "seed_summary.csv", summaries)
    exp3._write_csv(run_dir / "checkpoint_curves.csv", curves)
    exp3._write_csv(run_dir / "strategy_diagnostics.csv", strategy_rows)
    exp3._write_csv(run_dir / "q_oracle_diagnostics.csv", q_rows)
    exp3._write_csv(run_dir / "estimator_diagnostics.csv", estimator_rows)
    exp3._write_csv(
        run_dir / "predictor_ablation_diagnostics.csv",
        predictor_ablation_rows,
    )
    exp3._write_json(run_dir / "aggregate_summary.json", aggregate)
    exp3._write_json(
        run_dir / "summary.json",
        {"seed_summary": summaries, "aggregate": aggregate, "failures": failures},
    )

    if curves:
        _plot_ablation_curves(run_dir, curves, variant_ids)
        if CONTROL_VARIANT in variant_ids:
            _plot_strategy_decomposition(run_dir, curves)
        _plot_q_error(run_dir, curves, variant_ids)
        _plot_estimator_diagnostics(run_dir, curves, variant_ids)
        if (
            CONTROL_VARIANT in variant_ids
            and "nonpredictive_accumulator" in variant_ids
        ):
            _plot_predictor_error(run_dir, predictor_ablation_rows)
        _plot_final(run_dir, summaries, variant_ids)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
