"""Paired performance, cost, and mechanism analysis for Experiment 22."""

from __future__ import annotations

import csv
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (  # noqa: E402
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    holm_adjust,
    summary,
)

from .config import (  # noqa: E402
    FIXED_BETA_ONE,
    FULL_EXPERIMENT_6,
    NONINFERIORITY_MARGIN,
    TWO_CROSS_FITTED_CRITICS,
    VARIANTS,
    VARIANT_ORDER,
    contract_manifest,
)


COLOURS = {
    FULL_EXPERIMENT_6: "#9467bd",
    FIXED_BETA_ONE: "#1f77b4",
    TWO_CROSS_FITTED_CRITICS: "#2ca02c",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _curve_plot(rows: Sequence[Mapping[str, Any]], output: Path, *, x_key: str) -> None:
    time_axis = x_key == "wall_clock_seconds"
    divisor = 3600.0 if time_axis else 1_000_000.0
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for variant_id in VARIANT_ORDER:
        selected = [row for row in rows if row["variant_id"] == variant_id]
        for seed in sorted({int(row["seed"]) for row in selected}):
            seed_rows = sorted(
                [row for row in selected if int(row["seed"]) == seed],
                key=lambda row: float(row[x_key]),
            )
            ax.plot(
                [float(row[x_key]) / divisor for row in seed_rows],
                [float(row["exploitability"]) for row in seed_rows],
                color=COLOURS[variant_id],
                alpha=0.15,
                linewidth=0.8,
            )
        by_checkpoint = defaultdict(list)
        for row in selected:
            by_checkpoint[int(row["checkpoint_index"])].append(row)
        x_values, means, errors = [], [], []
        for checkpoint in sorted(by_checkpoint):
            group = by_checkpoint[checkpoint]
            x_values.append(np.mean([float(row[x_key]) for row in group]) / divisor)
            values = [float(row["exploitability"]) for row in group]
            stats = summary(values)
            means.append(stats["mean_ev"])
            errors.append(stats["standard_error"])
        ax.plot(
            x_values,
            means,
            color=COLOURS[variant_id],
            linewidth=2,
            marker="o",
            markersize=3,
            label=VARIANTS[variant_id]["variant_label"],
        )
        ax.fill_between(
            x_values,
            np.asarray(means) - np.asarray(errors),
            np.asarray(means) + np.asarray(errors),
            color=COLOURS[variant_id],
            alpha=0.12,
        )
    ax.set_xlabel("Wall-clock training time (hours)" if time_axis else "Nodes touched (millions)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Experiment 22 exploitability by " + ("wall-clock time" if time_axis else "nodes"),
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _final_plots(summary_rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    labels = [VARIANTS[variant]["variant_label"] for variant in VARIANT_ORDER]
    exploitability = []
    runtime = []
    memory = []
    for variant_id in VARIANT_ORDER:
        rows = [row for row in summary_rows if row["variant_id"] == variant_id]
        exploitability.append(summary([float(row["final_exploitability"]) for row in rows]))
        runtime.append(summary([float(row["final_wall_clock_seconds"]) / 3600 for row in rows]))
        memory.append(summary([float(row["peak_rss_mb"]) for row in rows]))
    for filename, title, ylabel, values in (
        ("final_exploitability.png", "Experiment 22 final exploitability", "Exploitability (NashConv / 2)", exploitability),
        ("final_runtime.png", "Experiment 22 matched-node runtime", "Wall-clock time (hours)", runtime),
        ("peak_memory.png", "Experiment 22 peak resident memory", "Peak RSS (MiB)", memory),
    ):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        positions = np.arange(len(labels))
        ax.bar(
            positions,
            [entry["mean_ev"] for entry in values],
            yerr=[entry["standard_error"] for entry in values],
            color=[COLOURS[variant] for variant in VARIANT_ORDER],
            capsize=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for variant_id, exp_stats, runtime_stats in zip(VARIANT_ORDER, exploitability, runtime):
        ax.errorbar(
            runtime_stats["mean_ev"],
            exp_stats["mean_ev"],
            xerr=runtime_stats["standard_error"],
            yerr=exp_stats["standard_error"],
            marker="o",
            capsize=3,
            color=COLOURS[variant_id],
            label=VARIANTS[variant_id]["variant_label"],
        )
    ax.set_xlabel("Wall-clock training time (hours)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 22 performance-cost frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "performance_cost_frontier.png", dpi=200)
    plt.close(fig)


def _mechanism_plot(
    curve_rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    fields: Sequence[str],
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for variant_id in VARIANT_ORDER:
        selected = [row for row in curve_rows if row["variant_id"] == variant_id]
        by_checkpoint = defaultdict(list)
        for row in selected:
            values = [float(row[field]) for field in fields]
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                by_checkpoint[int(row["checkpoint_index"])].append(
                    (float(row["nodes_touched"]) / 1_000_000.0, float(np.mean(finite)))
                )
        x, y = [], []
        for checkpoint in sorted(by_checkpoint):
            values = by_checkpoint[checkpoint]
            x.append(float(np.mean([value[0] for value in values])))
            y.append(float(np.mean([value[1] for value in values])))
        ax.plot(
            x,
            y,
            color=COLOURS[variant_id],
            marker="o",
            markersize=3,
            label=VARIANTS[variant_id]["variant_label"],
        )
    ax.set_xlabel("Nodes touched (millions)")
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _diagnostic_worker_summary(
    information_rows: Sequence[Mapping[str, Any]],
    critic_rows: Sequence[Mapping[str, Any]],
) -> dict:
    weights = np.asarray([float(row["target_count"]) for row in information_rows])
    current = np.asarray([float(row["realised_target_variance"]) for row in information_rows])
    fixed = np.asarray(
        [float(row["counterfactual_fixed_beta_one_target_variance"]) for row in information_rows]
    )
    sampled_weights = np.asarray([float(row["sample_count"]) for row in information_rows])
    observed = np.asarray([float(row["observed_residual_variance"]) for row in information_rows])
    predicted = np.asarray(
        [float(row["mean_predicted_residual_variance"]) for row in information_rows]
    )
    finite_target = np.isfinite(weights) & np.isfinite(current) & np.isfinite(fixed)
    finite_calibration = (
        np.isfinite(sampled_weights) & np.isfinite(observed) & np.isfinite(predicted)
    )
    target_numerator = float(np.sum(weights[finite_target] * current[finite_target]))
    target_denominator = float(np.sum(weights[finite_target] * fixed[finite_target]))
    observed_variance = float(
        np.average(observed[finite_calibration], weights=sampled_weights[finite_calibration])
    ) if np.any(finite_calibration) and np.sum(sampled_weights[finite_calibration]) > 0 else math.nan
    predicted_variance = float(
        np.average(predicted[finite_calibration], weights=sampled_weights[finite_calibration])
    ) if np.any(finite_calibration) and np.sum(sampled_weights[finite_calibration]) > 0 else math.nan
    x = np.asarray([float(row["sampled_critic_target_rmse"]) for row in critic_rows])
    y = np.asarray(
        [float(row["next_local_regret_target_abs_mean"]) for row in critic_rows]
    )
    finite_correlation = np.isfinite(x) & np.isfinite(y)
    correlation = (
        float(np.corrcoef(x[finite_correlation], y[finite_correlation])[0, 1])
        if np.sum(finite_correlation) >= 3
        and np.std(x[finite_correlation]) > 0
        and np.std(y[finite_correlation]) > 0
        else math.nan
    )
    return {
        "weighted_target_variance_ratio_vs_fixed_beta_one": (
            target_numerator / target_denominator if target_denominator > 0 else math.nan
        ),
        "weighted_observed_residual_variance": observed_variance,
        "weighted_predicted_residual_variance": predicted_variance,
        "weighted_residual_variance_calibration_ratio": (
            observed_variance / predicted_variance
            if predicted_variance > 0 and math.isfinite(observed_variance)
            else math.nan
        ),
        "critic_error_next_local_regret_correlation": correlation,
        "num_information_action_rows": len(information_rows),
        "num_critic_lag_rows": len(critic_rows),
    }


def _diagnostic_plots(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    beta_rows: Sequence[Mapping[str, Any]],
    critic_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    labels = [VARIANTS[variant]["variant_label"] for variant in VARIANT_ORDER]
    variance_stats = []
    for variant_id in VARIANT_ORDER:
        values = [
            float(row["weighted_target_variance_ratio_vs_fixed_beta_one"])
            for row in diagnostic_rows
            if row["variant_id"] == variant_id
        ]
        variance_stats.append(summary(values))
    fig, ax = plt.subplots(figsize=(9, 5.8))
    positions = np.arange(len(VARIANT_ORDER))
    ax.bar(
        positions,
        [entry["mean_ev"] for entry in variance_stats],
        yerr=[entry["standard_error"] for entry in variance_stats],
        color=[COLOURS[variant] for variant in VARIANT_ORDER],
        capsize=4,
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Realised target variance / counterfactual beta=1 variance")
    set_chart_title(ax, "Experiment 22 realised target-variance ratio")
    fig.tight_layout()
    fig.savefig(output_dir / "target_variance_ratio.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for variant_id in VARIANT_ORDER:
        rows = [row for row in diagnostic_rows if row["variant_id"] == variant_id]
        ax.scatter(
            [float(row["weighted_predicted_residual_variance"]) for row in rows],
            [float(row["weighted_observed_residual_variance"]) for row in rows],
            color=COLOURS[variant_id],
            label=VARIANTS[variant_id]["variant_label"],
            s=42,
        )
    limits = ax.get_xlim()
    lower = max(0.0, min(limits[0], ax.get_ylim()[0]))
    upper = max(limits[1], ax.get_ylim()[1])
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted residual variance")
    ax.set_ylabel("Mean realised residual variance")
    set_chart_title(ax, "Experiment 22 residual-variance calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_reliability.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for variant_id in VARIANT_ORDER:
        rows = [row for row in critic_rows if row["variant_id"] == variant_id]
        ax.scatter(
            [float(row["sampled_critic_target_rmse"]) for row in rows],
            [float(row["next_local_regret_target_abs_mean"]) for row in rows],
            color=COLOURS[variant_id],
            label=VARIANTS[variant_id]["variant_label"],
            s=10,
            alpha=0.18,
        )
    ax.set_xlabel("Sampled critic-target RMSE in iteration t")
    ax.set_ylabel("Mean absolute local-regret target in iteration t+1")
    set_chart_title(ax, "Experiment 22 critic residual and subsequent local regret")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "critic_error_vs_subsequent_local_regret.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for variant_id in VARIANT_ORDER:
        rows = [row for row in beta_rows if row["variant_id"] == variant_id]
        grouped = defaultdict(float)
        for row in rows:
            midpoint = 0.5 * (float(row["bin_lower"]) + float(row["bin_upper"]))
            grouped[midpoint] += float(row["count"])
        total = sum(grouped.values())
        x = sorted(grouped)
        y = [grouped[value] / total if total else 0.0 for value in x]
        ax.plot(
            x,
            y,
            color=COLOURS[variant_id],
            label=VARIANTS[variant_id]["variant_label"],
        )
    ax.set_xlabel("Beta bin midpoint")
    ax.set_ylabel("Fraction of recorded estimator targets")
    set_chart_title(ax, "Experiment 22 beta distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "beta_distribution.png", dpi=200)
    plt.close(fig)


def aggregate_workers(
    *, workers_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    workers_root = Path(workers_root).resolve()
    output_dir = Path(output_dir).resolve()
    expected = {(variant, int(seed)) for variant in VARIANT_ORDER for seed in seeds}
    results = {}
    for path in workers_root.rglob("worker_result.json"):
        result = read_json(path)
        key = (str(result["variant_id"]), int(result["seed"]))
        if key in results:
            raise ValueError(f"Duplicate worker result for {key}")
        if result.get("status") != "complete" or bool(result.get("smoke")) != bool(smoke):
            raise ValueError(f"Incomplete or wrong-mode worker: {path}")
        results[key] = (path, result)
    if set(results) != expected:
        raise ValueError(
            f"Worker set mismatch; missing={sorted(expected-set(results))}, "
            f"extra={sorted(set(results)-expected)}"
        )

    commits = {result["repository_commit"] for _, result in results.values()}
    if len(commits) != 1:
        raise ValueError(f"Workers used different repository commits: {sorted(commits)}")
    summaries, curves, information_rows, beta_rows, critic_rows = [], [], [], [], []
    manifest = []
    diagnostic_summaries = []
    for key in sorted(results):
        result_path, result = results[key]
        root = result_path.parent
        snapshot_path = root / result["snapshot"]["relative_path"]
        if not snapshot_path.is_file() or sha256(snapshot_path) != result["snapshot"]["sha256"]:
            raise ValueError(f"Missing or corrupt policy snapshot: {snapshot_path}")
        summaries.append(result["summary"])
        worker_curves = _read_csv(root / result["artifacts"]["checkpoint_curves"])
        worker_information = _read_csv(
            root / result["artifacts"]["information_action_diagnostics"]
        )
        worker_beta = _read_csv(root / result["artifacts"]["beta_histogram"])
        worker_critic = _read_csv(
            root / result["artifacts"]["critic_error_subsequent_local_regret"]
        )
        for collection in (worker_curves, worker_information, worker_beta, worker_critic):
            for row in collection:
                row.update({"variant_id": key[0], "seed": key[1]})
        curves.extend(worker_curves)
        information_rows.extend(worker_information)
        beta_rows.extend(worker_beta)
        critic_rows.extend(worker_critic)
        diagnostic_summaries.append(
            {
                "variant_id": key[0],
                "variant_label": VARIANTS[key[0]]["variant_label"],
                "seed": key[1],
                **_diagnostic_worker_summary(worker_information, worker_critic),
            }
        )
        manifest.append(
            {
                "variant_id": key[0],
                "seed": key[1],
                "worker_result": str(result_path),
                "snapshot": str(snapshot_path),
                "snapshot_sha256": result["snapshot"]["sha256"],
                "repository_commit": result["repository_commit"],
            }
        )

    indexed = {(row["variant_id"], int(row["seed"])): row for row in summaries}
    paired = []
    inference = []
    for candidate in VARIANT_ORDER[1:]:
        differences, runtime_ratios, memory_ratios = [], [], []
        for seed in seeds:
            control = indexed[(FULL_EXPERIMENT_6, int(seed))]
            arm = indexed[(candidate, int(seed))]
            difference = float(arm["final_exploitability"]) - float(
                control["final_exploitability"]
            )
            runtime_ratio = float(arm["final_wall_clock_seconds"]) / float(
                control["final_wall_clock_seconds"]
            )
            memory_ratio = float(arm["peak_rss_mb"]) / float(control["peak_rss_mb"])
            differences.append(difference)
            runtime_ratios.append(runtime_ratio)
            memory_ratios.append(memory_ratio)
            paired.append(
                {
                    "candidate_variant_id": candidate,
                    "candidate_variant_label": VARIANTS[candidate]["variant_label"],
                    "seed": int(seed),
                    "exploitability_difference_vs_full": difference,
                    "runtime_ratio_vs_full": runtime_ratio,
                    "peak_memory_ratio_vs_full": memory_ratio,
                    "candidate_node_overshoot": int(arm["node_overshoot"]),
                    "full_node_overshoot": int(control["node_overshoot"]),
                }
            )
        effect = summary(differences)
        runtime_effect = summary(runtime_ratios)
        memory_effect = summary(memory_ratios)
        inference.append(
            {
                "candidate_variant_id": candidate,
                "candidate_variant_label": VARIANTS[candidate]["variant_label"],
                "n_seeds": effect["n_seeds"],
                "mean_exploitability_difference_vs_full": effect["mean_ev"],
                "ci95_lower_exploitability_difference": effect["ci95_lower"],
                "ci95_upper_exploitability_difference": effect["ci95_upper"],
                "two_sided_exact_sign_flip_p": effect["two_sided_exact_sign_flip_p"],
                "noninferiority_margin": NONINFERIORITY_MARGIN,
                "noninferior_at_margin": effect["ci95_upper"] < NONINFERIORITY_MARGIN,
                "mean_runtime_ratio_vs_full": runtime_effect["mean_ev"],
                "ci95_runtime_ratio_lower": runtime_effect["ci95_lower"],
                "ci95_runtime_ratio_upper": runtime_effect["ci95_upper"],
                "mean_peak_memory_ratio_vs_full": memory_effect["mean_ev"],
                "ci95_peak_memory_ratio_lower": memory_effect["ci95_lower"],
                "ci95_peak_memory_ratio_upper": memory_effect["ci95_upper"],
            }
        )
    holm_adjust(inference, "two_sided_exact_sign_flip_p", "holm_adjusted_p")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", manifest)
    write_csv(output_dir / "seed_summary.csv", summaries)
    write_csv(output_dir / "checkpoint_curves.csv", curves)
    write_csv(output_dir / "paired_differences_vs_full.csv", paired)
    write_csv(output_dir / "paired_inference.csv", inference)
    write_csv(output_dir / "information_action_diagnostics.csv", information_rows)
    write_csv(output_dir / "beta_histogram.csv", beta_rows)
    write_csv(output_dir / "critic_error_subsequent_local_regret.csv", critic_rows)
    write_csv(output_dir / "diagnostic_summary_by_worker.csv", diagnostic_summaries)

    _curve_plot(curves, output_dir / "exploitability_by_nodes.png", x_key="nodes_touched")
    _curve_plot(
        curves,
        output_dir / "exploitability_by_wall_clock.png",
        x_key="wall_clock_seconds",
    )
    _final_plots(summaries, output_dir)
    _mechanism_plot(
        curves,
        output_dir / "beta_by_nodes.png",
        fields=("control_variate_beta_mean",),
        title="Experiment 22 control-variate beta",
        ylabel="Mean beta",
    )
    _mechanism_plot(
        curves,
        output_dir / "prediction_gate_by_nodes.png",
        fields=("prediction_gate_player_0", "prediction_gate_player_1"),
        title="Experiment 22 prediction-gate activation",
        ylabel="Mean prediction gate",
    )
    _mechanism_plot(
        curves,
        output_dir / "correction_magnitude_by_nodes.png",
        fields=("importance_correction_abs_mean",),
        title="Experiment 22 control-variate correction magnitude",
        ylabel="Mean absolute importance correction",
    )
    _diagnostic_plots(diagnostic_summaries, beta_rows, critic_rows, output_dir)

    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(results),
        "num_seeds": len(seeds),
        "num_information_action_rows": len(information_rows),
        "num_critic_lag_rows": len(critic_rows),
        "repository_commit": next(iter(commits)),
        "contract": contract_manifest(),
        "paired_inference": inference,
        "decision_rule": (
            "A simplified arm is eligible for selection only when the upper 95% "
            "paired confidence bound on exploitability harm is below 0.01; runtime, "
            "memory, trajectory, and mechanism diagnostics then determine preference."
        ),
        "inferential_note": (
            "Training seed is the inferential unit. With six paired seeds the minimum "
            "two-sided exact sign-flip p-value is 0.03125."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers"]
