"""Aggregate and visualise Experiment 30's exact policy-surgery audit."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (  # noqa: E402
    read_json,
    write_csv,
    write_json,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (
    summary,
)  # noqa: E402

from .config import (  # noqa: E402
    EXPERIMENT_NAME,
    RANKING_LABELS,
    RANKING_ORDER,
    SOURCE_CHECKPOINTS,
    contract_manifest,
)


COLOURS = {
    "rarest_first": "#9467bd",
    "largest_policy_error": "#ff7f0e",
    "largest_single_repair_gain": "#2ca02c",
    "rare_error_impact": "#d62728",
    "random": "#7f7f7f",
}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_workers(
    workers_root: Path,
    *,
    seeds: Sequence[int],
    smoke: bool,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    expected = {int(seed) for seed in seeds}
    results = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != EXPERIMENT_NAME:
            continue
        seed = int(result["source_seed"])
        if seed in results:
            raise ValueError(f"Duplicate Experiment 30 source seed {seed}")
        if (
            result.get("status") != "complete"
            or bool(result.get("smoke")) != bool(smoke)
        ):
            raise ValueError(f"Incomplete or mismatched audit worker {path}")
        for relative in result["artifacts"].values():
            if not (path.parent / relative).is_file():
                raise ValueError(f"Missing worker artifact {path.parent / relative}")
        results[seed] = (path, result)
    if set(results) != expected:
        raise ValueError(
            f"Audit worker seeds differ: {sorted(results)} != {sorted(expected)}"
        )
    commits = {result["repository_commit"] for _, result in results.values()}
    if len(commits) != 1:
        raise ValueError("Experiment 30 workers used different commits")

    sources, information, repairs, manifest = [], [], [], []
    for seed in sorted(results):
        path, result = results[seed]
        root = path.parent
        sources.extend(_read_csv(root / result["artifacts"]["source_summary"]))
        information.extend(
            _read_csv(root / result["artifacts"]["information_set_audit"])
        )
        repairs.extend(_read_csv(root / result["artifacts"]["repair_curves"]))
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "worker_result": str(path.resolve()),
                "source_state_hashes": " ".join(
                    row["source_sha256"] for row in result["source_states"]
                ),
            }
        )
    return sources, information, repairs, manifest


def _seed_level_repairs(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["source_seed"]),
                row["checkpoint_id"],
                row["ranking_id"],
                int(row["num_repaired"]),
            )
        ].append(row)
    result = []
    metrics = (
        "information_set_fraction_repaired",
        "reservoir_iteration_weight_fraction_repaired",
        "exact_average_weight_fraction_repaired",
        "hybrid_exploitability",
        "absolute_gap_recovered",
        "fraction_distillation_gap_recovered",
    )
    for (seed, checkpoint, ranking, count), values in sorted(grouped.items()):
        result.append(
            {
                "source_seed": seed,
                "checkpoint_id": checkpoint,
                "ranking_id": ranking,
                "ranking_label": RANKING_LABELS[ranking],
                "num_repaired": count,
                "num_information_sets": int(values[0]["num_information_sets"]),
                "n_ranking_replicates": len(values),
                **{
                    metric: float(
                        np.mean([float(row[metric]) for row in values])
                    )
                    for metric in metrics
                },
            }
        )
    return result


def _repair_summaries(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["checkpoint_id"],
                row["ranking_id"],
                int(row["num_repaired"]),
            )
        ].append(row)
    result = []
    metrics = (
        "information_set_fraction_repaired",
        "reservoir_iteration_weight_fraction_repaired",
        "exact_average_weight_fraction_repaired",
        "hybrid_exploitability",
        "absolute_gap_recovered",
        "fraction_distillation_gap_recovered",
    )
    for (checkpoint, ranking, count), values in sorted(grouped.items()):
        for metric in metrics:
            stats = summary([float(row[metric]) for row in values])
            result.append(
                {
                    "checkpoint_id": checkpoint,
                    "ranking_id": ranking,
                    "ranking_label": RANKING_LABELS[ranking],
                    "num_repaired": count,
                    "num_information_sets": int(values[0]["num_information_sets"]),
                    "metric": metric,
                    "mean": stats["mean_ev"],
                    "standard_error": stats["standard_error"],
                    "ci95_lower": stats["ci95_lower"],
                    "ci95_upper": stats["ci95_upper"],
                    "n_source_seeds": stats["n_seeds"],
                }
            )
    return result


def _safe_spearman(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    return float(spearmanr(left, right).statistic)


def _correlations(information: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in information:
        grouped[(int(row["source_seed"]), row["checkpoint_id"])].append(row)
    by_seed = []
    pairs = {
        "rarity_vs_policy_kl": ("rarity_score", "kl_exact_to_neural"),
        "rarity_vs_positive_repair_gain": (
            "rarity_score",
            "single_repair_gain_positive",
        ),
        "policy_kl_vs_positive_repair_gain": (
            "kl_exact_to_neural",
            "single_repair_gain_positive",
        ),
        "combined_score_vs_positive_repair_gain": (
            "rare_error_impact_score",
            "single_repair_gain_positive",
        ),
    }
    for (seed, checkpoint), rows in sorted(grouped.items()):
        for metric, (left, right) in pairs.items():
            by_seed.append(
                {
                    "source_seed": seed,
                    "checkpoint_id": checkpoint,
                    "metric": metric,
                    "spearman_correlation": _safe_spearman(
                        [float(row[left]) for row in rows],
                        [float(row[right]) for row in rows],
                    ),
                }
            )

    aggregate = []
    for checkpoint in SOURCE_CHECKPOINTS:
        for metric in pairs:
            values = [
                float(row["spearman_correlation"])
                for row in by_seed
                if row["checkpoint_id"] == checkpoint
                and row["metric"] == metric
                and np.isfinite(float(row["spearman_correlation"]))
            ]
            stats = summary(values)
            aggregate.append(
                {
                    "checkpoint_id": checkpoint,
                    "metric": metric,
                    "mean_spearman_correlation": stats["mean_ev"],
                    "standard_error": stats["standard_error"],
                    "ci95_lower": stats["ci95_lower"],
                    "ci95_upper": stats["ci95_upper"],
                    "n_source_seeds": stats["n_seeds"],
                }
            )
    return by_seed, aggregate


def _top_information_sets(information: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in information:
        grouped[(int(row["source_seed"]), row["checkpoint_id"])].append(row)
    result = []
    for (seed, checkpoint), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["rare_error_impact_score"]),
                -float(row["single_repair_gain"]),
            ),
        )
        for rank, row in enumerate(ordered[:20], start=1):
            result.append({**row, "combined_rank": rank})
    return result


def _summary_lookup(
    summaries: Sequence[Mapping],
    *,
    checkpoint: str,
    ranking: str,
    metric: str,
) -> list[dict]:
    return sorted(
        (
            row
            for row in summaries
            if row["checkpoint_id"] == checkpoint
            and row["ranking_id"] == ranking
            and row["metric"] == metric
        ),
        key=lambda row: int(row["num_repaired"]),
    )


def _plot_repair_curves(summaries: Sequence[Mapping], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    for ranking in RANKING_ORDER:
        exploit = _summary_lookup(
            summaries,
            checkpoint="time_36h",
            ranking=ranking,
            metric="hybrid_exploitability",
        )
        recovered = _summary_lookup(
            summaries,
            checkpoint="time_36h",
            ranking=ranking,
            metric="fraction_distillation_gap_recovered",
        )
        x = np.asarray(
            [float(row["num_repaired"]) / int(row["num_information_sets"]) for row in exploit]
        )
        axes[0].plot(
            100.0 * x,
            [float(row["mean"]) for row in exploit],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=COLOURS[ranking],
            label=RANKING_LABELS[ranking],
        )
        axes[1].plot(
            100.0 * x,
            [100.0 * float(row["mean"]) for row in recovered],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=COLOURS[ranking],
            label=RANKING_LABELS[ranking],
        )
    axes[0].set_xlabel("Information sets repaired (%)")
    axes[0].set_ylabel("Exact hybrid-policy exploitability")
    axes[1].set_xlabel("Information sets repaired (%)")
    axes[1].set_ylabel("Distillation gap recovered (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7.8)
    set_chart_title(
        axes[0],
        "Causal policy-repair exploitability at 36 hours",
        algorithm="Revised UCV",
        game_name="leduc_poker",
    )
    set_chart_title(
        axes[1],
        "Causal concentration of the distillation gap",
        algorithm="Revised UCV",
        game_name="leduc_poker",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_mass_efficiency(summaries: Sequence[Mapping], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for ranking in RANKING_ORDER:
        mass = _summary_lookup(
            summaries,
            checkpoint="time_36h",
            ranking=ranking,
            metric="reservoir_iteration_weight_fraction_repaired",
        )
        recovered = _summary_lookup(
            summaries,
            checkpoint="time_36h",
            ranking=ranking,
            metric="fraction_distillation_gap_recovered",
        )
        ax.plot(
            [100.0 * float(row["mean"]) for row in mass],
            [100.0 * float(row["mean"]) for row in recovered],
            marker="o",
            linewidth=2.0,
            markersize=3.5,
            color=COLOURS[ranking],
            label=RANKING_LABELS[ranking],
        )
    ax.set_xlabel("Reservoir iteration-weight mass repaired (%)")
    ax.set_ylabel("Distillation gap recovered (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    set_chart_title(
        ax,
        "Strategic benefit per unit of repaired reservoir mass",
        algorithm="Revised UCV",
        game_name="leduc_poker",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_rarity_scatter(information: Sequence[Mapping], output: Path) -> None:
    rows = [row for row in information if row["checkpoint_id"] == "time_36h"]
    x = np.asarray(
        [
            max(float(row["reservoir_iteration_weight_fraction"]), 1e-12)
            for row in rows
        ]
    )
    y = np.asarray([float(row["single_repair_gain"]) for row in rows])
    colour = np.asarray([float(row["kl_exact_to_neural"]) for row in rows])
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    points = ax.scatter(
        x,
        y,
        c=colour,
        cmap="viridis",
        alpha=0.55,
        s=18,
        edgecolors="none",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set_xlabel("Reservoir iteration-weight mass")
    ax.set_ylabel("Exploitability reduction from one-state repair")
    colour_bar = fig.colorbar(points, ax=ax)
    colour_bar.set_label("Exact-to-neural policy KL")
    ax.grid(alpha=0.20)
    set_chart_title(
        ax,
        "Rarity, policy error and exact strategic consequence",
        algorithm="Revised UCV",
        game_name="leduc_poker",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def aggregate_workers(
    *,
    workers_root: Path,
    seeds: Sequence[int],
    output_dir: Path,
    smoke: bool,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, information, repairs, manifest = _load_workers(
        workers_root, seeds=seeds, smoke=smoke
    )
    seed_repairs = _seed_level_repairs(repairs)
    repair_summaries = _repair_summaries(seed_repairs)
    correlation_rows, correlation_summaries = _correlations(information)
    top_rows = _top_information_sets(information)

    write_csv(output_dir / "source_summary.csv", sources)
    write_csv(output_dir / "information_set_audit.csv", information)
    write_csv(output_dir / "repair_curves.csv", repairs)
    write_csv(output_dir / "seed_level_repair_curves.csv", seed_repairs)
    write_csv(output_dir / "repair_curve_summary.csv", repair_summaries)
    write_csv(output_dir / "correlations_by_seed.csv", correlation_rows)
    write_csv(output_dir / "correlation_summary.csv", correlation_summaries)
    write_csv(output_dir / "top_rare_important_information_sets.csv", top_rows)
    write_csv(output_dir / "worker_manifest.csv", manifest)
    _plot_repair_curves(
        repair_summaries, output_dir / "causal_repair_curves.png"
    )
    _plot_mass_efficiency(
        repair_summaries, output_dir / "repair_mass_efficiency.png"
    )
    _plot_rarity_scatter(
        information, output_dir / "rarity_error_consequence.png"
    )

    primary = [
        row
        for row in repair_summaries
        if row["checkpoint_id"] == "time_36h"
        and row["metric"] == "fraction_distillation_gap_recovered"
        and row["ranking_id"] in {"rarest_first", "rare_error_impact", "random"}
        and np.isclose(
            int(row["num_repaired"]) / int(row["num_information_sets"]),
            0.10,
            atol=0.02,
        )
    ]
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_source_seeds": len(seeds),
        "num_source_checkpoints": len(SOURCE_CHECKPOINTS),
        "num_information_set_rows": len(information),
        "num_repair_rows": len(repairs),
        "contract": contract_manifest(),
        "ten_percent_repair_summary": primary,
        "outputs": {
            "source_summary": "source_summary.csv",
            "information_set_audit": "information_set_audit.csv",
            "repair_curves": "repair_curves.csv",
            "seed_level_repair_curves": "seed_level_repair_curves.csv",
            "repair_curve_summary": "repair_curve_summary.csv",
            "correlations_by_seed": "correlations_by_seed.csv",
            "correlation_summary": "correlation_summary.csv",
            "top_information_sets": "top_rare_important_information_sets.csv",
            "worker_manifest": "worker_manifest.csv",
            "causal_repair_curves": "causal_repair_curves.png",
            "repair_mass_efficiency": "repair_mass_efficiency.png",
            "rarity_error_consequence": "rarity_error_consequence.png",
        },
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = ["aggregate_workers"]
