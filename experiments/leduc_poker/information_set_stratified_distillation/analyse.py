"""Aggregate Experiment 28's paired information-set sampling results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

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
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402, E501
    summary,
)

from .config import (  # noqa: E402
    ARMS,
    ARM_ORDER,
    EMPIRICAL_INFORMATION_SET,
    EXPERIMENT_NAME,
    SOURCE_CHECKPOINTS,
    contract_manifest,
)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_workers(workers_root: Path, *, seeds: Sequence[int], smoke: bool):
    expected = {int(seed) for seed in seeds}
    results = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != EXPERIMENT_NAME:
            continue
        seed = int(result["source_seed"])
        if seed in results:
            raise ValueError(f"Duplicate Experiment 28 source seed {seed}")
        if result.get("status") != "complete" or bool(result.get("smoke")) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker {path}")
        for policy in result["policies"]:
            policy_path = path.parent / policy["relative_path"]
            if not policy_path.is_file() or sha256(policy_path) != policy["sha256"]:
                raise ValueError(f"Missing or corrupt policy {policy_path}")
        results[seed] = (path, result)
    if set(results) != expected:
        raise ValueError(f"Worker seeds differ: {sorted(results)} != {sorted(expected)}")
    commits = {result["repository_commit"] for _, result in results.values()}
    if len(commits) != 1:
        raise ValueError("Experiment 28 workers used different commits")

    metrics, decomposition, information, sampling = [], [], [], []
    manifest, policies = [], []
    for seed in sorted(results):
        path, result = results[seed]
        root = path.parent
        metrics.extend(_read_csv(root / result["artifacts"]["fit_metrics"]))
        decomposition.extend(
            _read_csv(root / result["artifacts"]["source_decomposition"])
        )
        information.extend(
            _read_csv(root / result["artifacts"]["information_set_errors"])
        )
        sampling.extend(
            _read_csv(root / result["artifacts"]["sampling_diagnostics"])
        )
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "worker_result": str(path.resolve()),
            }
        )
        policies.extend(
            {**row, "path": str((root / row["relative_path"]).resolve())}
            for row in result["policies"]
        )
    return metrics, decomposition, information, sampling, manifest, policies


def _seed_level(metrics: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in metrics:
        grouped[
            (int(row["source_seed"]), row["checkpoint_id"], row["arm_id"])
        ].append(row)
    result = []
    for (seed, checkpoint, arm), rows in sorted(grouped.items()):
        result.append(
            {
                "source_seed": seed,
                "checkpoint_id": checkpoint,
                "arm_id": arm,
                "arm_label": ARMS[arm]["label"],
                "n_fit_replicates": len(rows),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in (
                        "exploitability",
                        "distillation_gap",
                        "mean_l1",
                        "max_l1",
                        "reach_weighted_l1",
                        "mean_kl",
                        "reach_weighted_kl",
                        "fit_seconds",
                    )
                },
            }
        )
    return result


def _summaries(seed_rows: Sequence[Mapping]) -> list[dict]:
    result = []
    for checkpoint in SOURCE_CHECKPOINTS:
        for arm in ARM_ORDER:
            rows = [
                row
                for row in seed_rows
                if row["checkpoint_id"] == checkpoint and row["arm_id"] == arm
            ]
            for metric in (
                "exploitability",
                "distillation_gap",
                "reach_weighted_l1",
                "reach_weighted_kl",
                "fit_seconds",
            ):
                stats = summary([float(row[metric]) for row in rows])
                result.append(
                    {
                        "checkpoint_id": checkpoint,
                        "arm_id": arm,
                        "arm_label": ARMS[arm]["label"],
                        "metric": metric,
                        "mean": stats["mean_ev"],
                        "standard_error": stats["standard_error"],
                        "ci95_lower": stats["ci95_lower"],
                        "ci95_upper": stats["ci95_upper"],
                        "n_source_seeds": stats["n_seeds"],
                    }
                )
    return result


def _paired(seed_rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    indexed = {
        (int(row["source_seed"]), row["checkpoint_id"], row["arm_id"]): row
        for row in seed_rows
    }
    seeds = sorted({int(row["source_seed"]) for row in seed_rows})
    effects, summaries = [], []
    for checkpoint in SOURCE_CHECKPOINTS:
        for arm in ARM_ORDER:
            if arm == EMPIRICAL_INFORMATION_SET:
                continue
            for metric in (
                "exploitability",
                "distillation_gap",
                "reach_weighted_l1",
                "reach_weighted_kl",
                "fit_seconds",
            ):
                values = []
                for seed in seeds:
                    candidate = float(indexed[(seed, checkpoint, arm)][metric])
                    control = float(
                        indexed[(seed, checkpoint, EMPIRICAL_INFORMATION_SET)][metric]
                    )
                    value = candidate - control
                    values.append(value)
                    effects.append(
                        {
                            "source_seed": seed,
                            "checkpoint_id": checkpoint,
                            "candidate_arm_id": arm,
                            "candidate_arm_label": ARMS[arm]["label"],
                            "metric": metric,
                            "candidate_minus_control": value,
                        }
                    )
                stats = summary(values)
                summaries.append(
                    {
                        "checkpoint_id": checkpoint,
                        "candidate_arm_id": arm,
                        "candidate_arm_label": ARMS[arm]["label"],
                        "metric": metric,
                        "mean_candidate_minus_control": stats["mean_ev"],
                        "ci95_lower": stats["ci95_lower"],
                        "ci95_upper": stats["ci95_upper"],
                        "positive_seed_fraction": stats["positive_seed_fraction"],
                        "two_sided_exact_sign_flip_p": stats[
                            "two_sided_exact_sign_flip_p"
                        ],
                        "n_source_seeds": stats["n_seeds"],
                    }
                )
    return effects, summaries


def _plot_decomposition(decomposition: Sequence[Mapping], output: Path) -> None:
    policies = (
        "exact_tabular",
        "empirical_tabular_reservoir",
        "archived_neural_policy",
    )
    labels = (
        "Exact tabular average",
        "Empirical reservoir table",
        "Archived neural policy",
    )
    rows = [row for row in decomposition if row["checkpoint_id"] == "time_36h"]
    means, errors = [], []
    for policy in policies:
        values = [
            float(row["exploitability"]) for row in rows if row["policy"] == policy
        ]
        means.append(float(np.mean(values)))
        errors.append(
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.bar(
        np.arange(3),
        means,
        yerr=errors,
        capsize=4,
        color=("#2ca02c", "#ff7f0e", "#d62728"),
    )
    ax.set_xticks(np.arange(3), labels, rotation=15, ha="right")
    ax.set_ylabel("Exact exploitability")
    set_chart_title(ax, "Frozen average-policy decomposition at 36 hours")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_arms(seed_rows: Sequence[Mapping], output: Path) -> None:
    rows = [row for row in seed_rows if row["checkpoint_id"] == "time_36h"]
    means, errors = [], []
    for arm in ARM_ORDER:
        stats = summary(
            [float(row["distillation_gap"]) for row in rows if row["arm_id"] == arm]
        )
        means.append(stats["mean_ev"])
        errors.append(stats["standard_error"])
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    ax.bar(
        np.arange(len(ARM_ORDER)),
        means,
        yerr=errors,
        capsize=4,
        color=("#7f7f7f", "#1f77b4", "#2ca02c"),
    )
    ax.set_xticks(
        np.arange(len(ARM_ORDER)),
        [ARMS[arm]["label"] for arm in ARM_ORDER],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Neural minus exact exploitability")
    set_chart_title(ax, "Distillation gap by information-set sampling method")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_checkpoints(seed_rows: Sequence[Mapping], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(SOURCE_CHECKPOINTS))
    for arm in ARM_ORDER:
        means, errors = [], []
        for checkpoint in SOURCE_CHECKPOINTS:
            stats = summary(
                [
                    float(row["exploitability"])
                    for row in seed_rows
                    if row["checkpoint_id"] == checkpoint and row["arm_id"] == arm
                ]
            )
            means.append(stats["mean_ev"])
            errors.append(stats["standard_error"])
        ax.errorbar(
            x,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            label=ARMS[arm]["label"],
        )
    ax.set_xticks(x, ("24 hours", "36 hours"))
    ax.set_ylabel("Exact exploitability")
    set_chart_title(ax, "Distilled-policy exploitability by source checkpoint")
    ax.legend()
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
    (
        metrics,
        decomposition,
        information,
        sampling,
        manifest,
        policies,
    ) = _load_workers(workers_root, seeds=seeds, smoke=smoke)
    seed_rows = _seed_level(metrics)
    summaries = _summaries(seed_rows)
    paired_rows, paired_summary = _paired(seed_rows)
    write_csv(output_dir / "fit_metrics.csv", metrics)
    write_csv(output_dir / "source_decomposition.csv", decomposition)
    write_csv(output_dir / "information_set_errors.csv", information)
    write_csv(output_dir / "sampling_diagnostics.csv", sampling)
    write_csv(output_dir / "seed_level_metrics.csv", seed_rows)
    write_csv(output_dir / "metric_summary.csv", summaries)
    write_csv(output_dir / "paired_effects_by_seed.csv", paired_rows)
    write_csv(output_dir / "paired_effect_summary.csv", paired_summary)
    write_csv(output_dir / "worker_manifest.csv", manifest)
    write_csv(output_dir / "policy_inventory.csv", policies)
    _plot_decomposition(
        decomposition,
        output_dir / "frozen_policy_gap_decomposition.png",
    )
    _plot_arms(seed_rows, output_dir / "sampling_arm_comparison.png")
    _plot_checkpoints(seed_rows, output_dir / "checkpoint_comparison.png")
    primary = [
        row
        for row in paired_summary
        if row["checkpoint_id"] == "time_36h" and row["metric"] == "distillation_gap"
    ]
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "contract": contract_manifest(),
        "num_workers": len(manifest),
        "num_fit_rows": len(metrics),
        "num_information_set_rows": len(information),
        "primary_paired_effects": primary,
        "paired_effect_summary": paired_summary,
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(
        output_dir / "aggregate_manifest.json",
        {
            "smoke": bool(smoke),
            "seeds": list(seeds),
            "repository_commits": sorted(
                {row["repository_commit"] for row in manifest}
            ),
        },
    )
    return result


__all__ = ["aggregate_workers"]
