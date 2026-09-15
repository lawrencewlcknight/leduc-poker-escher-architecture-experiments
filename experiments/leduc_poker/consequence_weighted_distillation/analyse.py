"""Aggregate Experiment 33 proxy validation and redistillation evidence."""

from __future__ import annotations

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
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    summary,
)

from .common import read_csv  # noqa: E402
from .config import (  # noqa: E402
    ARM_LABELS,
    ARM_ORDER,
    BASELINE,
    EXPERIMENT_NAME,
    PROXY_LABELS,
    PROXY_ORDER,
    SOURCE_CHECKPOINTS,
    contract_manifest,
)
from .proxy import select_proxy  # noqa: E402


def _load_stage(root: Path, stage: str, seeds: Sequence[int], smoke: bool):
    expected = {int(seed) for seed in seeds}
    found = {}
    for path in Path(root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != EXPERIMENT_NAME or result.get("stage") != stage:
            continue
        seed = int(result["source_seed"])
        if seed in found:
            raise ValueError(f"Duplicate {stage} worker for seed {seed}")
        if result.get("status") != "complete" or bool(result.get("smoke")) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker {path}")
        found[seed] = (path, result)
    if set(found) != expected:
        raise ValueError(f"{stage} seeds differ: {sorted(found)} != {sorted(expected)}")
    commits = {result["repository_commit"] for _, result in found.values()}
    if len(commits) != 1:
        raise ValueError(f"{stage} workers used different commits")
    return found


def aggregate_proxy_workers(
    *, proxy_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    workers = _load_stage(proxy_root, "proxy", seeds, smoke)
    proxy_rows, metric_rows, manifest = [], [], []
    for seed in sorted(workers):
        path, result = workers[seed]
        root = path.parent
        proxy_rows.extend(read_csv(root / result["artifacts"]["proxy_information_sets"]))
        metric_rows.extend(read_csv(root / result["artifacts"]["proxy_repair_metrics"]))
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "worker_result": str(path.resolve()),
            }
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "proxy_information_sets.csv", proxy_rows)
    write_csv(output_dir / "proxy_repair_metrics.csv", metric_rows)
    write_csv(output_dir / "proxy_worker_manifest.csv", manifest)
    selection = select_proxy(metric_rows, output_dir)
    return selection


def _seed_level(metrics: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(int(row["source_seed"]), row["checkpoint_id"], row["arm_id"])].append(row)
    result = []
    for (seed, checkpoint, arm), rows in sorted(grouped.items()):
        result.append(
            {
                "source_seed": seed,
                "checkpoint_id": checkpoint,
                "arm_id": arm,
                "arm_label": ARM_LABELS[arm],
                "n_fit_replicates": len(rows),
                **{
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in (
                        "exploitability",
                        "distillation_gap",
                        "mean_l1",
                        "reach_weighted_l1",
                        "mean_kl",
                        "reach_weighted_kl",
                        "fit_seconds",
                    )
                },
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
            if arm == BASELINE:
                continue
            for metric in (
                "exploitability",
                "distillation_gap",
                "reach_weighted_kl",
                "fit_seconds",
            ):
                values = []
                for seed in seeds:
                    effect = float(indexed[(seed, checkpoint, arm)][metric]) - float(
                        indexed[(seed, checkpoint, BASELINE)][metric]
                    )
                    values.append(effect)
                    effects.append(
                        {
                            "source_seed": seed,
                            "checkpoint_id": checkpoint,
                            "candidate_arm_id": arm,
                            "candidate_arm_label": ARM_LABELS[arm],
                            "metric": metric,
                            "candidate_minus_baseline": effect,
                        }
                    )
                stats = summary(values)
                summaries.append(
                    {
                        "checkpoint_id": checkpoint,
                        "candidate_arm_id": arm,
                        "candidate_arm_label": ARM_LABELS[arm],
                        "metric": metric,
                        "mean_candidate_minus_baseline": stats["mean_ev"],
                        "standard_error": stats["standard_error"],
                        "ci95_lower": stats["ci95_lower"],
                        "ci95_upper": stats["ci95_upper"],
                        "positive_seed_fraction": stats["positive_seed_fraction"],
                        "two_sided_exact_sign_flip_p": stats["two_sided_exact_sign_flip_p"],
                        "n_source_seeds": stats["n_seeds"],
                    }
                )
    return effects, summaries


def _consequence_deciles(information: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    by_source = defaultdict(list)
    for row in information:
        by_source[
            (
                int(row["source_seed"]),
                row["checkpoint_id"],
                int(row["fit_replicate"]),
                row["arm_id"],
            )
        ].append(row)
    for key, rows in by_source.items():
        ordered = sorted(rows, key=lambda row: float(row["selected_proxy_score"]))
        for index, row in enumerate(ordered):
            decile = min(10, int(index * 10 / max(len(ordered), 1)) + 1)
            grouped[(*key, decile)].append(row)
    result = []
    for (seed, checkpoint, replicate, arm, decile), rows in sorted(grouped.items()):
        result.append(
            {
                "source_seed": seed,
                "checkpoint_id": checkpoint,
                "fit_replicate": replicate,
                "arm_id": arm,
                "arm_label": ARM_LABELS[arm],
                "consequence_decile": decile,
                "num_information_sets": len(rows),
                "mean_l1_error": float(
                    np.mean([float(row["l1_error"]) for row in rows])
                ),
                "mean_kl_exact_to_neural": float(
                    np.mean([float(row["kl_exact_to_neural"]) for row in rows])
                ),
                "mean_positive_single_repair_gain": float(
                    np.mean(
                        [float(row["single_repair_gain_positive"]) for row in rows]
                    )
                ),
            }
        )
    return result


def _plot_proxy_validation(proxy_metrics: Sequence[Mapping], output: Path) -> None:
    rows = [
        row for row in proxy_metrics
        if row["checkpoint_id"] == "time_36h" and np.isclose(float(row["repair_fraction"]), 0.10)
    ]
    means, errors = [], []
    for proxy in PROXY_ORDER:
        values = [
            float(row["fraction_distillation_gap_recovered"])
            for row in rows if row["proxy_id"] == proxy
        ]
        means.append(float(np.mean(values)))
        errors.append(
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(np.arange(len(PROXY_ORDER)), means, yerr=errors, capsize=3)
    ax.set_xticks(
        np.arange(len(PROXY_ORDER)),
        [PROXY_LABELS[p] for p in PROXY_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Fraction of distillation gap recovered")
    set_chart_title(ax, "Held-out 36-hour proxy validation at 10% repair")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_arms(seed_rows: Sequence[Mapping], output: Path) -> None:
    rows = [row for row in seed_rows if row["checkpoint_id"] == "time_36h"]
    means, errors = [], []
    for arm in ARM_ORDER:
        stats = summary([float(row["distillation_gap"]) for row in rows if row["arm_id"] == arm])
        means.append(stats["mean_ev"])
        errors.append(stats["standard_error"])
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(np.arange(len(ARM_ORDER)), means, yerr=errors, capsize=3)
    ax.set_xticks(
        np.arange(len(ARM_ORDER)),
        [ARM_LABELS[a] for a in ARM_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Neural minus exact exploitability")
    set_chart_title(ax, "Consequence-weighted distillation at 36 hours")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def aggregate_results(
    *,
    proxy_root: Path,
    distill_root: Path,
    selection_path: Path,
    seeds: Sequence[int],
    output_dir: Path,
    smoke: bool,
) -> dict:
    proxy_workers = _load_stage(proxy_root, "proxy", seeds, smoke)
    distill_workers = _load_stage(distill_root, "distill", seeds, smoke)
    selection = read_json(selection_path)
    metrics, information, sampling, decomposition, policies = [], [], [], [], []
    manifests = []
    proxy_metrics = []
    for seed in sorted(proxy_workers):
        path, result = proxy_workers[seed]
        proxy_metrics.extend(read_csv(path.parent / result["artifacts"]["proxy_repair_metrics"]))
    for seed in sorted(distill_workers):
        path, result = distill_workers[seed]
        if result["selected_proxy_id"] != selection["selected_proxy_id"]:
            raise ValueError("Distillation worker used a different selected proxy")
        if result["selection_sha256"] != sha256(selection_path):
            raise ValueError("Distillation worker selection checksum differs")
        root = path.parent
        metrics.extend(read_csv(root / result["artifacts"]["fit_metrics"]))
        information.extend(read_csv(root / result["artifacts"]["information_set_errors"]))
        sampling.extend(read_csv(root / result["artifacts"]["sampling_diagnostics"]))
        decomposition.extend(read_csv(root / result["artifacts"]["source_decomposition"]))
        for policy in result["policies"]:
            policy_path = root / policy["relative_path"]
            if not policy_path.is_file() or sha256(policy_path) != policy["sha256"]:
                raise ValueError(f"Missing or corrupt policy {policy_path}")
            policies.append({**policy, "path": str(policy_path.resolve())})
        manifests.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "worker_result": str(path.resolve()),
            }
        )
    seed_rows = _seed_level(metrics)
    paired_rows, paired_summary = _paired(seed_rows)
    decile_rows = _consequence_deciles(information)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "proxy_repair_metrics.csv", proxy_metrics)
    write_csv(output_dir / "fit_metrics.csv", metrics)
    write_csv(output_dir / "seed_level_metrics.csv", seed_rows)
    write_csv(output_dir / "paired_effects_by_seed.csv", paired_rows)
    write_csv(output_dir / "paired_effect_summary.csv", paired_summary)
    write_csv(output_dir / "information_set_errors.csv", information)
    write_csv(output_dir / "consequence_decile_summary.csv", decile_rows)
    write_csv(output_dir / "sampling_diagnostics.csv", sampling)
    write_csv(output_dir / "source_decomposition.csv", decomposition)
    write_csv(output_dir / "policy_inventory.csv", policies)
    write_csv(output_dir / "worker_manifest.csv", manifests)
    _plot_proxy_validation(proxy_metrics, output_dir / "proxy_validation_36h.png")
    _plot_arms(seed_rows, output_dir / "distillation_arm_comparison_36h.png")
    primary = [
        row for row in paired_summary
        if row["checkpoint_id"] == "time_36h" and row["metric"] == "distillation_gap"
    ]
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "contract": contract_manifest(),
        "selected_proxy": selection,
        "num_proxy_workers": len(proxy_workers),
        "num_distill_workers": len(distill_workers),
        "num_fit_rows": len(metrics),
        "primary_paired_effects": primary,
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(
        output_dir / "aggregate_manifest.json",
        {
            "smoke": bool(smoke),
            "seeds": list(seeds),
            "selected_proxy_sha256": sha256(selection_path),
            "repository_commits": sorted({row["repository_commit"] for row in manifests}),
        },
    )
    return result


__all__ = ["aggregate_proxy_workers", "aggregate_results"]
