"""Selection and aggregate analysis for Experiment 34."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.consequence_weighted_distillation.common import (  # noqa: E402
    read_csv,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (  # noqa: E402
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    summary,
)

from .config import (  # noqa: E402
    ARCHITECTURE_LABELS,
    ARCHITECTURE_ORDER,
    ARCHIVED_REFERENCE_EXPLOITABILITY,
    ARM_METADATA,
    ARM_ORDER,
    BASELINE_ARM,
    CONTRACT_BASELINE_LEARNING_RATE,
    CONTRACT_BASELINE_TRAINING_BUDGET,
    CURRENT_SHARED,
    EXPERIMENT_NAME,
    GROUPED_TARGETS,
    PROMOTION_MAX_SINGLE_SEED_DETERIORATION,
    PROMOTION_RELATIVE_GAP_REDUCTION,
    PROMOTION_REQUIRED_IMPROVED_SEEDS,
    ROW_TARGETS,
    TARGET_LABELS,
    contract_manifest,
    training_budgets,
)


def _load_workers(
    root: Path, stage: str, seeds: Sequence[int], smoke: bool
) -> dict[int, tuple[Path, dict]]:
    expected = {int(seed) for seed in seeds}
    found = {}
    for path in Path(root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != EXPERIMENT_NAME or result.get("stage") != stage:
            continue
        seed = int(result["source_seed"])
        if seed in found:
            raise ValueError(f"Duplicate {stage} worker for source seed {seed}")
        if result.get("status") != "complete" or bool(result.get("smoke")) != bool(smoke):
            raise ValueError(f"Incomplete or mode-mismatched worker {path}")
        found[seed] = (path, result)
    if set(found) != expected:
        raise ValueError(f"{stage} seeds differ: {sorted(found)} != {sorted(expected)}")
    commits = {result["repository_commit"] for _, result in found.values()}
    if len(commits) != 1:
        raise ValueError(f"{stage} workers used different repository commits")
    return found


def _seed_means(rows: Sequence[Mapping], metrics: Sequence[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["source_seed"]),
                row["arm_id"],
                float(row["learning_rate"]),
                int(row["training_budget"]),
            )
        ].append(row)
    result = []
    for (seed, arm, learning_rate, budget), values in sorted(grouped.items()):
        result.append(
            {
                "source_seed": seed,
                "arm_id": arm,
                "learning_rate": learning_rate,
                "training_budget": budget,
                "n_fit_replicates": len(values),
                **{
                    metric: float(np.mean([float(row[metric]) for row in values]))
                    for metric in metrics
                },
            }
        )
    return result


def select_configs(
    *, development_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    workers = _load_workers(development_root, "development", seeds, smoke)
    rows, manifests = [], []
    for seed in sorted(workers):
        path, result = workers[seed]
        rows.extend(read_csv(path.parent / result["artifacts"]["metrics"]))
        manifests.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                **result["dataset"],
                **result["objective_equivalence"],
                "worker_result": str(path.resolve()),
            }
        )
    seed_rows = _seed_means(
        rows,
        ("exploitability", "distillation_gap", "fit_seconds", "reservoir_evaluation_loss"),
    )
    config_rows = []
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["arm_id"], row["learning_rate"], row["training_budget"])].append(row)
    for (arm, learning_rate, budget), values in sorted(grouped.items()):
        exploits = [float(row["exploitability"]) for row in values]
        config_rows.append(
            {
                "arm_id": arm,
                "arm_label": ARM_METADATA[arm]["arm_label"],
                "architecture_id": ARM_METADATA[arm]["architecture_id"],
                "target_id": ARM_METADATA[arm]["target_id"],
                "learning_rate": learning_rate,
                "training_budget": budget,
                "mean_exploitability": float(np.mean(exploits)),
                "standard_error_exploitability": (
                    float(np.std(exploits, ddof=1) / np.sqrt(len(exploits)))
                    if len(exploits) > 1
                    else 0.0
                ),
                "worst_source_seed_exploitability": float(np.max(exploits)),
                "mean_distillation_gap": float(
                    np.mean([float(row["distillation_gap"]) for row in values])
                ),
                "mean_fit_seconds": float(
                    np.mean([float(row["fit_seconds"]) for row in values])
                ),
                "n_source_seeds": len(values),
            }
        )

    selected = {}
    baseline_budget = (
        training_budgets(smoke=True)[0]
        if smoke
        else CONTRACT_BASELINE_TRAINING_BUDGET
    )
    for arm in ARM_ORDER:
        candidates = [row for row in config_rows if row["arm_id"] == arm]
        if arm == BASELINE_ARM:
            candidates = [
                row
                for row in candidates
                if np.isclose(
                    float(row["learning_rate"]), CONTRACT_BASELINE_LEARNING_RATE
                )
                and int(row["training_budget"]) == int(baseline_budget)
            ]
        if not candidates:
            raise ValueError(f"No selectable development configuration for {arm}")
        chosen = min(
            candidates,
            key=lambda row: (
                float(row["mean_exploitability"]),
                float(row["worst_source_seed_exploitability"]),
                int(row["training_budget"]),
                float(row["learning_rate"]),
            ),
        )
        selected[arm] = {
            "learning_rate": float(chosen["learning_rate"]),
            "training_budget": int(chosen["training_budget"]),
            "development_mean_exploitability": float(chosen["mean_exploitability"]),
            "development_worst_source_seed_exploitability": float(
                chosen["worst_source_seed_exploitability"]
            ),
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "development_fit_metrics.csv", rows)
    write_csv(output_dir / "development_seed_level_metrics.csv", seed_rows)
    write_csv(output_dir / "development_config_summary.csv", config_rows)
    write_csv(output_dir / "development_worker_manifest.csv", manifests)
    selection = {
        "status": "complete",
        "smoke": bool(smoke),
        "selected_configs": selected,
        "selection_rule": contract_manifest()["selection_rule"],
        "baseline_rule": contract_manifest()["baseline_rule"],
        "source_seeds": list(seeds),
        "repository_commits": sorted({row["repository_commit"] for row in manifests}),
    }
    write_json(output_dir / "selected_configs.json", selection)
    return selection


def _validation_seed_rows(rows: Sequence[Mapping]) -> list[dict]:
    empirical = [row for row in rows if row["teacher_type"] == "empirical_reservoir"]
    return _seed_means(
        empirical,
        (
            "exploitability",
            "distillation_gap",
            "fit_seconds",
            "mean_l1",
            "reach_weighted_l1",
            "mean_kl",
            "reach_weighted_kl",
            "reservoir_evaluation_loss",
        ),
    )


def _paired(seed_rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    indexed = {(int(row["source_seed"]), row["arm_id"]): row for row in seed_rows}
    seeds = sorted({int(row["source_seed"]) for row in seed_rows})
    effects, summaries = [], []
    for arm in ARM_ORDER:
        if arm == BASELINE_ARM:
            continue
        for metric in ("exploitability", "distillation_gap", "fit_seconds"):
            values = []
            for seed in seeds:
                effect = float(indexed[(seed, arm)][metric]) - float(
                    indexed[(seed, BASELINE_ARM)][metric]
                )
                values.append(effect)
                effects.append(
                    {
                        "source_seed": seed,
                        "candidate_arm_id": arm,
                        "candidate_arm_label": ARM_METADATA[arm]["arm_label"],
                        "metric": metric,
                        "candidate_minus_baseline": effect,
                    }
                )
            stats = summary(values)
            summaries.append(
                {
                    "candidate_arm_id": arm,
                    "candidate_arm_label": ARM_METADATA[arm]["arm_label"],
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


def _factorial_contrasts(seed_rows: Sequence[Mapping]) -> list[dict]:
    indexed = {(int(row["source_seed"]), row["arm_id"]): row for row in seed_rows}
    seeds = sorted({int(row["source_seed"]) for row in seed_rows})
    rows = []
    for architecture_id in ARCHITECTURE_ORDER:
        grouped_arm = next(
            arm for arm in ARM_ORDER
            if ARM_METADATA[arm]["architecture_id"] == architecture_id
            and ARM_METADATA[arm]["target_id"] == GROUPED_TARGETS
        )
        row_arm = next(
            arm for arm in ARM_ORDER
            if ARM_METADATA[arm]["architecture_id"] == architecture_id
            and ARM_METADATA[arm]["target_id"] == ROW_TARGETS
        )
        for seed in seeds:
            rows.append(
                {
                    "source_seed": seed,
                    "contrast_id": f"aggregation_within_{architecture_id}",
                    "contrast_label": f"Grouped minus rows: {ARCHITECTURE_LABELS[architecture_id]}",
                    "exploitability_effect": float(indexed[(seed, grouped_arm)]["exploitability"])
                    - float(indexed[(seed, row_arm)]["exploitability"]),
                }
            )
    for target_id in (ROW_TARGETS, GROUPED_TARGETS):
        current_arm = next(
            arm for arm in ARM_ORDER
            if ARM_METADATA[arm]["architecture_id"] == CURRENT_SHARED
            and ARM_METADATA[arm]["target_id"] == target_id
        )
        for architecture_id in ARCHITECTURE_ORDER[1:]:
            candidate_arm = next(
                arm for arm in ARM_ORDER
                if ARM_METADATA[arm]["architecture_id"] == architecture_id
                and ARM_METADATA[arm]["target_id"] == target_id
            )
            for seed in seeds:
                rows.append(
                    {
                        "source_seed": seed,
                        "contrast_id": f"{architecture_id}_vs_current_with_{target_id}",
                        "contrast_label": (
                            f"{ARCHITECTURE_LABELS[architecture_id]} minus current, "
                            f"{TARGET_LABELS[target_id]}"
                        ),
                        "exploitability_effect": float(
                            indexed[(seed, candidate_arm)]["exploitability"]
                        )
                        - float(indexed[(seed, current_arm)]["exploitability"]),
                    }
                )
    return rows


def _promotion(seed_rows: Sequence[Mapping]) -> list[dict]:
    indexed = {(int(row["source_seed"]), row["arm_id"]): row for row in seed_rows}
    seeds = sorted({int(row["source_seed"]) for row in seed_rows})
    baseline_gap = float(
        np.mean([float(indexed[(seed, BASELINE_ARM)]["distillation_gap"]) for seed in seeds])
    )
    rows = []
    for arm in ARM_ORDER:
        if arm == BASELINE_ARM:
            continue
        effects = [
            float(indexed[(seed, arm)]["exploitability"])
            - float(indexed[(seed, BASELINE_ARM)]["exploitability"])
            for seed in seeds
        ]
        mean_gap = float(
            np.mean([float(indexed[(seed, arm)]["distillation_gap"]) for seed in seeds])
        )
        mean_exploitability = float(
            np.mean([float(indexed[(seed, arm)]["exploitability"]) for seed in seeds])
        )
        relative_reduction = (
            (baseline_gap - mean_gap) / baseline_gap if baseline_gap > 0.0 else 0.0
        )
        improved = int(sum(effect < 0.0 for effect in effects))
        passes = (
            relative_reduction >= PROMOTION_RELATIVE_GAP_REDUCTION
            and improved >= min(PROMOTION_REQUIRED_IMPROVED_SEEDS, len(seeds))
            and max(effects) <= PROMOTION_MAX_SINGLE_SEED_DETERIORATION
            and mean_exploitability < ARCHIVED_REFERENCE_EXPLOITABILITY
        )
        rows.append(
            {
                "arm_id": arm,
                "arm_label": ARM_METADATA[arm]["arm_label"],
                "mean_exploitability": mean_exploitability,
                "mean_distillation_gap": mean_gap,
                "relative_gap_reduction_vs_baseline": relative_reduction,
                "improved_source_seeds": improved,
                "maximum_single_seed_deterioration": max(effects),
                "beats_archived_experiment_29_mean": int(
                    mean_exploitability < ARCHIVED_REFERENCE_EXPLOITABILITY
                ),
                "passes_promotion_rule": int(passes),
            }
        )
    return rows


def _optimisation_variability(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["teacher_type"] != "empirical_reservoir":
            continue
        grouped[(int(row["source_seed"]), row["arm_id"])].append(
            float(row["exploitability"])
        )
    result = []
    for (seed, arm), values in sorted(grouped.items()):
        result.append(
            {
                "source_seed": seed,
                "arm_id": arm,
                "arm_label": ARM_METADATA[arm]["arm_label"],
                "n_fit_replicates": len(values),
                "mean_exploitability": float(np.mean(values)),
                "standard_deviation_across_fits": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
                "range_across_fits": float(np.max(values) - np.min(values)),
            }
        )
    return result


def _gradient_summary(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm_id"], int(row["left_group"]), int(row["right_group"]))].append(
            float(row["cosine_similarity"])
        )
    result = []
    for (arm, left, right), values in sorted(grouped.items()):
        result.append(
            {
                "arm_id": arm,
                "arm_label": ARM_METADATA[arm]["arm_label"],
                "left_group": left,
                "right_group": right,
                "mean_cosine_similarity": float(np.mean(values)),
                "negative_fraction": float(np.mean(np.asarray(values) < 0.0)),
                "num_measurements": len(values),
            }
        )
    return result


def _ensemble_effects(ensembles: Sequence[Mapping], seed_rows: Sequence[Mapping]) -> list[dict]:
    singles = {(int(row["source_seed"]), row["arm_id"]): row for row in seed_rows}
    result = []
    for row in ensembles:
        arm = row["arm_id"]
        if arm not in ARM_METADATA:
            continue
        seed = int(row["source_seed"])
        single = float(singles[(seed, arm)]["exploitability"])
        ensemble = float(row["exploitability"])
        result.append(
            {
                "source_seed": seed,
                "arm_id": arm,
                "arm_label": ARM_METADATA[arm]["arm_label"],
                "mean_single_fit_exploitability": single,
                "ensemble_exploitability": ensemble,
                "ensemble_minus_mean_single": ensemble - single,
            }
        )
    return result


def _plot_factorial(seed_rows: Sequence[Mapping], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(ARCHITECTURE_ORDER))
    width = 0.36
    for target_index, target_id in enumerate((ROW_TARGETS, GROUPED_TARGETS)):
        means, errors = [], []
        for architecture_id in ARCHITECTURE_ORDER:
            arm = next(
                candidate for candidate in ARM_ORDER
                if ARM_METADATA[candidate]["architecture_id"] == architecture_id
                and ARM_METADATA[candidate]["target_id"] == target_id
            )
            values = [
                float(row["exploitability"])
                for row in seed_rows if row["arm_id"] == arm
            ]
            means.append(float(np.mean(values)))
            errors.append(
                float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else 0.0
            )
        offset = (target_index - 0.5) * width
        ax.bar(x + offset, means, width, yerr=errors, capsize=3, label=TARGET_LABELS[target_id])
    ax.axhline(
        ARCHIVED_REFERENCE_EXPLOITABILITY,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Archived Experiment 29 mean",
    )
    ax.set_xticks(x, [ARCHITECTURE_LABELS[value] for value in ARCHITECTURE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.legend()
    set_chart_title(ax, "Structure-aware average-policy distillation at 36 hours")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_exact_teacher(rows: Sequence[Mapping], output: Path) -> None:
    diagnostic = [row for row in rows if row["teacher_type"] == "exact_tabular_diagnostic"]
    means, errors = [], []
    for architecture_id in ARCHITECTURE_ORDER:
        values = [float(row["exploitability"]) for row in diagnostic if row["architecture_id"] == architecture_id]
        means.append(float(np.mean(values)))
        errors.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(np.arange(len(means)), means, yerr=errors, capsize=3)
    ax.set_xticks(np.arange(len(means)), [ARCHITECTURE_LABELS[a] for a in ARCHITECTURE_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    set_chart_title(ax, "Exact-tabular teacher capacity diagnostic")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_variability(rows: Sequence[Mapping], output: Path) -> None:
    empirical = [row for row in rows if row["teacher_type"] == "empirical_reservoir"]
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [
        [float(row["exploitability"]) for row in empirical if row["arm_id"] == arm]
        for arm in ARM_ORDER
    ]
    ax.boxplot(data, tick_labels=[ARM_METADATA[arm]["arm_label"] for arm in ARM_ORDER])
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    set_chart_title(ax, "Source-seed and fitting variability at 36 hours")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def aggregate_results(
    *,
    development_root: Path,
    validation_root: Path,
    selection_path: Path,
    seeds: Sequence[int],
    output_dir: Path,
    smoke: bool,
) -> dict:
    development = _load_workers(development_root, "development", seeds, smoke)
    validation = _load_workers(validation_root, "validation", seeds, smoke)
    selection = read_json(selection_path)
    metrics, gradients, groups, errors, ensembles, decomposition, policies = [], [], [], [], [], [], []
    manifests = []
    selection_digest = sha256(selection_path)
    for seed in sorted(validation):
        path, result = validation[seed]
        if result["selection_sha256"] != selection_digest:
            raise ValueError(f"Validation worker {seed} used a different selection")
        root = path.parent
        metrics.extend(read_csv(root / result["artifacts"]["metrics"]))
        gradients.extend(read_csv(root / result["artifacts"]["gradient_cosines"]))
        groups.extend(read_csv(root / result["artifacts"]["group_metrics"]))
        errors.extend(read_csv(root / result["artifacts"]["information_set_errors"]))
        ensembles.extend(read_csv(root / result["artifacts"]["ensemble_metrics"]))
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
    seed_rows = _validation_seed_rows(metrics)
    paired_rows, paired_summary = _paired(seed_rows)
    contrasts = _factorial_contrasts(seed_rows)
    promotion = _promotion(seed_rows)
    variability = _optimisation_variability(metrics)
    gradient_summary = _gradient_summary(gradients)
    ensemble_effects = _ensemble_effects(ensembles, seed_rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "validation_fit_metrics.csv", metrics)
    write_csv(output_dir / "seed_level_metrics.csv", seed_rows)
    write_csv(output_dir / "paired_effects_by_seed.csv", paired_rows)
    write_csv(output_dir / "paired_effect_summary.csv", paired_summary)
    write_csv(output_dir / "factorial_contrasts_by_seed.csv", contrasts)
    write_csv(output_dir / "promotion_assessment.csv", promotion)
    write_csv(output_dir / "optimisation_variability.csv", variability)
    write_csv(output_dir / "gradient_cosines.csv", gradients)
    write_csv(output_dir / "gradient_conflict_summary.csv", gradient_summary)
    write_csv(output_dir / "group_metrics.csv", groups)
    write_csv(output_dir / "information_set_errors.csv", errors)
    write_csv(output_dir / "ensemble_metrics.csv", ensembles)
    write_csv(output_dir / "ensemble_effects.csv", ensemble_effects)
    write_csv(output_dir / "source_decomposition.csv", decomposition)
    write_csv(output_dir / "policy_inventory.csv", policies)
    write_csv(output_dir / "worker_manifest.csv", manifests)
    _plot_factorial(seed_rows, output_dir / "factorial_exploitability_36h.png")
    _plot_exact_teacher(metrics, output_dir / "exact_teacher_capacity_36h.png")
    _plot_variability(metrics, output_dir / "fitting_variability_36h.png")
    passing = [row for row in promotion if int(row["passes_promotion_rule"])]
    best = min(promotion, key=lambda row: float(row["mean_exploitability"]))
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "contract": contract_manifest(),
        "selected_configs": selection["selected_configs"],
        "num_development_workers": len(development),
        "num_validation_workers": len(validation),
        "num_validation_fit_rows": len(metrics),
        "best_descriptive_candidate": best,
        "promotion_candidates": passing,
        "fresh_36h_confirmation_recommended": bool(passing),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(
        output_dir / "aggregate_manifest.json",
        {
            "smoke": bool(smoke),
            "seeds": list(seeds),
            "selection_sha256": selection_digest,
            "repository_commits": sorted({row["repository_commit"] for row in manifests}),
        },
    )
    return result


__all__ = ["aggregate_results", "select_configs"]
