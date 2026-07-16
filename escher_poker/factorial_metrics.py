"""Pure analysis helpers for staged, node-matched factorial experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def metric_at_target_nodes(curve_rows, target_nodes: float, metric="exploitability"):
    """Interpolate ``metric`` at a node budget without endpoint extrapolation."""
    points = []
    for row in curve_rows:
        if bool(row.get("is_final_policy_evaluation", False)):
            continue
        nodes = float(row.get("nodes_touched", np.nan))
        value = float(row.get(metric, np.nan))
        if np.isfinite(nodes) and np.isfinite(value):
            points.append((nodes, value))
    if not points:
        return {
            "value": np.nan,
            "evaluation_nodes": np.nan,
            "target_nodes_reached": False,
            "node_gap": np.nan,
        }

    by_nodes = defaultdict(list)
    for nodes, value in points:
        by_nodes[nodes].append(value)
    nodes = np.asarray(sorted(by_nodes), dtype=np.float64)
    values = np.asarray(
        [np.mean(by_nodes[node]) for node in nodes],
        dtype=np.float64,
    )
    target = float(target_nodes)
    if target <= nodes[0]:
        evaluation_nodes = float(nodes[0])
        value = float(values[0])
    elif target >= nodes[-1]:
        evaluation_nodes = float(nodes[-1])
        value = float(values[-1])
    else:
        evaluation_nodes = target
        value = float(np.interp(target, nodes, values))

    return {
        "value": value,
        "evaluation_nodes": evaluation_nodes,
        "target_nodes_reached": bool(nodes[-1] >= target),
        "node_gap": float(abs(evaluation_nodes - target)),
    }


def rank_screening_variants(
    summary_rows: Sequence[Mapping],
    *,
    baseline_variant_id: str,
    confirmation_top_k: int,
    metric: str = "exploitability_at_target_nodes",
):
    """Rank screening arms and select the top treatments plus the baseline."""
    variants = sorted({str(row["variant_id"]) for row in summary_rows})
    ranking = []
    for variant_id in variants:
        rows = [row for row in summary_rows if str(row["variant_id"]) == variant_id]
        values = np.asarray([float(row.get(metric, np.nan)) for row in rows])
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if values.size else np.nan
        standard_error = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0 if values.size == 1 else np.nan
        )
        thresholds = np.asarray([
            float(row.get("meaningful_success_threshold", np.nan))
            for row in rows
        ])
        thresholds = thresholds[np.isfinite(thresholds)]
        success_threshold = float(thresholds[0]) if thresholds.size else np.nan
        ranking.append({
            "variant_id": variant_id,
            "variant_label": rows[0].get("variant_label", variant_id) if rows else variant_id,
            "screening_seed_count": int(values.size),
            f"mean_{metric}": mean,
            f"standard_error_{metric}": standard_error,
            "screening_success_fraction": (
                float(np.mean([bool(row.get("meaningful_success", False)) for row in rows]))
                if rows
                else np.nan
            ),
            "screening_mean_below_success_threshold": bool(
                np.isfinite(mean)
                and np.isfinite(success_threshold)
                and mean < success_threshold
            ),
        })
    ranking.sort(
        key=lambda row: (
            not np.isfinite(row[f"mean_{metric}"]),
            row[f"mean_{metric}"] if np.isfinite(row[f"mean_{metric}"]) else np.inf,
        )
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    treatments = [
        row["variant_id"]
        for row in ranking
        if row["variant_id"] != baseline_variant_id
    ]
    selected = [baseline_variant_id] + treatments[:max(1, int(confirmation_top_k))]
    return ranking, selected


def factorial_effect_rows(
    summary_rows: Sequence[Mapping],
    *,
    baseline_variant_id: str,
    policy_only_variant_id: str,
    scale_only_variant_id: str,
    both_variant_id: str,
    metric: str = "exploitability_at_target_nodes",
):
    """Compute paired 2x2 main effects and interaction for complete seeds."""
    by_seed_variant = {
        (int(row["seed"]), str(row["variant_id"])): row
        for row in summary_rows
    }
    required = [
        baseline_variant_id,
        policy_only_variant_id,
        scale_only_variant_id,
        both_variant_id,
    ]
    output = []
    for seed in sorted({int(row["seed"]) for row in summary_rows}):
        if not all((seed, variant_id) in by_seed_variant for variant_id in required):
            continue
        y00 = float(by_seed_variant[(seed, baseline_variant_id)].get(metric, np.nan))
        y10 = float(by_seed_variant[(seed, policy_only_variant_id)].get(metric, np.nan))
        y01 = float(by_seed_variant[(seed, scale_only_variant_id)].get(metric, np.nan))
        y11 = float(by_seed_variant[(seed, both_variant_id)].get(metric, np.nan))
        if not np.all(np.isfinite([y00, y10, y01, y11])):
            continue
        output.append({
            "seed": seed,
            "baseline": y00,
            "policy_weighted_q_only": y10,
            "scale_only_normalization_only": y01,
            "both_corrections": y11,
            "policy_weighted_q_main_effect": ((y10 + y11) - (y00 + y01)) / 2.0,
            "scale_only_normalization_main_effect": ((y01 + y11) - (y00 + y10)) / 2.0,
            "correction_interaction": y11 - y10 - y01 + y00,
            "policy_only_delta_vs_baseline": y10 - y00,
            "scale_only_delta_vs_baseline": y01 - y00,
            "both_delta_vs_baseline": y11 - y00,
        })
    return output
