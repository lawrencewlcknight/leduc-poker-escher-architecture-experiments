"""Tests for staged factorial selection and node-matched analysis."""

from __future__ import annotations

import pytest

from escher_poker.factorial_metrics import (
    factorial_effect_rows,
    metric_at_target_nodes,
    rank_screening_variants,
)


def test_node_matched_metric_interpolates_and_ignores_final_policy_row():
    result = metric_at_target_nodes(
        [
            {"nodes_touched": 0.0, "exploitability": 1.0},
            {"nodes_touched": 2_000_000.0, "exploitability": 0.0},
            {
                "nodes_touched": 2_000_000.0,
                "exploitability": 99.0,
                "is_final_policy_evaluation": True,
            },
        ],
        1_000_000.0,
    )
    assert result["value"] == pytest.approx(0.5)
    assert result["evaluation_nodes"] == pytest.approx(1_000_000.0)
    assert result["target_nodes_reached"] is True
    assert result["node_gap"] == pytest.approx(0.0)


def test_node_matched_metric_uses_nearest_endpoint_without_extrapolation():
    result = metric_at_target_nodes(
        [{"nodes_touched": 940_000.0, "exploitability": 0.7}],
        1_000_000.0,
    )
    assert result["value"] == pytest.approx(0.7)
    assert result["evaluation_nodes"] == pytest.approx(940_000.0)
    assert result["target_nodes_reached"] is False
    assert result["node_gap"] == pytest.approx(60_000.0)


def test_screening_selection_keeps_baseline_and_top_two_treatments():
    rows = []
    for seed in [1, 2, 3]:
        for variant_id, value in [
            ("baseline", 0.7),
            ("policy", 0.4),
            ("scale", 0.5),
            ("both", 0.2),
        ]:
            rows.append({
                "seed": seed,
                "variant_id": variant_id,
                "variant_label": variant_id,
                "exploitability_at_target_nodes": value,
                "meaningful_success": value < 0.3,
                "meaningful_success_threshold": 0.3,
            })
    ranking, selected = rank_screening_variants(
        rows,
        baseline_variant_id="baseline",
        confirmation_top_k=2,
    )
    assert [row["variant_id"] for row in ranking] == [
        "both",
        "policy",
        "scale",
        "baseline",
    ]
    assert selected == ["baseline", "both", "policy"]
    assert ranking[0]["screening_mean_below_success_threshold"] is True


def test_factorial_effects_are_paired_difference_in_differences():
    rows = [
        {"seed": 1, "variant_id": "baseline", "metric": 0.7},
        {"seed": 1, "variant_id": "policy", "metric": 0.5},
        {"seed": 1, "variant_id": "scale", "metric": 0.4},
        {"seed": 1, "variant_id": "both", "metric": 0.2},
    ]
    effects = factorial_effect_rows(
        rows,
        baseline_variant_id="baseline",
        policy_only_variant_id="policy",
        scale_only_variant_id="scale",
        both_variant_id="both",
        metric="metric",
    )
    assert effects[0]["policy_weighted_q_main_effect"] == pytest.approx(-0.2)
    assert effects[0]["scale_only_normalization_main_effect"] == pytest.approx(-0.3)
    assert effects[0]["correction_interaction"] == pytest.approx(0.0)
