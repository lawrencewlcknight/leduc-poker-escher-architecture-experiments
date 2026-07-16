"""Exact checkpoint-stability analysis for saved ESCHER policies."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, expected_game_score

from .constants import LEDUC_GAME_VALUE_PLAYER_0
from .policy_snapshots import LoadedESCHERPolicy


def policy_metrics(game, pol) -> Dict[str, float]:
    """Compute exact exploitability and policy-value diagnostics."""
    tab_pol = policy.tabular_policy_from_callable(game, pol.action_probabilities)
    nash_conv = exploitability.nash_conv(game, tab_pol)
    policy_value = expected_game_score.policy_value(
        game.new_initial_state(),
        [tab_pol] * game.num_players(),
    )[0]
    return {
        "nash_conv": float(nash_conv),
        "exploitability": float(nash_conv / 2.0),
        "policy_value": float(policy_value),
        "policy_value_error": float(abs(policy_value - LEDUC_GAME_VALUE_PLAYER_0)),
    }


def exact_expected_value_two_player(game, pol0, pol1) -> tuple[float, float]:
    vals = expected_game_score.policy_value(game.new_initial_state(), [pol0, pol1])
    return float(vals[0]), float(vals[1])


def exact_seat_averaged_value_for_a(game, policy_a, policy_b) -> Dict[str, float]:
    """Return exact EV for policy A against policy B, averaging both seats."""
    v_as_p0 = exact_expected_value_two_player(game, policy_a, policy_b)[0]
    v_as_p1 = exact_expected_value_two_player(game, policy_b, policy_a)[1]
    return {
        "A_EV_as_player0": float(v_as_p0),
        "A_EV_as_player1": float(v_as_p1),
        "A_EV_seat_averaged": 0.5 * float(v_as_p0 + v_as_p1),
    }


def classify_ev(ev: float, eps: float) -> str:
    if ev > eps:
        return "clear_win"
    if ev < -eps:
        return "clear_loss"
    return "tie"


def _safe_sem(values: List[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size <= 1:
        return 0.0
    return float(stats.sem(finite))


def _mean(values: List[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return float(np.mean(finite)) if finite.size else np.nan


def _matrix_rows(matrix: Dict[int, Dict[int, float]], value_name: str) -> List[Dict[str, Any]]:
    rows = []
    for row_key in sorted(matrix):
        row = {"checkpoint": int(row_key)}
        for col_key in sorted(matrix[row_key]):
            row[str(col_key)] = float(matrix[row_key][col_key])
        rows.append(row)
    return rows


def _monotonicity_summary_for_seed(
    seed: int,
    matrix: Dict[int, Dict[int, float]],
    eps: float,
) -> Dict[str, Any]:
    iterations = sorted(matrix)
    later_earlier_evs = []
    adjacent_evs = []
    for idx, later in enumerate(iterations):
        for earlier in iterations[:idx]:
            later_earlier_evs.append(float(matrix[later][earlier]))
        if idx > 0:
            adjacent_evs.append(float(matrix[later][iterations[idx - 1]]))

    classes = [classify_ev(value, eps) for value in later_earlier_evs]
    adj_classes = [classify_ev(value, eps) for value in adjacent_evs]
    arr = np.asarray(later_earlier_evs, dtype=float)
    adj = np.asarray(adjacent_evs, dtype=float)
    return {
        "seed": int(seed),
        "num_later_vs_earlier_pairs": int(len(arr)),
        "all_pairs_clear_improvement_rate": (
            float(np.mean([label == "clear_win" for label in classes])) if classes else np.nan
        ),
        "all_pairs_tie_rate": (
            float(np.mean([label == "tie" for label in classes])) if classes else np.nan
        ),
        "all_pairs_clear_regression_rate": (
            float(np.mean([label == "clear_loss" for label in classes])) if classes else np.nan
        ),
        "adjacent_clear_improvement_rate": (
            float(np.mean([label == "clear_win" for label in adj_classes])) if adj_classes else np.nan
        ),
        "adjacent_tie_rate": (
            float(np.mean([label == "tie" for label in adj_classes])) if adj_classes else np.nan
        ),
        "adjacent_clear_regression_rate": (
            float(np.mean([label == "clear_loss" for label in adj_classes])) if adj_classes else np.nan
        ),
        "mean_later_vs_earlier_ev": float(np.mean(arr)) if len(arr) else np.nan,
        "worst_monotonicity_violation_ev": float(np.min(arr)) if len(arr) else np.nan,
        "best_later_vs_earlier_ev": float(np.max(arr)) if len(arr) else np.nan,
        "mean_adjacent_ev": float(np.mean(adj)) if len(adj) else np.nan,
    }


def analyze_checkpoint_snapshots(
    game,
    snapshot_rows: List[Dict[str, Any]],
    checkpoint_schedule: List[int],
    final_iteration: int,
    equivalence_epsilon: float,
) -> Dict[str, Any]:
    """Load policy snapshots and compute exact checkpoint-stability outputs."""
    checkpoint_rows = [row for row in snapshot_rows if row["arm"] == "checkpointed"]
    baseline_rows = [row for row in snapshot_rows if row["arm"] == "continuous_baseline"]

    policies_by_seed = defaultdict(dict)
    baseline_policy_by_seed = {}
    loaded_rows = []

    for row in checkpoint_rows:
        pol = LoadedESCHERPolicy(game, row["path"])
        seed = int(row["seed"])
        checkpoint = int(row["iteration"])
        policies_by_seed[seed][checkpoint] = pol
        loaded_rows.append({
            "seed": seed,
            "arm": row["arm"],
            "checkpoint": checkpoint,
            "path": row["path"],
            "nodes_visited": pol.nodes_visited,
            "policy_layers": str(pol.policy_network_layers),
            "input_size": pol.input_size,
            "num_actions": pol.num_actions,
        })

    for row in baseline_rows:
        pol = LoadedESCHERPolicy(game, row["path"])
        seed = int(row["seed"])
        baseline_policy_by_seed[seed] = pol
        loaded_rows.append({
            "seed": seed,
            "arm": row["arm"],
            "checkpoint": int(row["iteration"]),
            "path": row["path"],
            "nodes_visited": pol.nodes_visited,
            "policy_layers": str(pol.policy_network_layers),
            "input_size": pol.input_size,
            "num_actions": pol.num_actions,
        })

    metric_rows = []
    for seed, policies in policies_by_seed.items():
        for checkpoint, pol in sorted(policies.items()):
            metric_rows.append({
                "seed": int(seed),
                "arm": "checkpointed",
                "checkpoint": int(checkpoint),
                **policy_metrics(game, pol),
            })
    for seed, pol in baseline_policy_by_seed.items():
        metric_rows.append({
            "seed": int(seed),
            "arm": "continuous_baseline",
            "checkpoint": int(final_iteration),
            **policy_metrics(game, pol),
        })

    exact_records = []
    exact_matrices = {}
    for seed, policies in policies_by_seed.items():
        iterations = sorted(policies)
        matrix = {iteration: {} for iteration in iterations}
        for i in iterations:
            for j in iterations:
                ev = exact_seat_averaged_value_for_a(game, policies[i], policies[j])
                matrix[i][j] = ev["A_EV_seat_averaged"]
                exact_records.append({
                    "seed": int(seed),
                    "checkpoint_A": int(i),
                    "checkpoint_B": int(j),
                    **ev,
                })
        exact_matrices[seed] = matrix

    mean_matrix = {iteration: {} for iteration in checkpoint_schedule}
    win_fraction_matrix = {iteration: {} for iteration in checkpoint_schedule}
    for a in checkpoint_schedule:
        for b in checkpoint_schedule:
            vals = [
                row["A_EV_seat_averaged"]
                for row in exact_records
                if row["checkpoint_A"] == a and row["checkpoint_B"] == b
            ]
            mean_matrix[a][b] = _mean(vals)
            win_fraction_matrix[a][b] = (
                float(np.mean(np.asarray(vals, dtype=float) > equivalence_epsilon))
                if vals
                else np.nan
            )

    monotonicity_rows = []
    strength_rows = []
    best_rows = []
    metrics_by_seed_checkpoint = {
        (row["seed"], row["checkpoint"]): row
        for row in metric_rows
        if row["arm"] == "checkpointed"
    }

    for seed, matrix in exact_matrices.items():
        monotonicity_rows.append(
            _monotonicity_summary_for_seed(seed, matrix, equivalence_epsilon)
        )
        iterations = sorted(matrix)
        seed_strength_rows = []
        for idx, checkpoint in enumerate(iterations):
            earlier = iterations[:idx]
            later = iterations[idx + 1:]
            other = [it for it in iterations if it != checkpoint]
            row = {
                "seed": int(seed),
                "checkpoint": int(checkpoint),
                "mean_EV_vs_all_other_checkpoints": (
                    _mean([matrix[checkpoint][it] for it in other]) if other else np.nan
                ),
                "mean_EV_vs_earlier_checkpoints": (
                    _mean([matrix[checkpoint][it] for it in earlier]) if earlier else np.nan
                ),
                "mean_EV_vs_later_checkpoints": (
                    _mean([matrix[checkpoint][it] for it in later]) if later else np.nan
                ),
                "EV_vs_previous_checkpoint": (
                    float(matrix[checkpoint][iterations[idx - 1]]) if idx > 0 else np.nan
                ),
                "wins_vs_earlier": (
                    int(sum(matrix[checkpoint][it] > equivalence_epsilon for it in earlier))
                    if earlier
                    else 0
                ),
                "ties_vs_earlier": (
                    int(sum(abs(matrix[checkpoint][it]) <= equivalence_epsilon for it in earlier))
                    if earlier
                    else 0
                ),
                "losses_vs_earlier": (
                    int(sum(matrix[checkpoint][it] < -equivalence_epsilon for it in earlier))
                    if earlier
                    else 0
                ),
            }
            seed_strength_rows.append(row)
            strength_rows.append(row)

        metric_seed = [
            row
            for row in metric_rows
            if row["seed"] == seed and row["arm"] == "checkpointed"
        ]
        best_h2h = max(
            seed_strength_rows,
            key=lambda row: row["mean_EV_vs_all_other_checkpoints"],
        )
        best_expl = min(metric_seed, key=lambda row: row["exploitability"])
        final_metric = metrics_by_seed_checkpoint.get((seed, int(final_iteration)), {})
        best_rows.append({
            "seed": int(seed),
            "best_checkpoint_by_head_to_head": int(best_h2h["checkpoint"]),
            "best_checkpoint_by_head_to_head_mean_EV": float(
                best_h2h["mean_EV_vs_all_other_checkpoints"]
            ),
            "best_checkpoint_by_exploitability": int(best_expl["checkpoint"]),
            "best_checkpoint_exploitability": float(best_expl["exploitability"]),
            "final_checkpoint_exploitability": float(
                final_metric.get("exploitability", np.nan)
            ),
        })

    strength_with_metrics = []
    for row in strength_rows:
        merged = dict(row)
        merged.update(metrics_by_seed_checkpoint.get((row["seed"], row["checkpoint"]), {}))
        strength_with_metrics.append(merged)

    aggregate_strength_rows = []
    for checkpoint in checkpoint_schedule:
        rows = [row for row in strength_with_metrics if row["checkpoint"] == checkpoint]
        if not rows:
            continue
        aggregate_strength_rows.append({
            "checkpoint": int(checkpoint),
            "mean_EV_vs_earlier_mean": _mean([
                row.get("mean_EV_vs_earlier_checkpoints", np.nan) for row in rows
            ]),
            "mean_EV_vs_earlier_sem": _safe_sem([
                row.get("mean_EV_vs_earlier_checkpoints", np.nan) for row in rows
            ]),
            "EV_vs_previous_mean": _mean([
                row.get("EV_vs_previous_checkpoint", np.nan) for row in rows
            ]),
            "EV_vs_previous_sem": _safe_sem([
                row.get("EV_vs_previous_checkpoint", np.nan) for row in rows
            ]),
            "exploitability_mean": _mean([row.get("exploitability", np.nan) for row in rows]),
            "exploitability_sem": _safe_sem([row.get("exploitability", np.nan) for row in rows]),
            "policy_value_mean": _mean([row.get("policy_value", np.nan) for row in rows]),
            "policy_value_sem": _safe_sem([row.get("policy_value", np.nan) for row in rows]),
            "policy_value_error_mean": _mean([
                row.get("policy_value_error", np.nan) for row in rows
            ]),
            "policy_value_error_sem": _safe_sem([
                row.get("policy_value_error", np.nan) for row in rows
            ]),
        })

    baseline_comparison_rows = []
    metric_by_seed_arm = {
        (row["seed"], row["arm"], row["checkpoint"]): row
        for row in metric_rows
    }
    for seed in sorted(set(policies_by_seed) & set(baseline_policy_by_seed)):
        if final_iteration not in policies_by_seed[seed]:
            continue
        final_ckpt_pol = policies_by_seed[seed][final_iteration]
        baseline_pol = baseline_policy_by_seed[seed]
        ev_ckpt_vs_base = exact_seat_averaged_value_for_a(game, final_ckpt_pol, baseline_pol)
        ev_base_vs_ckpt = exact_seat_averaged_value_for_a(game, baseline_pol, final_ckpt_pol)
        ckpt_metrics = metric_by_seed_arm[(seed, "checkpointed", final_iteration)]
        base_metrics = metric_by_seed_arm[(seed, "continuous_baseline", final_iteration)]
        baseline_comparison_rows.append({
            "seed": int(seed),
            "checkpointed_final_vs_continuous_baseline_EV": ev_ckpt_vs_base[
                "A_EV_seat_averaged"
            ],
            "continuous_baseline_vs_checkpointed_final_EV": ev_base_vs_ckpt[
                "A_EV_seat_averaged"
            ],
            "checkpointed_final_exploitability": float(ckpt_metrics["exploitability"]),
            "continuous_baseline_exploitability": float(base_metrics["exploitability"]),
            "delta_exploitability_checkpointed_minus_baseline": float(
                ckpt_metrics["exploitability"] - base_metrics["exploitability"]
            ),
            "checkpointed_final_policy_value": float(ckpt_metrics["policy_value"]),
            "continuous_baseline_policy_value": float(base_metrics["policy_value"]),
            "delta_policy_value_checkpointed_minus_baseline": float(
                ckpt_metrics["policy_value"] - base_metrics["policy_value"]
            ),
            "checkpointed_final_policy_value_error": float(
                ckpt_metrics["policy_value_error"]
            ),
            "continuous_baseline_policy_value_error": float(
                base_metrics["policy_value_error"]
            ),
            "delta_policy_value_error_checkpointed_minus_baseline": float(
                ckpt_metrics["policy_value_error"] - base_metrics["policy_value_error"]
            ),
        })

    return {
        "loaded_policy_inventory": sorted(
            loaded_rows,
            key=lambda row: (row["seed"], row["arm"], row["checkpoint"]),
        ),
        "checkpoint_exploitability_metrics": sorted(
            metric_rows,
            key=lambda row: (row["seed"], row["arm"], row["checkpoint"]),
        ),
        "head_to_head_exact_pairwise": exact_records,
        "head_to_head_exact_mean_matrix": _matrix_rows(mean_matrix, "A_EV_seat_averaged"),
        "head_to_head_seed_win_fraction_matrix": _matrix_rows(
            win_fraction_matrix,
            "win_fraction",
        ),
        "head_to_head_monotonicity_summary_by_seed": monotonicity_rows,
        "head_to_head_strength_with_metrics": strength_with_metrics,
        "head_to_head_aggregate_strength_summary": aggregate_strength_rows,
        "best_checkpoint_summary": best_rows,
        "final_checkpoint_vs_continuous_baseline": baseline_comparison_rows,
        "matrix_values": {
            "mean_matrix": mean_matrix,
            "win_fraction_matrix": win_fraction_matrix,
        },
    }
