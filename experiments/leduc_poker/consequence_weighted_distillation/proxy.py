"""Scalable strategic-consequence proxies and frozen selection logic."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, spearmanr

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    SourceData,
    empirical_table,
    evaluate_model_state,
)
from experiments.leduc_poker.causal_rare_state_audit.audit import (
    _full_exact_table,
    _infoset_catalog,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    write_csv,
    write_json,
)

from .config import (
    PROXY_LABELS,
    PROXY_ORDER,
    PROXY_REPAIR_FRACTIONS,
    SELECTABLE_PROXY_ORDER,
    SELECTION_CHECKPOINT,
)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _policy_key(state) -> tuple[int, str]:
    player = int(state.current_player())
    return player, str(state.information_state_string(player))


def counterfactual_action_values(game, policy_table: Mapping) -> dict:
    """Return counterfactual reach and policy-continuation values per infoset."""
    action_size = int(game.num_distinct_actions())
    accumulators: dict[tuple[int, str], dict] = {}

    def walk(state, reach: tuple[float, float], chance_reach: float) -> np.ndarray:
        if state.is_terminal():
            return np.asarray(state.returns(), dtype=np.float64)
        if state.is_chance_node():
            value = np.zeros(2, dtype=np.float64)
            for action, probability in state.chance_outcomes():
                child = walk(
                    state.child(action),
                    reach,
                    chance_reach * float(probability),
                )
                value += float(probability) * child
            return value

        player = int(state.current_player())
        key = _policy_key(state)
        legal = tuple(int(action) for action in state.legal_actions())
        policy = np.asarray(policy_table[key], dtype=np.float64)
        child_values = {}
        value = np.zeros(2, dtype=np.float64)
        for action in legal:
            next_reach = list(reach)
            next_reach[player] *= float(policy[action])
            child = walk(state.child(action), tuple(next_reach), chance_reach)
            child_values[action] = child
            value += float(policy[action]) * child

        weight = float(chance_reach * reach[1 - player])
        row = accumulators.setdefault(
            key,
            {
                "player": player,
                "legal_actions": legal,
                "counterfactual_reach": 0.0,
                "q_numerator": np.zeros(action_size, dtype=np.float64),
            },
        )
        row["counterfactual_reach"] += weight
        for action in legal:
            row["q_numerator"][action] += weight * float(child_values[action][player])
        return value

    walk(game.new_initial_state(), (1.0, 1.0), 1.0)
    result = {}
    for key, row in accumulators.items():
        denominator = max(float(row["counterfactual_reach"]), 1e-30)
        q_values = row["q_numerator"] / denominator
        legal = row["legal_actions"]
        policy = np.asarray(policy_table[key], dtype=np.float64)
        state_value = float(np.dot(policy[list(legal)], q_values[list(legal)]))
        result[key] = {
            "counterfactual_reach": float(row["counterfactual_reach"]),
            "q_values": q_values,
            "action_value_span": float(
                np.max(q_values[list(legal)]) - np.min(q_values[list(legal)])
            ),
            "expected_absolute_advantage": float(
                np.dot(
                    policy[list(legal)],
                    np.abs(q_values[list(legal)] - state_value),
                )
            ),
        }
    return result


def build_proxy_rows(
    source: SourceData,
    game,
    audit_rows: Sequence[Mapping],
) -> tuple[list[dict], list[dict]]:
    """Attach deployable proxies to Experiment 30's exact repair labels."""
    catalog = _infoset_catalog(game)
    exact_table = _full_exact_table(source, catalog)
    consequence = counterfactual_action_values(game, exact_table)
    neural_value, neural_table = evaluate_model_state(
        source, game, source.source_model_state
    )
    empirical_policy, _ = empirical_table(source, game)
    exact_value = exact_exploitability(game, exact_table)
    indexed = {
        (int(row["player"]), str(row["information_set"])): row
        for row in audit_rows
        if row["checkpoint_id"] == source.checkpoint_id
    }
    if set(indexed) != set(catalog):
        raise ValueError("Experiment 30 audit rows do not match the Leduc infoset catalog")

    raw_rows = []
    for key in sorted(catalog):
        audit = indexed[key]
        values = consequence[key]
        empirical = np.asarray(empirical_policy[key], dtype=np.float64)
        fitted = np.asarray(neural_table[key], dtype=np.float64)
        legal = list(catalog[key]["legal_actions"])
        policy_error = float(
            np.sum(
                np.where(
                    empirical[legal] > 0.0,
                    empirical[legal]
                    * (
                        np.log(np.clip(empirical[legal], 1e-12, 1.0))
                        - np.log(np.clip(fitted[legal], 1e-12, 1.0))
                    ),
                    0.0,
                )
            )
        )
        policy_error = max(policy_error, 0.0)
        reach = max(float(values["counterfactual_reach"]), 0.0)
        span = max(float(values["action_value_span"]), 0.0)
        expected = max(float(values["expected_absolute_advantage"]), 0.0)
        raw_rows.append(
            {
                "source_seed": int(source.seed),
                "checkpoint_id": source.checkpoint_id,
                "player": int(key[0]),
                "information_set": str(key[1]),
                "single_repair_gain": float(audit["single_repair_gain"]),
                "single_repair_gain_positive": float(
                    audit["single_repair_gain_positive"]
                ),
                "policy_error": policy_error,
                "exact_policy_error": max(float(audit["kl_exact_to_neural"]), 0.0),
                "counterfactual_reach": reach,
                "action_value_span": span,
                "expected_absolute_advantage": expected,
                "value_span_x_sqrt_reach": span * math.sqrt(reach),
                "expected_advantage_x_sqrt_reach": expected * math.sqrt(reach),
                "error_x_value_span": policy_error * span,
                "error_x_expected_advantage": policy_error * expected,
                "error_x_value_span_x_sqrt_reach": (
                    policy_error * span * math.sqrt(reach)
                ),
            }
        )

    metric_rows = []
    positive_total = max(
        sum(float(row["single_repair_gain_positive"]) for row in raw_rows), 1e-30
    )
    gap = float(neural_value - exact_value)
    for proxy_id in PROXY_ORDER:
        scores = np.asarray([float(row[proxy_id]) for row in raw_rows])
        gains = np.asarray(
            [float(row["single_repair_gain_positive"]) for row in raw_rows]
        )
        if np.ptp(scores) <= 0.0 or np.ptp(gains) <= 0.0:
            correlation = 0.0
        else:
            correlation = spearmanr(scores, gains).statistic
            correlation = 0.0 if not np.isfinite(correlation) else float(correlation)
        order = np.argsort(-scores, kind="stable")
        for fraction in PROXY_REPAIR_FRACTIONS:
            count = max(1, int(math.ceil(float(fraction) * len(raw_rows))))
            selected = order[:count]
            hybrid = dict(neural_table)
            for index in selected:
                row = raw_rows[int(index)]
                key = (int(row["player"]), str(row["information_set"]))
                hybrid[key] = exact_table[key]
            repaired = exact_exploitability(game, hybrid)
            metric_rows.append(
                {
                    "source_seed": int(source.seed),
                    "checkpoint_id": source.checkpoint_id,
                    "proxy_id": proxy_id,
                    "proxy_label": PROXY_LABELS[proxy_id],
                    "repair_fraction": float(fraction),
                    "num_repaired": count,
                    "spearman_positive_gain": correlation,
                    "positive_single_gain_capture": float(gains[selected].sum())
                    / positive_total,
                    "neural_exploitability": neural_value,
                    "exact_exploitability": exact_value,
                    "repaired_exploitability": repaired,
                    "fraction_distillation_gap_recovered": (
                        float((neural_value - repaired) / gap)
                        if abs(gap) > 1e-15
                        else float("nan")
                    ),
                }
            )
        percentiles = rankdata(scores, method="average") / float(len(scores))
        for index, row in enumerate(raw_rows):
            row[f"{proxy_id}_percentile"] = float(percentiles[index])
    return raw_rows, metric_rows


def select_proxy(proxy_metric_rows: Sequence[Mapping], output_dir: Path) -> dict:
    """Lock one proxy using only the predeclared 24-hour selection endpoint."""
    selection = [
        row
        for row in proxy_metric_rows
        if row["checkpoint_id"] == SELECTION_CHECKPOINT
        and np.isclose(float(row["repair_fraction"]), 0.10)
    ]
    summaries = []
    for proxy_id in SELECTABLE_PROXY_ORDER:
        rows = [row for row in selection if row["proxy_id"] == proxy_id]
        if not rows:
            raise ValueError(f"No 24-hour selection rows for {proxy_id}")
        summaries.append(
            {
                "proxy_id": proxy_id,
                "proxy_label": PROXY_LABELS[proxy_id],
                "n_source_seeds": len(rows),
                "mean_positive_single_gain_capture_at_10pct": float(
                    np.mean([float(row["positive_single_gain_capture"]) for row in rows])
                ),
                "mean_actual_gap_recovery_at_10pct": float(
                    np.mean(
                        [
                            float(row["fraction_distillation_gap_recovered"])
                            for row in rows
                        ]
                    )
                ),
                "mean_spearman_positive_gain": float(
                    np.mean([float(row["spearman_positive_gain"]) for row in rows])
                ),
            }
        )
    ranked = sorted(
        summaries,
        key=lambda row: (
            -row["mean_positive_single_gain_capture_at_10pct"],
            -row["mean_spearman_positive_gain"],
            row["proxy_id"],
        ),
    )
    selected = ranked[0]
    result = {
        "status": "complete",
        "selection_checkpoint": SELECTION_CHECKPOINT,
        "selection_fraction": 0.10,
        "selection_metric": "positive_single_gain_capture",
        "selected_proxy_id": selected["proxy_id"],
        "selected_proxy_label": selected["proxy_label"],
        "ranking": ranked,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "proxy_selection_summary.csv", ranked)
    write_json(output_dir / "selected_proxy.json", result)
    return result


__all__ = [
    "build_proxy_rows",
    "counterfactual_action_values",
    "select_proxy",
]
