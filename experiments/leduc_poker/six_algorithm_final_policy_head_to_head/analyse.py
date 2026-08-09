"""Exact final-policy league and paired-seed analysis for Experiment 17."""

from __future__ import annotations

from collections import defaultdict
import itertools
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402

from .config import ALGORITHMS, ALGORITHM_ORDER, EQUIVALENCE_EPSILON, GAME_NAME
from .io_utils import write_csv, write_json
from .policies import load_policy, validate_policy_probabilities
from .statistics import holm_adjust, summary


def _exact_ev(game, policy_a, policy_b) -> dict:
    as_player_0 = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [policy_a, policy_b]
        )[0]
    )
    as_player_1 = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [policy_b, policy_a]
        )[1]
    )
    return {
        "algorithm_a_ev_as_player_0": as_player_0,
        "algorithm_a_ev_as_player_1": as_player_1,
        "algorithm_a_seat_averaged_ev": 0.5 * (as_player_0 + as_player_1),
    }


def _plot_matrix(matrix: np.ndarray, output_path: Path) -> None:
    labels = [ALGORITHMS[item]["algorithm_label"] for item in ALGORITHM_ORDER]
    limit = float(np.nanmax(np.abs(matrix))) if np.any(np.isfinite(matrix)) else 1.0
    limit = max(limit, 1e-9)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Opponent (algorithm B)")
    ax.set_ylabel("Algorithm A")
    set_chart_title(ax, "Six-Algorithm Final-Policy Head-to-Head\nMean Exact Two-Seat EV")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * limit else "black",
            )
    fig.colorbar(image, ax=ax, label="Exact seat-averaged EV for algorithm A")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_strength(rows: Sequence[Mapping], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: ALGORITHM_ORDER.index(row["algorithm_id"]))
    labels = [row["algorithm_label"] for row in ordered]
    values = np.asarray([row["mean_ev"] for row in ordered], dtype=float)
    lower = values - np.asarray([row["ci95_lower"] for row in ordered], dtype=float)
    upper = np.asarray([row["ci95_upper"] for row in ordered], dtype=float) - values
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(
        np.arange(len(rows)),
        values,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean exact EV versus the other five algorithms")
    set_chart_title(ax, "Final-Policy League Strength Across Five Paired Seeds")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_exploitability(rows: Sequence[Mapping], output_path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["algorithm_id"]].append(float(row["exploitability"]))
    labels = [ALGORITHMS[item]["algorithm_label"] for item in ALGORITHM_ORDER]
    means = [float(np.mean(grouped[item])) for item in ALGORITHM_ORDER]
    sems = [
        (
            float(np.std(grouped[item], ddof=1) / np.sqrt(len(grouped[item])))
            if len(grouped[item]) > 1
            else 0.0
        )
        for item in ALGORITHM_ORDER
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=sems, capsize=4)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Final Exploitability of the Six Head-to-Head Policies")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    snapshot_inventory: Sequence[Mapping],
    seeds: Sequence[int],
    run_dir: Path,
    equivalence_epsilon: float = EQUIVALENCE_EPSILON,
) -> dict:
    game = pyspiel.load_game(GAME_NAME)
    indexed_inventory = {
        (str(row["algorithm_id"]), int(row["seed"])): dict(row)
        for row in snapshot_inventory
    }
    expected_keys = {
        (algorithm_id, int(seed))
        for algorithm_id in ALGORITHM_ORDER
        for seed in seeds
    }
    if set(indexed_inventory) != expected_keys:
        missing = sorted(expected_keys - set(indexed_inventory))
        extra = sorted(set(indexed_inventory) - expected_keys)
        raise ValueError(f"Snapshot inventory mismatch; missing={missing}, extra={extra}")

    tabular_policies = {}
    loaded_rows = []
    for algorithm_id in ALGORITHM_ORDER:
        for seed in seeds:
            inventory = indexed_inventory[(algorithm_id, int(seed))]
            loaded = load_policy(game, algorithm_id, inventory["path"])
            validate_policy_probabilities(game, loaded)
            tabular = policy.tabular_policy_from_callable(
                game, loaded.action_probabilities
            )
            tabular_policies[(algorithm_id, int(seed))] = tabular
            loaded_rows.append(
                {
                    **inventory,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "probability_validation": "passed_exhaustive_tabular_conversion",
                }
            )

    metric_rows = []
    for (algorithm_id, seed), tabular in tabular_policies.items():
        nash_conv = float(exploitability.nash_conv(game, tabular))
        self_play_value = float(
            expected_game_score.policy_value(
                game.new_initial_state(), [tabular, tabular]
            )[0]
        )
        metric_rows.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                "seed": int(seed),
                "nash_conv": nash_conv,
                "exploitability": nash_conv / 2.0,
                "self_play_value_player_0": self_play_value,
            }
        )

    same_seed_rows = []
    for seed in seeds:
        for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
            values = _exact_ev(
                game,
                tabular_policies[(algorithm_a, int(seed))],
                tabular_policies[(algorithm_b, int(seed))],
            )
            ev = values["algorithm_a_seat_averaged_ev"]
            same_seed_rows.append(
                {
                    "algorithm_a": algorithm_a,
                    "algorithm_a_label": ALGORITHMS[algorithm_a]["algorithm_label"],
                    "algorithm_b": algorithm_b,
                    "algorithm_b_label": ALGORITHMS[algorithm_b]["algorithm_label"],
                    "seed": int(seed),
                    **values,
                    "classification_for_a": (
                        "win"
                        if ev > equivalence_epsilon
                        else "loss"
                        if ev < -equivalence_epsilon
                        else "practical_tie"
                    ),
                }
            )

    pair_inference = []
    for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
        values = [
            row["algorithm_a_seat_averaged_ev"]
            for row in same_seed_rows
            if row["algorithm_a"] == algorithm_a
            and row["algorithm_b"] == algorithm_b
        ]
        pair_inference.append(
            {
                "algorithm_a": algorithm_a,
                "algorithm_a_label": ALGORITHMS[algorithm_a]["algorithm_label"],
                "algorithm_b": algorithm_b,
                "algorithm_b_label": ALGORITHMS[algorithm_b]["algorithm_label"],
                **summary(values),
            }
        )
    holm_adjust(
        pair_inference,
        "two_sided_exact_sign_flip_p",
        "holm_adjusted_two_sided_p",
    )

    cross_seed_rows = []
    for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
        for seed_a in seeds:
            for seed_b in seeds:
                cross_seed_rows.append(
                    {
                        "algorithm_a": algorithm_a,
                        "algorithm_a_label": ALGORITHMS[algorithm_a]["algorithm_label"],
                        "seed_a": int(seed_a),
                        "algorithm_b": algorithm_b,
                        "algorithm_b_label": ALGORITHMS[algorithm_b]["algorithm_label"],
                        "seed_b": int(seed_b),
                        **_exact_ev(
                            game,
                            tabular_policies[(algorithm_a, int(seed_a))],
                            tabular_policies[(algorithm_b, int(seed_b))],
                        ),
                    }
                )

    strength_by_seed = []
    for algorithm_id in ALGORITHM_ORDER:
        for seed in seeds:
            effects = []
            for row in same_seed_rows:
                if row["seed"] != int(seed):
                    continue
                if row["algorithm_a"] == algorithm_id:
                    effects.append(row["algorithm_a_seat_averaged_ev"])
                elif row["algorithm_b"] == algorithm_id:
                    effects.append(-row["algorithm_a_seat_averaged_ev"])
            strength_by_seed.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "seed": int(seed),
                    "mean_ev_vs_other_algorithms": float(np.mean(effects)),
                    "num_opponents": len(effects),
                }
            )
    strength_summary = []
    for algorithm_id in ALGORITHM_ORDER:
        values = [
            row["mean_ev_vs_other_algorithms"]
            for row in strength_by_seed
            if row["algorithm_id"] == algorithm_id
        ]
        strength_summary.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                **summary(values),
            }
        )
    strength_summary.sort(key=lambda row: row["mean_ev"], reverse=True)
    for rank, row in enumerate(strength_summary, start=1):
        row["rank_by_mean_exact_ev"] = rank

    matrix = np.zeros((len(ALGORITHM_ORDER), len(ALGORITHM_ORDER)), dtype=float)
    for row in pair_inference:
        a = ALGORITHM_ORDER.index(row["algorithm_a"])
        b = ALGORITHM_ORDER.index(row["algorithm_b"])
        matrix[a, b] = row["mean_ev"]
        matrix[b, a] = -row["mean_ev"]
    matrix_rows = []
    for index, algorithm_id in enumerate(ALGORITHM_ORDER):
        matrix_rows.append(
            {
                "algorithm_id": algorithm_id,
                **{
                    opponent: float(matrix[index, opponent_index])
                    for opponent_index, opponent in enumerate(ALGORITHM_ORDER)
                },
            }
        )

    outputs = {
        "loaded_policy_inventory": run_dir / "loaded_policy_inventory.csv",
        "final_policy_metrics": run_dir / "final_policy_metrics.csv",
        "same_seed_pairwise": run_dir / "head_to_head_same_seed_pairwise.csv",
        "pairwise_inference": run_dir / "head_to_head_pairwise_inference.csv",
        "cross_seed_league": run_dir / "head_to_head_cross_seed_league.csv",
        "strength_by_seed": run_dir / "algorithm_strength_by_seed.csv",
        "strength_summary": run_dir / "algorithm_strength_summary.csv",
        "mean_ev_matrix": run_dir / "head_to_head_mean_ev_matrix.csv",
        "aggregate_summary": run_dir / "aggregate_summary.json",
        "mean_ev_heatmap": run_dir / "head_to_head_mean_ev_heatmap.png",
        "strength_plot": run_dir / "algorithm_strength.png",
        "exploitability_plot": run_dir / "final_exploitability.png",
    }
    write_csv(outputs["loaded_policy_inventory"], loaded_rows)
    write_csv(outputs["final_policy_metrics"], metric_rows)
    write_csv(outputs["same_seed_pairwise"], same_seed_rows)
    write_csv(outputs["pairwise_inference"], pair_inference)
    write_csv(outputs["cross_seed_league"], cross_seed_rows)
    write_csv(outputs["strength_by_seed"], strength_by_seed)
    write_csv(outputs["strength_summary"], strength_summary)
    write_csv(outputs["mean_ev_matrix"], matrix_rows)
    write_json(
        outputs["aggregate_summary"],
        {
            "evaluation": "exact OpenSpiel expected value in both seats",
            "sampled_games": 0,
            "training_seeds_are_inferential_unit": True,
            "num_algorithms": len(ALGORITHM_ORDER),
            "num_training_seeds": len(seeds),
            "num_same_seed_pairwise_effects": len(same_seed_rows),
            "num_cross_seed_exact_matchups": len(cross_seed_rows),
            "num_exact_seat_assignments": 2
            * (len(same_seed_rows) + len(cross_seed_rows)),
            "equivalence_epsilon": equivalence_epsilon,
            "pairwise_inference": pair_inference,
            "algorithm_strength_summary": strength_summary,
            "multiplicity_note": (
                "Holm correction covers 15 pairwise two-sided tests; with five seeds "
                "the minimum exact two-sided sign-flip p-value is 0.0625"
            ),
        },
    )
    _plot_matrix(matrix, outputs["mean_ev_heatmap"])
    _plot_strength(strength_summary, outputs["strength_plot"])
    _plot_exploitability(metric_rows, outputs["exploitability_plot"])
    return {key: str(value) for key, value in outputs.items()}


__all__ = ["run_analysis"]
