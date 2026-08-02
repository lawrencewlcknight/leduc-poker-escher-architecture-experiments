"""Exact temporal head-to-head analysis for Experiment 16 snapshots."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.checkpoint_analysis import analyze_checkpoint_snapshots  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from escher_poker.json_utils import json_safe  # noqa: E402
from escher_poker.policy_snapshots import (  # noqa: E402
    discover_policy_snapshots,
    load_pickle,
)

from .statistics import build_inference_tables  # noqa: E402


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_sem(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return float("nan"), float("nan")
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), sem


def _node_summaries(training_rows: Sequence[Mapping]) -> dict:
    by_checkpoint = defaultdict(list)
    targets = {}
    for row in training_rows:
        checkpoint = int(row["checkpoint_iteration"])
        by_checkpoint[checkpoint].append(float(row["nodes_touched"]))
        targets[checkpoint] = int(row["checkpoint_target_nodes"])
    return {
        checkpoint: {
            "checkpoint_target_nodes": targets[checkpoint],
            "nodes_touched_mean": _mean_sem(values)[0],
            "nodes_touched_sem": _mean_sem(values)[1],
            "nodes_touched_n": len(values),
        }
        for checkpoint, values in by_checkpoint.items()
    }


def _plot_heatmap(
    matrix: Mapping[int, Mapping[int, float]],
    schedule: Sequence[int],
    node_lookup: Mapping[int, Mapping[str, float]],
    output_path: Path,
    *,
    annotate: bool,
) -> None:
    schedule = [int(value) for value in schedule]
    values = np.asarray(
        [
            [
                matrix[row].get(column, np.nan) if row > column else np.nan
                for column in schedule
            ]
            for row in schedule
        ],
        dtype=np.float64,
    )
    finite = values[np.isfinite(values)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    if limit == 0.0:
        limit = 1.0
    labels = [
        f"{node_lookup[value]['nodes_touched_mean'] / 1e6:.1f}M"
        if value in node_lookup
        else str(value)
        for value in schedule
    ]
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(schedule)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Earlier policy checkpoint (mean nodes touched)")
    ax.set_ylabel("Later policy checkpoint (mean nodes touched)")
    set_chart_title(
        ax,
        "Unbiased Control-Variate ESCHER Later-versus-Earlier Performance\n"
        "Mean Exact Two-Seat EV Across Seeds",
    )
    fig.colorbar(
        image,
        ax=ax,
        label="EV of later checkpoint against earlier checkpoint",
    )
    if annotate:
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                if np.isfinite(value):
                    colour = "white" if abs(value) > 0.5 * limit else "black"
                    ax.text(
                        column,
                        row,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=colour,
                    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_curve(
    aggregate_rows: Sequence[Mapping],
    y_key: str,
    sem_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    *,
    target: Optional[float] = None,
    target_label: Optional[str] = None,
) -> None:
    rows = sorted(aggregate_rows, key=lambda row: int(row["checkpoint"]))
    x = np.asarray([row["nodes_touched_mean"] for row in rows], dtype=np.float64)
    y = np.asarray([row[y_key] for row in rows], dtype=np.float64)
    yerr = np.asarray([row[sem_key] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=3,
        label="Mean across seeds",
    )
    if target is not None:
        ax.axhline(
            target,
            color="black",
            linestyle="--",
            linewidth=1,
            label=target_label,
        )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel(ylabel)
    set_chart_title(ax, title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_seed_effects(seed_rows: Sequence[Mapping], output_path: Path) -> None:
    rows = sorted(seed_rows, key=lambda row: int(row["seed"]))
    values = np.asarray(
        [row["mean_later_vs_earlier_ev"] for row in rows],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(values.size)
    ax.scatter(x, values, s=42, label="Per-seed mean")
    if values.size:
        ax.axhline(float(np.mean(values)), linewidth=2, label="Mean across seeds")
    ax.axhline(0.0, linestyle="--", linewidth=1, label="No head-to-head difference")
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["seed"]) for row in rows])
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean exact EV of later vs earlier checkpoints")
    set_chart_title(ax, "Temporal Checkpoint Improvement Across Seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_strength_vs_exploitability(
    aggregate_rows: Sequence[Mapping],
    output_path: Path,
) -> None:
    rows = sorted(aggregate_rows, key=lambda row: int(row["checkpoint"]))
    x = np.asarray([row["exploitability_mean"] for row in rows], dtype=np.float64)
    y = np.asarray(
        [row["mean_EV_vs_all_other_mean"] for row in rows],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(x, y, s=55)
    for row, x_value, y_value in zip(rows, x, y):
        ax.annotate(
            f"{row['nodes_touched_mean'] / 1e6:.1f}M",
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Exploitability (NashConv / 2)")
    ax.set_ylabel("Mean exact EV vs all other checkpoints")
    set_chart_title(ax, "Equilibrium Quality versus Temporal Head-to-Head Strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    *,
    config: Mapping[str, object],
    checkpoint_schedule: Sequence[int],
    checkpoint_node_thresholds: Sequence[int],
    equivalence_epsilon: float,
    run_dir: Path,
    snapshots_dir: Optional[Path] = None,
) -> dict:
    """Evaluate every complete five-checkpoint seed trajectory exactly."""
    run_dir = Path(run_dir)
    snapshots_dir = Path(snapshots_dir or run_dir / "snapshots")
    schedule = tuple(int(value) for value in checkpoint_schedule)
    required = set(schedule)
    inventory = discover_policy_snapshots(snapshots_dir)
    if not inventory:
        raise FileNotFoundError(f"No ESCHER policy snapshots found under {snapshots_dir}")

    by_seed = defaultdict(set)
    for row in inventory:
        if row["arm"] == "checkpointed":
            by_seed[int(row["seed"])].add(int(row["iteration"]))
    complete_seeds = {
        seed for seed, checkpoints in by_seed.items() if required.issubset(checkpoints)
    }
    filtered_inventory = [
        row
        for row in inventory
        if int(row["seed"]) in complete_seeds
        and row["arm"] == "checkpointed"
        and int(row["iteration"]) in required
    ]
    if not filtered_inventory:
        raise RuntimeError("No seed has the complete requested checkpoint schedule")

    training_path = run_dir / "training_stage_metrics.csv"
    if training_path.exists():
        training_rows = _read_csv(training_path)
    else:
        training_rows = [
            {
                "seed": row["seed"],
                "checkpoint_iteration": row["iteration"],
                "checkpoint_target_nodes": load_pickle(row["path"])[
                    "checkpoint_target_nodes"
                ],
                "nodes_touched": load_pickle(row["path"])["nodes_visited"],
            }
            for row in filtered_inventory
        ]
    training_rows = [
        row for row in training_rows if int(row["seed"]) in complete_seeds
    ]
    nodes_by_seed_checkpoint = {
        (int(row["seed"]), int(row["checkpoint_iteration"])): float(
            row["nodes_touched"]
        )
        for row in training_rows
    }
    nodes_by_checkpoint = _node_summaries(training_rows)

    game = pyspiel.load_game(str(config["game_name"]))
    analysis = analyze_checkpoint_snapshots(
        game,
        filtered_inventory,
        list(schedule),
        schedule[-1],
        float(equivalence_epsilon),
    )

    pairwise_rows = [
        {
            "seed": int(row["seed"]),
            "checkpoint_a": int(row["checkpoint_A"]),
            "checkpoint_b": int(row["checkpoint_B"]),
            "A_EV_as_player0": float(row["A_EV_as_player0"]),
            "A_EV_as_player1": float(row["A_EV_as_player1"]),
            "A_EV_seat_averaged": float(row["A_EV_seat_averaged"]),
        }
        for row in analysis["head_to_head_exact_pairwise"]
    ]
    metric_rows = []
    for row in analysis["checkpoint_exploitability_metrics"]:
        merged = dict(row)
        merged["nodes_touched"] = nodes_by_seed_checkpoint.get(
            (int(row["seed"]), int(row["checkpoint"])),
            float("nan"),
        )
        metric_rows.append(merged)

    strength_rows = []
    for row in analysis["head_to_head_strength_with_metrics"]:
        merged = dict(row)
        merged["nodes_touched"] = nodes_by_seed_checkpoint.get(
            (int(row["seed"]), int(row["checkpoint"])),
            float("nan"),
        )
        strength_rows.append(merged)

    aggregate_rows = []
    for row in analysis["head_to_head_aggregate_strength_summary"]:
        merged = dict(row)
        checkpoint = int(row["checkpoint"])
        merged.update(nodes_by_checkpoint.get(checkpoint, {}))
        all_other_values = [
            float(candidate["mean_EV_vs_all_other_checkpoints"])
            for candidate in strength_rows
            if int(candidate["checkpoint"]) == checkpoint
        ]
        merged["mean_EV_vs_all_other_mean"] = _mean_sem(all_other_values)[0]
        merged["mean_EV_vs_all_other_sem"] = _mean_sem(all_other_values)[1]
        aggregate_rows.append(merged)

    seed_rows, inference_rows, pair_inference_rows = build_inference_tables(
        pairwise_rows,
        schedule,
    )
    for row in pair_inference_rows:
        row["later_nodes_touched_mean"] = nodes_by_checkpoint.get(
            int(row["later_checkpoint"]),
            {},
        ).get("nodes_touched_mean", float("nan"))
        row["earlier_nodes_touched_mean"] = nodes_by_checkpoint.get(
            int(row["earlier_checkpoint"]),
            {},
        ).get("nodes_touched_mean", float("nan"))

    outputs = {
        "checkpoint_inventory": run_dir / "checkpoint_inventory.csv",
        "loaded_policy_inventory": run_dir / "loaded_policy_inventory.csv",
        "checkpoint_exploitability_metrics": (
            run_dir / "checkpoint_exploitability_metrics.csv"
        ),
        "head_to_head_pairwise": run_dir / "head_to_head_pairwise.csv",
        "head_to_head_mean_matrix": run_dir / "head_to_head_mean_matrix.csv",
        "head_to_head_seed_win_fraction_matrix": (
            run_dir / "head_to_head_seed_win_fraction_matrix.csv"
        ),
        "head_to_head_monotonicity_by_seed": (
            run_dir / "head_to_head_monotonicity_by_seed.csv"
        ),
        "head_to_head_strength_by_checkpoint": (
            run_dir / "head_to_head_strength_by_checkpoint.csv"
        ),
        "head_to_head_strength_aggregate": (
            run_dir / "head_to_head_strength_aggregate.csv"
        ),
        "best_checkpoint_summary": run_dir / "best_checkpoint_summary.csv",
        "head_to_head_primary_effect_by_seed": (
            run_dir / "head_to_head_primary_effect_by_seed.csv"
        ),
        "head_to_head_inference_summary": (
            run_dir / "head_to_head_inference_summary.csv"
        ),
        "head_to_head_pairwise_inference": (
            run_dir / "head_to_head_pairwise_inference.csv"
        ),
        "aggregate_summary": run_dir / "aggregate_summary.json",
    }
    _write_csv(outputs["checkpoint_inventory"], filtered_inventory)
    _write_csv(
        outputs["loaded_policy_inventory"],
        analysis["loaded_policy_inventory"],
    )
    _write_csv(outputs["checkpoint_exploitability_metrics"], metric_rows)
    _write_csv(outputs["head_to_head_pairwise"], pairwise_rows)
    _write_csv(
        outputs["head_to_head_mean_matrix"],
        analysis["head_to_head_exact_mean_matrix"],
    )
    _write_csv(
        outputs["head_to_head_seed_win_fraction_matrix"],
        analysis["head_to_head_seed_win_fraction_matrix"],
    )
    _write_csv(
        outputs["head_to_head_monotonicity_by_seed"],
        analysis["head_to_head_monotonicity_summary_by_seed"],
    )
    _write_csv(outputs["head_to_head_strength_by_checkpoint"], strength_rows)
    _write_csv(outputs["head_to_head_strength_aggregate"], aggregate_rows)
    _write_csv(
        outputs["best_checkpoint_summary"],
        analysis["best_checkpoint_summary"],
    )
    _write_csv(outputs["head_to_head_primary_effect_by_seed"], seed_rows)
    _write_csv(outputs["head_to_head_inference_summary"], inference_rows)
    _write_csv(outputs["head_to_head_pairwise_inference"], pair_inference_rows)

    summaries_by_estimand = {row["estimand"]: row for row in inference_rows}
    with open(outputs["aggregate_summary"], "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(
                {
                    "analysis_unit": "independent_training_seed",
                    "completed_seeds": sorted(complete_seeds),
                    "evaluation": (
                        "exact OpenSpiel expected value, averaged over both seats"
                    ),
                    "checkpoint_schedule": list(schedule),
                    "checkpoint_node_thresholds": list(
                        int(value) for value in checkpoint_node_thresholds
                    ),
                    "actual_nodes_touched_by_checkpoint": nodes_by_checkpoint,
                    "primary_estimand": summaries_by_estimand.get(
                        "seed_mean_ev_later_vs_all_earlier_checkpoints"
                    ),
                    "adjacent_checkpoint_estimand": summaries_by_estimand.get(
                        "seed_mean_ev_vs_immediately_previous_checkpoint"
                    ),
                    "final_vs_first_estimand": summaries_by_estimand.get(
                        "final_checkpoint_ev_vs_first_checkpoint"
                    ),
                    "multiple_testing": (
                        "Secondary checkpoint-pair sign-flip p-values use Holm "
                        "family-wise error correction."
                    ),
                }
            ),
            handle,
            indent=2,
        )

    matrix = analysis["matrix_values"]["mean_matrix"]
    _plot_heatmap(
        matrix,
        schedule,
        nodes_by_checkpoint,
        run_dir / "head_to_head_later_vs_earlier.png",
        annotate=True,
    )
    _plot_curve(
        aggregate_rows,
        "mean_EV_vs_earlier_mean",
        "mean_EV_vs_earlier_sem",
        "Mean EV vs all earlier checkpoints",
        "Does Later ESCHER Training Improve Head-to-Head Performance?",
        run_dir / "head_to_head_strength_vs_earlier_by_nodes.png",
        target=0.0,
        target_label="No head-to-head difference",
    )
    _plot_curve(
        aggregate_rows,
        "EV_vs_previous_mean",
        "EV_vs_previous_sem",
        "EV vs immediately previous checkpoint",
        "Adjacent ESCHER Checkpoint Improvement",
        run_dir / "head_to_head_strength_vs_previous_by_nodes.png",
        target=0.0,
        target_label="No head-to-head difference",
    )
    _plot_curve(
        aggregate_rows,
        "exploitability_mean",
        "exploitability_sem",
        "Exploitability (NashConv / 2)",
        "Unbiased Control-Variate ESCHER Checkpoint Exploitability",
        run_dir / "exploitability_by_nodes.png",
        target=NASH_EXPLOITABILITY_TARGET,
        target_label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    _plot_curve(
        aggregate_rows,
        "policy_value_mean",
        "policy_value_sem",
        "Average-policy value",
        "Unbiased Control-Variate ESCHER Average-Policy Value",
        run_dir / "average_policy_value_by_nodes.png",
        target=LEDUC_GAME_VALUE_PLAYER_0,
        target_label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    _plot_seed_effects(
        seed_rows,
        run_dir / "head_to_head_primary_effect_by_seed.png",
    )
    _plot_strength_vs_exploitability(
        aggregate_rows,
        run_dir / "strength_vs_exploitability.png",
    )

    outputs.update(
        {
            "head_to_head_later_vs_earlier_plot": (
                run_dir / "head_to_head_later_vs_earlier.png"
            ),
            "head_to_head_strength_vs_earlier_plot": (
                run_dir / "head_to_head_strength_vs_earlier_by_nodes.png"
            ),
            "head_to_head_strength_vs_previous_plot": (
                run_dir / "head_to_head_strength_vs_previous_by_nodes.png"
            ),
            "exploitability_plot": run_dir / "exploitability_by_nodes.png",
            "average_policy_value_plot": (
                run_dir / "average_policy_value_by_nodes.png"
            ),
            "primary_seed_effect_plot": (
                run_dir / "head_to_head_primary_effect_by_seed.png"
            ),
            "strength_vs_exploitability_plot": (
                run_dir / "strength_vs_exploitability.png"
            ),
        }
    )
    return outputs


__all__ = ["run_analysis"]
