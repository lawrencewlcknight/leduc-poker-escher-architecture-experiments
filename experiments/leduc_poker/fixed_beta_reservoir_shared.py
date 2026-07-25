"""Shared runner utilities for fixed-beta reservoir ESCHER experiments."""

from __future__ import annotations

import csv
import gc
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from escher_poker.chart_titles import set_chart_title
from escher_poker.constants import (
    DEFAULT_FINAL_WINDOW,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (
    run as shared,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes import (
    run as experiment_6,
)


DIAGNOSTIC_FIELDS = tuple(
    dict.fromkeys(
        (
            *experiment_6.DIAGNOSTIC_FIELDS,
            "q_lifetime_seen_count",
            "q_fold_0_lifetime_seen_count",
            "q_fold_1_lifetime_seen_count",
            "q_fold_2_lifetime_seen_count",
        )
    )
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value) -> float:
    return np.nan if value in {None, ""} else float(value)


def run_candidate(
    *,
    seed: int,
    config: Dict[str, Any],
    target_nodes: int,
    algorithm_id: str,
    algorithm_label: str,
    result_source: str,
):
    import torch

    from fixed_beta_reservoir_escher import FixedBetaReservoirEscher
    from vr_deep_cfr.logger import Logger

    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    solver = FixedBetaReservoirEscher(**kwargs)
    solver.target_nodes_touched = int(target_nodes)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )
    raw_checkpoints = solver.solve()

    curves = []
    for checkpoint_index, raw in enumerate(raw_checkpoints):
        value = float(raw["average_policy_value"])
        row = {
            "algorithm_id": algorithm_id,
            "algorithm_label": algorithm_label,
            "seed": int(seed),
            "checkpoint_index": int(checkpoint_index),
            "iteration": int(raw["iteration"]),
            "episode": int(raw["episode"]),
            "nodes_touched": float(raw["nodes_touched"]),
            "wall_clock_seconds": float(raw["wall_clock_seconds"]),
            "exploitability": float(raw["exp"]),
            "average_policy_value": value,
            "policy_value_error": abs(value - LEDUC_GAME_VALUE_PLAYER_0),
            "average_policy_loss": parse_float(raw.get("average_policy_loss")),
            "regret_loss_player_0": parse_float(raw.get("regret_loss_0")),
            "regret_loss_player_1": parse_float(raw.get("regret_loss_1")),
            "baseline_loss_player_0": parse_float(raw.get("baseline_loss_0")),
            "baseline_loss_player_1": parse_float(raw.get("baseline_loss_1")),
            "checkpoint_kind": str(raw.get("checkpoint_kind", "outer_iteration")),
            "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
            "is_initial_policy_evaluation": (
                raw.get("checkpoint_kind") == "initial_untrained_policy"
            ),
            "is_final_policy_evaluation": False,
            "result_source": result_source,
        }
        for field in DIAGNOSTIC_FIELDS:
            row[field] = parse_float(raw.get(field))
        curves.append(row)

    final = curves[-1]
    training_curves = [
        row for row in curves if not row["is_initial_policy_evaluation"]
    ]
    exploitabilities = [row["exploitability"] for row in training_curves]
    nodes = [row["nodes_touched"] for row in training_curves]
    wall_times = [row["wall_clock_seconds"] for row in training_curves]
    node_delta = float(final["nodes_touched"] - target_nodes)
    fold_sizes = solver.q_value_trainer.fold_sizes()
    lifetime_seen_counts = solver.q_value_trainer.fold_lifetime_seen_counts()
    summary = {
        "algorithm_id": algorithm_id,
        "algorithm_label": algorithm_label,
        "seed": int(seed),
        "final_exploitability": float(final["exploitability"]),
        "best_exploitability": float(np.min(exploitabilities)),
        "final_window_mean_exploitability": float(
            np.mean(
                exploitabilities[
                    -min(DEFAULT_FINAL_WINDOW, len(exploitabilities)) :
                ]
            )
        ),
        "final_policy_value": float(final["average_policy_value"]),
        "final_policy_value_error": float(final["policy_value_error"]),
        "final_nash_conv_recomputed": 2.0 * float(final["exploitability"]),
        "final_nodes_touched": float(final["nodes_touched"]),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "num_iterations_completed": int(final["iteration"]),
        "num_intermediate_points": len(curves),
        "exploitability_normalised_auc_nodes": shared._normalised_auc(
            nodes,
            exploitabilities,
        ),
        "nodes_to_exploitability_threshold": shared._first_x_to_threshold(
            nodes,
            exploitabilities,
            EXPLOITABILITY_THRESHOLD,
        ),
        "seconds_to_exploitability_threshold": shared._first_x_to_threshold(
            wall_times,
            exploitabilities,
            EXPLOITABILITY_THRESHOLD,
        ),
        "target_nodes_touched": float(target_nodes),
        "node_budget_delta": node_delta,
        "node_budget_relative_delta": node_delta / float(target_nodes),
        "final_average_policy_buffer_size": len(solver.ave_policy_trainer.buffer),
        "final_advantage_buffer_size_player_0": len(
            solver.regret_trainers[0].buffer
        ),
        "final_advantage_buffer_size_player_1": len(
            solver.regret_trainers[1].buffer
        ),
        "final_history_value_buffer_size": int(sum(fold_sizes)),
        "final_lifetime_q_seen_count": int(sum(lifetime_seen_counts)),
        "result_source": result_source,
    }
    for field in DIAGNOSTIC_FIELDS:
        summary[f"final_{field}"] = float(final[field])

    estimator_rows = [
        row
        for row in curves
        if float(row.get("unbiased_estimator_sample_count", 0.0)) > 0.0
    ]
    if not estimator_rows:
        raise RuntimeError("The run produced no unbiased estimator samples")
    for row in estimator_rows:
        if not np.isclose(row["control_variate_beta_min"], 1.0):
            raise RuntimeError("The candidate did not keep beta fixed at one")
        if not np.isclose(row["control_variate_beta_max"], 1.0):
            raise RuntimeError("The candidate did not keep beta fixed at one")
    if float(final["policy_weighted_advantage_abs_mean"]) > 1e-10:
        raise RuntimeError("Control-variate advantages were not policy-centred")
    if min(fold_sizes) <= 0:
        raise RuntimeError("A cross-fitted critic fold received no data")
    if min(lifetime_seen_counts) <= 0:
        raise RuntimeError("A lifetime reservoir observed no transitions")

    del solver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"seed": int(seed), "summary": summary, "curves": curves}


def load_reference_curves(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_algorithm_ids: Sequence[str],
    expected_seeds: Sequence[int],
    result_source: str,
    label_overrides: Mapping[str, str] | None = None,
):
    digest = sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            f"Reference curve checksum mismatch: expected {expected_sha256}, "
            f"found {digest}"
        )
    labels = dict(label_overrides or {})
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in ("seed", "checkpoint_index", "iteration", "episode"):
                row[field] = int(float(row[field]))
            for field in (
                "nodes_touched",
                "wall_clock_seconds",
                "exploitability",
                "average_policy_value",
                "policy_value_error",
            ):
                row[field] = parse_float(row.get(field))
            row["is_initial_policy_evaluation"] = shared._parse_bool(
                row.get("is_initial_policy_evaluation", False)
            )
            row["is_final_policy_evaluation"] = shared._parse_bool(
                row.get("is_final_policy_evaluation", False)
            )
            if row["algorithm_id"] in labels:
                row["algorithm_label"] = labels[row["algorithm_id"]]
            row["result_source"] = result_source
            rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} reference curves, found {len(rows)}")
    if {row["algorithm_id"] for row in rows} != set(expected_algorithm_ids):
        raise ValueError("Reference curves contain unexpected algorithms")
    if {int(row["seed"]) for row in rows} != set(expected_seeds):
        raise ValueError("Reference curves contain unexpected seeds")
    return rows


def load_reference_summaries(
    path: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
    expected_algorithm_ids: Sequence[str],
    expected_seeds: Sequence[int],
    result_source: str,
    label_overrides: Mapping[str, str] | None = None,
):
    digest = sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            f"Reference summary checksum mismatch: expected {expected_sha256}, "
            f"found {digest}"
        )
    labels = dict(label_overrides or {})
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for key, value in list(row.items()):
                if key in {
                    "algorithm_id",
                    "algorithm_label",
                    "variant_id",
                    "variant_label",
                    "result_source",
                }:
                    continue
                try:
                    row[key] = parse_float(value)
                except ValueError:
                    pass
            row["seed"] = int(row["seed"])
            if row["algorithm_id"] in labels:
                row["algorithm_label"] = labels[row["algorithm_id"]]
            row["result_source"] = result_source
            rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} reference summaries, found {len(rows)}"
        )
    if {row["algorithm_id"] for row in rows} != set(expected_algorithm_ids):
        raise ValueError("Reference summaries contain unexpected algorithms")
    if {int(row["seed"]) for row in rows} != set(expected_seeds):
        raise ValueError("Reference summaries contain unexpected seeds")
    return rows


def aggregate(
    summary_rows: Sequence[Mapping[str, Any]],
    algorithm_ids: Sequence[str],
):
    result = {}
    for algorithm_id in algorithm_ids:
        rows = [row for row in summary_rows if row["algorithm_id"] == algorithm_id]
        if not rows:
            continue
        numeric_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        result[algorithm_id] = {
            field: shared._stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return result


def paired_differences(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_algorithm_id: str,
    reference_algorithm_ids: Sequence[str],
    algorithm_labels: Mapping[str, str],
    seeds: Sequence[int],
):
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for baseline_id in reference_algorithm_ids:
        for seed in seeds:
            baseline = indexed.get((baseline_id, int(seed)))
            candidate = indexed.get((candidate_algorithm_id, int(seed)))
            if baseline is None or candidate is None:
                continue
            rows.append(
                {
                    "baseline_algorithm_id": baseline_id,
                    "baseline_algorithm_label": algorithm_labels[baseline_id],
                    "seed": int(seed),
                    "exploitability_difference": (
                        candidate["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "normalised_auc_difference": (
                        candidate["exploitability_normalised_auc_nodes"]
                        - baseline["exploitability_normalised_auc_nodes"]
                    ),
                    "nodes_difference": (
                        candidate["final_nodes_touched"]
                        - baseline["final_nodes_touched"]
                    ),
                    "wall_clock_seconds_difference": (
                        candidate["final_wall_clock_seconds"]
                        - baseline["final_wall_clock_seconds"]
                    ),
                    "wall_clock_ratio": (
                        candidate["final_wall_clock_seconds"]
                        / baseline["final_wall_clock_seconds"]
                    ),
                }
            )
    return rows


def mean_curve(rows, algorithm_id: str, x_key: str):
    selected = [
        row
        for row in rows
        if row["algorithm_id"] == algorithm_id
        and not bool(row.get("is_final_policy_evaluation", False))
    ]
    checkpoints = sorted({int(row["checkpoint_index"]) for row in selected})
    xs, means, ses = [], [], []
    for checkpoint in checkpoints:
        current = [
            row for row in selected if int(row["checkpoint_index"]) == checkpoint
        ]
        x = np.asarray([row[x_key] for row in current], dtype=float)
        y = np.asarray([row["exploitability"] for row in current], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            xs.append(float(np.mean(x[finite])))
            means.append(float(np.mean(y[finite])))
            ses.append(float(shared._stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def plot_exploitability(
    run_dir: Path,
    rows,
    *,
    x_key: str,
    algorithm_ids: Sequence[str],
    algorithm_labels: Mapping[str, str],
    colors: Mapping[str, str],
    title: str,
):
    is_time = x_key == "wall_clock_seconds"
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algorithm_id in algorithm_ids:
        algorithm_rows = [
            row
            for row in rows
            if row["algorithm_id"] == algorithm_id
            and not bool(row.get("is_final_policy_evaluation", False))
        ]
        if not algorithm_rows:
            continue
        divisor = 3600.0 if is_time else 1.0
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                [row for row in algorithm_rows if int(row["seed"]) == seed],
                key=lambda row: row[x_key],
            )
            ax.plot(
                [row[x_key] / divisor for row in seed_rows],
                [row["exploitability"] for row in seed_rows],
                color=colors[algorithm_id],
                linewidth=1,
                alpha=0.16,
            )
        x, mean, se = mean_curve(rows, algorithm_id, x_key)
        ax.plot(
            x / divisor,
            mean,
            marker="o",
            linewidth=2.2,
            color=colors[algorithm_id],
            label=algorithm_labels[algorithm_id],
        )
        ax.fill_between(
            x / divisor,
            mean - se,
            mean + se,
            color=colors[algorithm_id],
            alpha=0.14,
        )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Wall-clock training time (hours)" if is_time else "Nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    dimension = "wall-clock time" if is_time else "nodes touched"
    set_chart_title(ax, f"{title} by {dimension}")
    ax.legend()
    fig.tight_layout()
    filename = (
        "combined_exploitability_by_wall_clock.png"
        if is_time
        else "combined_exploitability_by_nodes.png"
    )
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_final(
    run_dir: Path,
    summaries,
    *,
    algorithm_ids: Sequence[str],
    algorithm_labels: Mapping[str, str],
    colors: Mapping[str, str],
    title: str,
):
    labels, means, ses, bar_colors = [], [], [], []
    for algorithm_id in algorithm_ids:
        values = [
            row["final_exploitability"]
            for row in summaries
            if row["algorithm_id"] == algorithm_id
        ]
        if not values:
            continue
        stats = shared._stats(values)
        labels.append(algorithm_labels[algorithm_id])
        means.append(stats["mean"])
        ses.append(stats["se"])
        bar_colors.append(colors[algorithm_id])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        np.arange(len(labels)),
        means,
        yerr=ses,
        color=bar_colors,
        capsize=5,
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, title)
    fig.tight_layout()
    fig.savefig(
        run_dir / "combined_final_exploitability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
