"""Exact policy and stability analysis for Experiment 23."""

from __future__ import annotations

import csv
from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.policies import (  # noqa: E402
    load_policy,
    validate_policy_probabilities,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    summary,
)

from .common import read_json, sha256, write_csv, write_json  # noqa: E402
from .config import (  # noqa: E402
    FULL_ADAPTIVE,
    GAME_NAME,
    VARIANTS,
    VARIANT_ORDER,
    checkpoint_schedule,
    contract_manifest,
)


COLOURS = {
    VARIANT_ORDER[0]: "#9467bd",
    VARIANT_ORDER[1]: "#1f77b4",
    VARIANT_ORDER[2]: "#ff7f0e",
    VARIANT_ORDER[3]: "#2ca02c",
}
UCV_POLICY_LOADER_ID = "unbiased_control_variate_escher"


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant_id"], row["checkpoint_id"])].append(row)
    result = []
    for variant_id in VARIANT_ORDER:
        for checkpoint_id in {
            row["checkpoint_id"] for row in rows if row["variant_id"] == variant_id
        }:
            values = grouped[(variant_id, checkpoint_id)]
            stats = summary([float(row["exploitability"]) for row in values])
            result.append(
                {
                    "variant_id": variant_id,
                    "variant_label": VARIANTS[variant_id]["variant_label"],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_type": values[0]["checkpoint_type"],
                    "checkpoint_target_active_hours": values[0][
                        "checkpoint_target_active_hours"
                    ],
                    "checkpoint_target_nodes": values[0]["checkpoint_target_nodes"],
                    "mean_actual_active_hours": float(
                        np.mean([float(row["actual_active_hours"]) for row in values])
                    ),
                    "mean_nodes_touched": float(
                        np.mean([int(row["nodes_touched"]) for row in values])
                    ),
                    "mean_exploitability": float(stats["mean_ev"]),
                    "standard_deviation_exploitability": float(
                        stats["standard_deviation"]
                    ),
                    "standard_error_exploitability": float(stats["standard_error"]),
                    "ci95_lower_exploitability": float(stats["ci95_lower"]),
                    "ci95_upper_exploitability": float(stats["ci95_upper"]),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )
    return result


def _plot_trajectory(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    x_axis: str,
) -> None:
    time_axis = x_axis == "time"
    selected = [row for row in rows if row["checkpoint_type"] == "active_time"]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    for variant_id in VARIANT_ORDER:
        for seed in sorted(
            {int(row["seed"]) for row in selected if row["variant_id"] == variant_id}
        ):
            seed_rows = [
                row
                for row in selected
                if row["variant_id"] == variant_id and int(row["seed"]) == seed
            ]
            seed_rows.sort(
                key=lambda row: (
                    float(row["checkpoint_target_active_hours"])
                    if time_axis
                    else int(row["nodes_touched"])
                )
            )
            x = [
                (
                    float(row["checkpoint_target_active_hours"])
                    if time_axis
                    else int(row["nodes_touched"]) / 1_000_000.0
                )
                for row in seed_rows
            ]
            ax.plot(
                x,
                [float(row["exploitability"]) for row in seed_rows],
                color=COLOURS[variant_id],
                alpha=0.16,
                linewidth=0.9,
            )
        variant_summaries = [
            row
            for row in summaries
            if row["variant_id"] == variant_id
            and row["checkpoint_type"] == "active_time"
        ]
        variant_summaries.sort(
            key=lambda row: (
                float(row["checkpoint_target_active_hours"])
                if time_axis
                else float(row["mean_nodes_touched"])
            )
        )
        x = np.asarray(
            [
                (
                    float(row["checkpoint_target_active_hours"])
                    if time_axis
                    else float(row["mean_nodes_touched"]) / 1_000_000.0
                )
                for row in variant_summaries
            ]
        )
        mean = np.asarray([float(row["mean_exploitability"]) for row in variant_summaries])
        lower = np.asarray(
            [float(row["ci95_lower_exploitability"]) for row in variant_summaries]
        )
        upper = np.asarray(
            [float(row["ci95_upper_exploitability"]) for row in variant_summaries]
        )
        ax.plot(
            x,
            mean,
            color=COLOURS[variant_id],
            marker="o",
            markersize=4,
            linewidth=2,
            label=VARIANTS[variant_id]["variant_label"],
        )
        ax.fill_between(x, lower, upper, color=COLOURS[variant_id], alpha=0.12)
    ax.set_xlabel(
        "Active training time (hours)" if time_axis else "Training nodes touched (millions)"
    )
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    set_chart_title(
        ax,
        "Experiment 23 exploitability by "
        + ("active training time" if time_axis else "nodes touched"),
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _selection_metrics(
    rows: Sequence[Mapping[str, Any]], *, smoke: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    indexed = {
        (row["variant_id"], int(row["seed"]), row["checkpoint_id"]): row
        for row in rows
    }
    seed_rows = []
    for variant_id in VARIANT_ORDER:
        seeds = sorted(
            {int(row["seed"]) for row in rows if row["variant_id"] == variant_id}
        )
        for seed in seeds:
            time_rows = sorted(
                [
                    row
                    for row in rows
                    if row["variant_id"] == variant_id
                    and int(row["seed"]) == seed
                    and row["checkpoint_type"] == "active_time"
                ],
                key=lambda row: float(row["checkpoint_target_active_hours"]),
            )
            if smoke:
                late = time_rows
            else:
                late = [
                    row
                    for row in time_rows
                    if float(row["checkpoint_target_active_hours"]) >= 12.0
                ]
            values = np.asarray([float(row["exploitability"]) for row in late])
            adjacent = np.diff(values)
            node = next(
                row
                for row in rows
                if row["variant_id"] == variant_id
                and int(row["seed"]) == seed
                and row["checkpoint_type"] == "nodes"
            )
            seed_rows.append(
                {
                    "variant_id": variant_id,
                    "variant_label": VARIANTS[variant_id]["variant_label"],
                    "seed": seed,
                    "late_window_start_hours": float(
                        late[0]["checkpoint_target_active_hours"]
                    ),
                    "late_window_end_hours": float(
                        late[-1]["checkpoint_target_active_hours"]
                    ),
                    "late_window_mean_exploitability": float(np.mean(values)),
                    "late_window_checkpoint_standard_deviation": float(
                        np.std(values, ddof=1) if len(values) > 1 else 0.0
                    ),
                    "late_window_adjacent_rmssd": float(
                        np.sqrt(np.mean(np.square(adjacent))) if len(adjacent) else 0.0
                    ),
                    "late_window_max_deterioration": float(
                        max(0.0, float(np.max(adjacent))) if len(adjacent) else 0.0
                    ),
                    "late_window_improvement": float(values[0] - values[-1]),
                    "final_exploitability": float(values[-1]),
                    "node_target_exploitability": float(node["exploitability"]),
                    "time_to_node_target_hours": float(node["actual_active_hours"]),
                    "final_nodes_touched": int(time_rows[-1]["nodes_touched"]),
                    "peak_rss_mb": float(time_rows[-1]["peak_rss_mb"]),
                }
            )

    metric_names = (
        "late_window_mean_exploitability",
        "late_window_checkpoint_standard_deviation",
        "late_window_adjacent_rmssd",
        "late_window_max_deterioration",
        "late_window_improvement",
        "final_exploitability",
        "node_target_exploitability",
        "time_to_node_target_hours",
        "final_nodes_touched",
        "peak_rss_mb",
    )
    summary_rows = []
    for variant_id in VARIANT_ORDER:
        variants = [row for row in seed_rows if row["variant_id"] == variant_id]
        for metric in metric_names:
            stats = summary([float(row[metric]) for row in variants])
            summary_rows.append(
                {
                    "variant_id": variant_id,
                    "variant_label": VARIANTS[variant_id]["variant_label"],
                    "metric": metric,
                    "mean": float(stats["mean_ev"]),
                    "standard_deviation": float(stats["standard_deviation"]),
                    "standard_error": float(stats["standard_error"]),
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )

    by_seed = {(row["variant_id"], int(row["seed"])): row for row in seed_rows}
    paired_rows = []
    for candidate in VARIANT_ORDER[1:]:
        for metric in metric_names:
            effects = []
            for seed in sorted({int(row["seed"]) for row in seed_rows}):
                effect = float(by_seed[(candidate, seed)][metric]) - float(
                    by_seed[(FULL_ADAPTIVE, seed)][metric]
                )
                effects.append(effect)
            stats = summary(effects)
            paired_rows.append(
                {
                    "candidate_variant_id": candidate,
                    "candidate_variant_label": VARIANTS[candidate]["variant_label"],
                    "metric": metric,
                    "mean_candidate_minus_original": float(stats["mean_ev"]),
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "positive_seed_fraction": float(stats["positive_seed_fraction"]),
                    "two_sided_exact_sign_flip_p": float(
                        stats["two_sided_exact_sign_flip_p"]
                    ),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )
    return seed_rows, summary_rows, paired_rows


def _plot_stability(seed_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, metric, title, ylabel in (
        (
            axes[0],
            "late_window_mean_exploitability",
            "Late-window performance",
            "Mean exact exploitability",
        ),
        (
            axes[1],
            "late_window_adjacent_rmssd",
            "Late-window checkpoint volatility",
            "Adjacent-checkpoint RMSSD",
        ),
    ):
        stats = []
        for variant_id in VARIANT_ORDER:
            values = [
                float(row[metric])
                for row in seed_rows
                if row["variant_id"] == variant_id
            ]
            stats.append(summary(values))
        positions = np.arange(len(VARIANT_ORDER))
        ax.bar(
            positions,
            [row["mean_ev"] for row in stats],
            yerr=[row["standard_error"] for row in stats],
            color=[COLOURS[variant] for variant in VARIANT_ORDER],
            capsize=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [VARIANTS[variant]["variant_label"] for variant in VARIANT_ORDER],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        set_chart_title(ax, f"Experiment 23: {title}")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def aggregate_workers(
    *, workers_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    workers_root = Path(workers_root).resolve()
    output_dir = Path(output_dir).resolve()
    schedule = checkpoint_schedule(smoke=smoke)
    expected = {(variant, int(seed)) for variant in VARIANT_ORDER for seed in seeds}
    results = {}
    for path in workers_root.rglob("worker_result.json"):
        result = read_json(path)
        key = (str(result["variant_id"]), int(result["seed"]))
        if key in results:
            raise ValueError(f"Duplicate worker result for {key}")
        if (
            result.get("status") != "complete"
            or bool(result.get("smoke")) != bool(smoke)
            or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
        ):
            raise ValueError(f"Incomplete or mismatched worker: {path}")
        results[key] = (path, result)
    if set(results) != expected:
        raise ValueError(
            f"Worker set mismatch; missing={sorted(expected-set(results))}, "
            f"extra={sorted(set(results)-expected)}"
        )
    commits = {result["repository_commit"] for _, result in results.values()}
    if len(commits) != 1:
        raise ValueError(f"Workers used different commits: {sorted(commits)}")

    game = pyspiel.load_game(GAME_NAME)
    inventory, metrics, manifest, curves = [], [], [], []
    information_rows, beta_rows, critic_rows = [], [], []
    for key in sorted(results):
        result_path, result = results[key]
        root = result_path.parent
        for record in result["snapshots"]:
            snapshot = root / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Missing or corrupt snapshot: {snapshot}")
            loaded = load_policy(game, UCV_POLICY_LOADER_ID, snapshot)
            validate_policy_probabilities(game, loaded)
            tabular = policy.tabular_policy_from_callable(
                game, loaded.action_probabilities
            )
            nash_conv = float(exploitability.nash_conv(game, tabular))
            self_play = float(
                expected_game_score.policy_value(
                    game.new_initial_state(), [tabular, tabular]
                )[0]
            )
            inventory.append({**record, "path": str(snapshot.resolve())})
            metrics.append(
                {
                    "variant_id": key[0],
                    "variant_label": VARIANTS[key[0]]["variant_label"],
                    "seed": key[1],
                    "checkpoint_id": record["checkpoint_id"],
                    "checkpoint_type": record["checkpoint_type"],
                    "checkpoint_target_active_hours": record[
                        "checkpoint_target_active_hours"
                    ],
                    "checkpoint_target_active_seconds": record[
                        "checkpoint_target_active_seconds"
                    ],
                    "checkpoint_target_nodes": record["checkpoint_target_nodes"],
                    "actual_active_hours": float(record["active_seconds"]) / 3600.0,
                    "actual_active_seconds": float(record["active_seconds"]),
                    "nodes_touched": int(record["nodes_touched"]),
                    "completed_iteration": int(record["completed_iteration"]),
                    "nash_conv": nash_conv,
                    "exploitability": nash_conv / 2.0,
                    "self_play_value_player_0": self_play,
                    "peak_rss_mb": float(result["peak_rss_mb"]),
                    "snapshot_sha256": record["sha256"],
                    "snapshot_path": str(snapshot.resolve()),
                }
            )
        worker_curves = _read_csv(root / result["artifacts"]["checkpoint_curves"])
        worker_information = _read_csv(
            root / result["artifacts"]["information_action_diagnostics"]
        )
        worker_beta = _read_csv(root / result["artifacts"]["beta_histogram"])
        worker_critic = _read_csv(
            root / result["artifacts"]["critic_error_subsequent_local_regret"]
        )
        for collection in (worker_curves, worker_information, worker_beta, worker_critic):
            for row in collection:
                row.update({"variant_id": key[0], "seed": key[1]})
        curves.extend(worker_curves)
        information_rows.extend(worker_information)
        beta_rows.extend(worker_beta)
        critic_rows.extend(worker_critic)
        manifest.append(
            {
                "variant_id": key[0],
                "seed": key[1],
                "worker_result": str(result_path.resolve()),
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
            }
        )

    checkpoint_summaries = _checkpoint_summaries(metrics)
    seed_metrics, selection_summaries, paired = _selection_metrics(
        metrics, smoke=smoke
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", manifest)
    write_csv(output_dir / "snapshot_inventory.csv", inventory)
    write_csv(output_dir / "checkpoint_policy_metrics.csv", metrics)
    write_csv(output_dir / "checkpoint_summary.csv", checkpoint_summaries)
    write_csv(output_dir / "development_metrics_by_seed.csv", seed_metrics)
    write_csv(output_dir / "development_metric_summary.csv", selection_summaries)
    write_csv(output_dir / "paired_metrics_vs_original.csv", paired)
    write_csv(output_dir / "training_checkpoint_curves.csv", curves)
    write_csv(output_dir / "information_action_diagnostics.csv", information_rows)
    write_csv(output_dir / "beta_histogram.csv", beta_rows)
    write_csv(output_dir / "critic_error_subsequent_local_regret.csv", critic_rows)
    _plot_trajectory(
        metrics,
        checkpoint_summaries,
        output_dir / "exploitability_by_training_time.png",
        x_axis="time",
    )
    _plot_trajectory(
        metrics,
        checkpoint_summaries,
        output_dir / "exploitability_by_nodes_touched.png",
        x_axis="nodes",
    )
    _plot_stability(seed_metrics, output_dir / "late_window_performance_stability.png")

    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(results),
        "num_snapshots": len(inventory),
        "num_exact_policy_evaluations": len(metrics),
        "repository_commit": next(iter(commits)),
        "contract": contract_manifest(),
        "selection_metrics": selection_summaries,
        "paired_metrics_vs_original": paired,
        "decision_note": (
            "This is development evidence. Prefer low mean exploitability across 12--24h, "
            "low adjacent-checkpoint volatility, continued 12--24h improvement, and lower "
            "time-to-15M. Do not select a result-dependent best checkpoint. Any promoted "
            "architecture requires a fresh-seed confirmatory rerun."
        ),
        "inferential_note": (
            "Training seed is the inferential unit. Four paired development seeds are "
            "primarily effect-estimation evidence; the minimum two-sided exact sign-flip "
            "p-value is 0.125."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers"]
