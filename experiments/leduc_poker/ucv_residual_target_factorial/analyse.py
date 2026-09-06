"""Exact policy, stability, and factorial analysis for Experiment 25."""

from __future__ import annotations

import csv
from collections import defaultdict
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
    AVERAGED_TARGET_ONLY,
    COMBINED,
    CONTROL,
    GAME_NAME,
    LATE_WINDOW_START_HOURS,
    RESIDUAL_ONLY,
    VARIANTS,
    VARIANT_ORDER,
    checkpoint_schedule,
    contract_manifest,
)


COLOURS = {
    CONTROL: "#4c78a8",
    RESIDUAL_ONLY: "#f58518",
    AVERAGED_TARGET_ONLY: "#54a24b",
    COMBINED: "#b279a2",
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
    summaries = []
    for variant_id in VARIANT_ORDER:
        ids = sorted(
            {row["checkpoint_id"] for row in rows if row["variant_id"] == variant_id},
            key=lambda checkpoint_id: (
                grouped[(variant_id, checkpoint_id)][0]["checkpoint_type"] == "nodes",
                float(
                    grouped[(variant_id, checkpoint_id)][0][
                        "checkpoint_target_active_hours"
                    ]
                    or 0.0
                ),
            ),
        )
        for checkpoint_id in ids:
            values = grouped[(variant_id, checkpoint_id)]
            stats = summary([float(row["exploitability"]) for row in values])
            summaries.append(
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
    return summaries


def _plot_trajectory(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    x_axis: str,
) -> None:
    time_axis = x_axis == "time"
    selected = [row for row in rows if row["checkpoint_type"] == "active_time"]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for variant_id in VARIANT_ORDER:
        variant_rows = [row for row in selected if row["variant_id"] == variant_id]
        for seed in sorted({int(row["seed"]) for row in variant_rows}):
            seed_rows = sorted(
                (row for row in variant_rows if int(row["seed"]) == seed),
                key=lambda row: (
                    float(row["checkpoint_target_active_hours"])
                    if time_axis
                    else int(row["nodes_touched"])
                ),
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
                alpha=0.15,
                linewidth=0.8,
            )
        means = sorted(
            (
                row
                for row in summaries
                if row["variant_id"] == variant_id
                and row["checkpoint_type"] == "active_time"
            ),
            key=lambda row: (
                float(row["checkpoint_target_active_hours"])
                if time_axis
                else float(row["mean_nodes_touched"])
            ),
        )
        x = np.asarray(
            [
                (
                    float(row["checkpoint_target_active_hours"])
                    if time_axis
                    else float(row["mean_nodes_touched"]) / 1_000_000.0
                )
                for row in means
            ]
        )
        mean = np.asarray([float(row["mean_exploitability"]) for row in means])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in means])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in means])
        ax.plot(
            x,
            mean,
            color=COLOURS[variant_id],
            marker="o",
            markersize=4,
            linewidth=2.1,
            label=VARIANTS[variant_id]["variant_label"],
        )
        ax.fill_between(x, lower, upper, color=COLOURS[variant_id], alpha=0.11)
    ax.set_xlabel(
        "Active training time (hours)" if time_axis else "Training nodes touched (millions)"
    )
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    if time_axis and selected and max(float(row["checkpoint_target_active_hours"]) for row in selected) >= 2:
        ax.set_xticks(np.arange(2, 37, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    set_chart_title(
        ax,
        "Experiment 25 exploitability by "
        + ("active training time" if time_axis else "nodes touched"),
    )
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _development_metrics(
    rows: Sequence[Mapping[str, Any]], *, smoke: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    seed_rows = []
    for variant_id in VARIANT_ORDER:
        seeds = sorted({int(row["seed"]) for row in rows if row["variant_id"] == variant_id})
        for seed in seeds:
            time_rows = sorted(
                (
                    row
                    for row in rows
                    if row["variant_id"] == variant_id
                    and int(row["seed"]) == seed
                    and row["checkpoint_type"] == "active_time"
                ),
                key=lambda row: float(row["checkpoint_target_active_hours"]),
            )
            late = (
                time_rows
                if smoke
                else [
                    row
                    for row in time_rows
                    if float(row["checkpoint_target_active_hours"])
                    >= LATE_WINDOW_START_HOURS
                ]
            )
            values = np.asarray([float(row["exploitability"]) for row in late])
            adjacent = np.diff(values)
            hours = np.asarray(
                [float(row["checkpoint_target_active_hours"]) for row in late]
            )
            slope = (
                float(np.polyfit(hours, values, 1)[0]) if len(values) > 1 else 0.0
            )
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
                    "late_window_start_hours": float(hours[0]),
                    "late_window_end_hours": float(hours[-1]),
                    "late_window_mean_exploitability": float(np.mean(values)),
                    "late_window_slope_per_hour": slope,
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
        "late_window_slope_per_hour",
        "late_window_adjacent_rmssd",
        "late_window_max_deterioration",
        "late_window_improvement",
        "final_exploitability",
        "node_target_exploitability",
        "time_to_node_target_hours",
        "final_nodes_touched",
        "peak_rss_mb",
    )
    summaries = []
    for variant_id in VARIANT_ORDER:
        selected = [row for row in seed_rows if row["variant_id"] == variant_id]
        for metric in metric_names:
            stats = summary([float(row[metric]) for row in selected])
            summaries.append(
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

    indexed = {(row["variant_id"], int(row["seed"])): row for row in seed_rows}
    paired = []
    for variant_id in VARIANT_ORDER[1:]:
        for metric in metric_names:
            effects = [
                float(indexed[(variant_id, seed)][metric])
                - float(indexed[(CONTROL, seed)][metric])
                for seed in sorted({int(row["seed"]) for row in seed_rows})
            ]
            stats = summary(effects)
            paired.append(
                {
                    "candidate_variant_id": variant_id,
                    "candidate_variant_label": VARIANTS[variant_id]["variant_label"],
                    "metric": metric,
                    "mean_candidate_minus_control": float(stats["mean_ev"]),
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "positive_seed_fraction": float(stats["positive_seed_fraction"]),
                    "two_sided_exact_sign_flip_p": float(
                        stats["two_sided_exact_sign_flip_p"]
                    ),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )
    return seed_rows, summaries, paired


def _factorial_effects(seed_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    metrics = (
        "late_window_mean_exploitability",
        "late_window_slope_per_hour",
        "late_window_adjacent_rmssd",
        "late_window_max_deterioration",
        "late_window_improvement",
        "final_exploitability",
        "node_target_exploitability",
        "time_to_node_target_hours",
        "final_nodes_touched",
    )
    indexed = {(row["variant_id"], int(row["seed"])): row for row in seed_rows}
    seeds = sorted({int(row["seed"]) for row in seed_rows})
    rows = []
    for seed in seeds:
        for metric in metrics:
            a = float(indexed[(CONTROL, seed)][metric])
            b = float(indexed[(RESIDUAL_ONLY, seed)][metric])
            c = float(indexed[(AVERAGED_TARGET_ONLY, seed)][metric])
            d = float(indexed[(COMBINED, seed)][metric])
            for effect, value in (
                ("residual_regret_main_effect", 0.5 * ((b - a) + (d - c))),
                ("averaged_critic_target_main_effect", 0.5 * ((c - a) + (d - b))),
                ("interaction", (d - c) - (b - a)),
            ):
                rows.append(
                    {"seed": seed, "metric": metric, "effect": effect, "value": value}
                )
    summaries = []
    for metric in metrics:
        for effect in (
            "residual_regret_main_effect",
            "averaged_critic_target_main_effect",
            "interaction",
        ):
            values = [
                float(row["value"])
                for row in rows
                if row["metric"] == metric and row["effect"] == effect
            ]
            stats = summary(values)
            summaries.append(
                {
                    "metric": metric,
                    "effect": effect,
                    "mean_effect": float(stats["mean_ev"]),
                    "standard_deviation": float(stats["standard_deviation"]),
                    "standard_error": float(stats["standard_error"]),
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "positive_seed_fraction": float(stats["positive_seed_fraction"]),
                    "two_sided_exact_sign_flip_p": float(
                        stats["two_sided_exact_sign_flip_p"]
                    ),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )
    return rows, summaries


def _plot_stability(seed_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    for ax, metric, title, ylabel in (
        (
            axes[0],
            "late_window_mean_exploitability",
            "Hours 24–36 performance",
            "Mean exact exploitability",
        ),
        (
            axes[1],
            "late_window_adjacent_rmssd",
            "Hours 24–36 volatility",
            "Adjacent-checkpoint RMSSD",
        ),
    ):
        stats = [
            summary(
                [float(row[metric])
                for row in seed_rows
                if row["variant_id"] == variant_id]
            )
            for variant_id in VARIANT_ORDER
        ]
        positions = np.arange(len(VARIANT_ORDER))
        ax.bar(
            positions,
            [row["mean_ev"] for row in stats],
            yerr=[row["standard_error"] for row in stats],
            color=[COLOURS[variant_id] for variant_id in VARIANT_ORDER],
            capsize=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [VARIANTS[variant_id]["variant_label"] for variant_id in VARIANT_ORDER],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_policy_diagnostics(curves: Sequence[Mapping[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for variant_id in VARIANT_ORDER:
        selected = [row for row in curves if row["variant_id"] == variant_id]
        grouped = defaultdict(list)
        for row in selected:
            grouped[int(row["checkpoint_index"])].append(row)
        indices = sorted(grouped)
        hours = [
            float(np.mean([float(row["wall_clock_seconds"]) for row in grouped[index]]))
            / 3600.0
            for index in indices
        ]
        exact = [
            np.mean(
                [float(row["exact_average_exploitability"]) for row in grouped[index]]
            )
            for index in indices
        ]
        gap = [
            np.mean(
                [float(row["average_policy_distillation_gap"]) for row in grouped[index]]
            )
            for index in indices
        ]
        axes[0].plot(hours, exact, color=COLOURS[variant_id], label=VARIANTS[variant_id]["variant_label"])
        axes[1].plot(hours, gap, color=COLOURS[variant_id], label=VARIANTS[variant_id]["variant_label"])
    axes[0].set_xlabel("Active training time (hours)")
    axes[0].set_ylabel("Exact tabular-average exploitability")
    axes[1].set_xlabel("Active training time (hours)")
    axes[1].set_ylabel("Neural minus exact-average exploitability")
    for ax, title in zip(axes, ("Underlying average strategy", "Average-policy distillation gap")):
        ax.grid(axis="y", alpha=0.25)
        set_chart_title(ax, title)
    axes[0].legend()
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
    inventory, state_inventory, metrics, manifest, curves = [], [], [], [], []
    information_rows, beta_rows, critic_rows, q_oracle_rows = [], [], [], []
    for key in sorted(results):
        result_path, result = results[key]
        root = result_path.parent
        for record in result["snapshots"]:
            snapshot = root / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Missing or corrupt snapshot: {snapshot}")
            loaded = load_policy(game, UCV_POLICY_LOADER_ID, snapshot)
            validate_policy_probabilities(game, loaded)
            tabular = policy.tabular_policy_from_callable(game, loaded.action_probabilities)
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
        for record in result["training_states"]:
            state = root / record["relative_path"]
            if not state.is_file() or sha256(state) != record["sha256"]:
                raise ValueError(f"Missing or corrupt training state: {state}")
            state_inventory.append(
                {"variant_id": key[0], "seed": key[1], **record, "path": str(state.resolve())}
            )
        artifact_targets = (
            ("checkpoint_curves", curves),
            ("information_action_diagnostics", information_rows),
            ("beta_histogram", beta_rows),
            ("critic_error_subsequent_local_regret", critic_rows),
            ("exact_q_oracle_diagnostics", q_oracle_rows),
        )
        for artifact_id, destination in artifact_targets:
            rows = _read_csv(root / result["artifacts"][artifact_id])
            for row in rows:
                row.update({"variant_id": key[0], "seed": key[1]})
            destination.extend(rows)
        manifest.append(
            {
                "variant_id": key[0],
                "seed": key[1],
                "worker_result": str(result_path.resolve()),
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "resumed_from_training_state": result["resumed_from_training_state"],
            }
        )

    checkpoint_summaries = _checkpoint_summaries(metrics)
    seed_metrics, development_summaries, paired = _development_metrics(
        metrics, smoke=smoke
    )
    factorial_rows, factorial_summaries = _factorial_effects(seed_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", manifest)
    write_csv(output_dir / "snapshot_inventory.csv", inventory)
    write_csv(output_dir / "training_state_inventory.csv", state_inventory)
    write_csv(output_dir / "checkpoint_policy_metrics.csv", metrics)
    write_csv(output_dir / "checkpoint_summary.csv", checkpoint_summaries)
    write_csv(output_dir / "development_metrics_by_seed.csv", seed_metrics)
    write_csv(output_dir / "development_metric_summary.csv", development_summaries)
    write_csv(output_dir / "paired_metrics_vs_control.csv", paired)
    write_csv(output_dir / "factorial_effects_by_seed.csv", factorial_rows)
    write_csv(output_dir / "factorial_effect_summary.csv", factorial_summaries)
    write_csv(output_dir / "training_checkpoint_curves.csv", curves)
    write_csv(output_dir / "information_action_diagnostics.csv", information_rows)
    write_csv(output_dir / "beta_histogram.csv", beta_rows)
    write_csv(output_dir / "critic_error_subsequent_local_regret.csv", critic_rows)
    write_csv(output_dir / "exact_q_oracle_diagnostics.csv", q_oracle_rows)
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
    _plot_policy_diagnostics(curves, output_dir / "average_policy_diagnostics.png")

    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(results),
        "num_snapshots": len(inventory),
        "num_training_states": len(state_inventory),
        "num_exact_policy_evaluations": len(metrics),
        "repository_commit": next(iter(commits)),
        "contract": contract_manifest(),
        "factorial_effect_summary": factorial_summaries,
        "paired_metrics_vs_control": paired,
        "decision_note": (
            "This is paired development evidence. Select using hours 24--36 mean "
            "exploitability, late slope, adjacent-checkpoint RMSSD, worst rebound, "
            "equal-node performance, and throughput. Inspect the exact-average curve "
            "and distillation gap before attributing volatility to regret learning."
        ),
        "inferential_note": (
            "Training seed is the inferential unit. Three paired seeds estimate effects "
            "and interactions but cannot provide a two-sided exact sign-flip p-value "
            "below 0.25. A promoted architecture requires fresh confirmation."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers"]
