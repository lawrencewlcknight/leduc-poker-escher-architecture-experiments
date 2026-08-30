"""Exact convergence analysis for Experiment 21 policy checkpoints."""

from __future__ import annotations

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
    ALGORITHMS,
    ALGORITHM_ORDER,
    GAME_NAME,
    checkpoint_schedule,
)


COLOURS = {
    ALGORITHM_ORDER[0]: "#1f77b4",
    ALGORITHM_ORDER[1]: "#d62728",
}


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
        "deep_cfr_ev_as_player_0": as_player_0,
        "deep_cfr_ev_as_player_1": as_player_1,
        "deep_cfr_seat_averaged_ev": 0.5 * (as_player_0 + as_player_1),
    }


def _plot_time(
    metric_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for algorithm_id in ALGORITHM_ORDER:
        colour = COLOURS[algorithm_id]
        for seed in sorted(
            {int(row["seed"]) for row in metric_rows if row["algorithm_id"] == algorithm_id}
        ):
            rows = sorted(
                (
                    row
                    for row in metric_rows
                    if row["algorithm_id"] == algorithm_id and int(row["seed"]) == seed
                ),
                key=lambda row: float(row["actual_active_hours"]),
            )
            ax.plot(
                [float(row["actual_active_hours"]) for row in rows],
                [float(row["exploitability"]) for row in rows],
                color=colour,
                alpha=0.16,
                linewidth=0.9,
            )
        rows = sorted(
            (row for row in summary_rows if row["algorithm_id"] == algorithm_id),
            key=lambda row: float(row["checkpoint_target_active_hours"]),
        )
        x = np.asarray(
            [float(row["checkpoint_target_active_hours"]) for row in rows], dtype=float
        )
        mean = np.asarray([float(row["mean_exploitability"]) for row in rows])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in rows])
        ax.plot(
            x,
            mean,
            color=colour,
            marker="o",
            markersize=4,
            linewidth=2.1,
            label=ALGORITHMS[algorithm_id]["algorithm_label"],
        )
        ax.fill_between(x, lower, upper, color=colour, alpha=0.13)
    ax.set_xlabel("Active training time (hours)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    maximum_target_hours = max(
        float(row["checkpoint_target_active_hours"]) for row in summary_rows
    )
    if maximum_target_hours >= 2.0:
        ax.set_xticks(np.arange(2, maximum_target_hours + 0.1, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    set_chart_title(ax, "Exploitability by active training time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_nodes(
    metric_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for algorithm_id in ALGORITHM_ORDER:
        colour = COLOURS[algorithm_id]
        for seed in sorted(
            {int(row["seed"]) for row in metric_rows if row["algorithm_id"] == algorithm_id}
        ):
            rows = sorted(
                (
                    row
                    for row in metric_rows
                    if row["algorithm_id"] == algorithm_id and int(row["seed"]) == seed
                ),
                key=lambda row: int(row["nodes_touched"]),
            )
            ax.plot(
                [int(row["nodes_touched"]) / 1_000_000.0 for row in rows],
                [float(row["exploitability"]) for row in rows],
                color=colour,
                alpha=0.16,
                linewidth=0.9,
            )
        rows = sorted(
            (row for row in summary_rows if row["algorithm_id"] == algorithm_id),
            key=lambda row: float(row["mean_nodes_touched"]),
        )
        x = np.asarray([float(row["mean_nodes_touched"]) / 1_000_000.0 for row in rows])
        mean = np.asarray([float(row["mean_exploitability"]) for row in rows])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in rows])
        ax.plot(
            x,
            mean,
            color=colour,
            marker="o",
            markersize=4,
            linewidth=2.1,
            label=ALGORITHMS[algorithm_id]["algorithm_label"],
        )
        ax.fill_between(x, lower, upper, color=colour, alpha=0.13)
    ax.set_xlabel("Training nodes touched (millions)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    set_chart_title(ax, "Exploitability by training nodes touched")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _checkpoint_summaries(
    metric_rows: Sequence[Mapping[str, Any]], *, smoke: bool
) -> list[dict]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["algorithm_id"]), str(row["checkpoint_id"]))].append(row)
    rows = []
    for algorithm_id in ALGORITHM_ORDER:
        for checkpoint in checkpoint_schedule(smoke=smoke):
            checkpoint_id = str(checkpoint["checkpoint_id"])
            values = grouped.get((algorithm_id, checkpoint_id), [])
            if not values:
                continue
            stats = summary([float(row["exploitability"]) for row in values])
            rows.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_target_active_hours": float(
                        values[0]["checkpoint_target_active_hours"]
                    ),
                    "mean_actual_active_hours": float(
                        np.mean([float(row["actual_active_hours"]) for row in values])
                    ),
                    "mean_nodes_touched": float(
                        np.mean([int(row["nodes_touched"]) for row in values])
                    ),
                    "n_seeds": int(stats["n_seeds"]),
                    "mean_exploitability": float(stats["mean_ev"]),
                    "standard_deviation_exploitability": float(
                        stats["standard_deviation"]
                    ),
                    "standard_error_exploitability": float(stats["standard_error"]),
                    "ci95_lower_exploitability": float(stats["ci95_lower"]),
                    "ci95_upper_exploitability": float(stats["ci95_upper"]),
                }
            )
    return rows


def _late_window_changes(
    metric_rows: Sequence[Mapping[str, Any]], *, smoke: bool
) -> tuple[list[dict], list[dict]]:
    schedule = checkpoint_schedule(smoke=smoke)
    if smoke:
        windows = ((schedule[0]["checkpoint_id"], schedule[-1]["checkpoint_id"]),)
    else:
        windows = (
            ("time_24h", "time_30h"),
            ("time_30h", "time_36h"),
            ("time_24h", "time_36h"),
        )
    indexed = {
        (str(row["algorithm_id"]), int(row["seed"]), str(row["checkpoint_id"])): row
        for row in metric_rows
    }
    seed_rows = []
    summary_rows = []
    for algorithm_id in ALGORITHM_ORDER:
        seeds = sorted(
            {int(row["seed"]) for row in metric_rows if row["algorithm_id"] == algorithm_id}
        )
        for earlier, later in windows:
            changes = []
            for seed in seeds:
                before = indexed[(algorithm_id, seed, str(earlier))]
                after = indexed[(algorithm_id, seed, str(later))]
                # Positive values represent an improvement (lower exploitability).
                change = float(before["exploitability"]) - float(after["exploitability"])
                changes.append(change)
                seed_rows.append(
                    {
                        "algorithm_id": algorithm_id,
                        "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                        "seed": seed,
                        "earlier_checkpoint_id": earlier,
                        "later_checkpoint_id": later,
                        "earlier_exploitability": float(before["exploitability"]),
                        "later_exploitability": float(after["exploitability"]),
                        "exploitability_improvement": change,
                    }
                )
            stats = summary(changes)
            summary_rows.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "earlier_checkpoint_id": earlier,
                    "later_checkpoint_id": later,
                    "mean_exploitability_improvement": float(stats["mean_ev"]),
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
    return seed_rows, summary_rows


def aggregate_workers(
    *,
    workers_root: Path,
    seeds: Sequence[int],
    output_dir: Path,
    smoke: bool = False,
) -> dict:
    workers_root = Path(workers_root).resolve()
    output_dir = Path(output_dir).resolve()
    schedule = checkpoint_schedule(smoke=smoke)
    expected = {
        (algorithm_id, int(seed))
        for algorithm_id in ALGORITHM_ORDER
        for seed in seeds
    }
    worker_results = {}
    for result_path in workers_root.rglob("worker_result.json"):
        result = read_json(result_path)
        key = (str(result["algorithm_id"]), int(result["seed"]))
        if key in worker_results:
            raise ValueError(f"Duplicate worker result for {key}")
        if result.get("status") != "complete" or bool(result.get("smoke")) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker result: {result_path}")
        if tuple(result.get("checkpoint_schedule", ())) != tuple(schedule):
            raise ValueError(f"Worker checkpoint contract mismatch: {result_path}")
        worker_results[key] = (result_path, result)
    if set(worker_results) != expected:
        raise ValueError(
            f"Worker set mismatch; missing={sorted(expected - set(worker_results))}, "
            f"extra={sorted(set(worker_results) - expected)}"
        )

    inventory = []
    worker_manifest = []
    for key in sorted(worker_results):
        result_path, result = worker_results[key]
        records = list(result["snapshots"])
        expected_ids = [str(row["checkpoint_id"]) for row in schedule]
        if [str(row["checkpoint_id"]) for row in records] != expected_ids:
            raise ValueError(f"Snapshot schedule mismatch in {result_path}")
        for record in records:
            if record.get("repository_commit") != result.get("repository_commit"):
                raise ValueError(f"Snapshot/worker commit mismatch in {result_path}")
            snapshot_path = result_path.parent / record["relative_path"]
            if not snapshot_path.is_file():
                raise FileNotFoundError(snapshot_path)
            if sha256(snapshot_path) != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {snapshot_path}")
            inventory.append({**record, "path": str(snapshot_path.resolve())})
        worker_manifest.append(
            {
                "algorithm_id": key[0],
                "seed": key[1],
                "result_path": str(result_path),
                "repository_commit": result["repository_commit"],
            }
        )

    game = pyspiel.load_game(GAME_NAME)
    policies = {}
    metric_rows = []
    for row in inventory:
        algorithm_id = str(row["algorithm_id"])
        seed = int(row["seed"])
        checkpoint_id = str(row["checkpoint_id"])
        loaded = load_policy(game, algorithm_id, row["path"])
        validate_policy_probabilities(game, loaded)
        tabular = policy.tabular_policy_from_callable(game, loaded.action_probabilities)
        policies[(algorithm_id, seed, checkpoint_id)] = tabular
        nash_conv = float(exploitability.nash_conv(game, tabular))
        self_play = float(
            expected_game_score.policy_value(
                game.new_initial_state(), [tabular, tabular]
            )[0]
        )
        metric_rows.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                "seed": seed,
                "checkpoint_id": checkpoint_id,
                "checkpoint_target_active_hours": float(
                    row["checkpoint_target_active_hours"]
                ),
                "checkpoint_target_active_seconds": float(
                    row["checkpoint_target_active_seconds"]
                ),
                "actual_active_hours": float(row["wall_clock_seconds"]) / 3600.0,
                "actual_active_seconds": float(row["wall_clock_seconds"]),
                "time_overshoot_seconds": float(row["wall_clock_seconds"])
                - float(row["checkpoint_target_active_seconds"]),
                "nodes_touched": int(row["nodes_touched"]),
                "completed_iteration": int(row["completed_iteration"]),
                "nash_conv": nash_conv,
                "exploitability": nash_conv / 2.0,
                "self_play_value_player_0": self_play,
                "snapshot_sha256": row["sha256"],
                "snapshot_path": row["path"],
            }
        )

    summary_rows = _checkpoint_summaries(metric_rows, smoke=smoke)
    late_seed_rows, late_summary_rows = _late_window_changes(metric_rows, smoke=smoke)
    final_checkpoint_id = str(schedule[-1]["checkpoint_id"])
    same_seed_h2h = []
    for seed in seeds:
        same_seed_h2h.append(
            {
                "seed": int(seed),
                "checkpoint_id": final_checkpoint_id,
                **_exact_ev(
                    game,
                    policies[(ALGORITHM_ORDER[0], int(seed), final_checkpoint_id)],
                    policies[(ALGORITHM_ORDER[1], int(seed), final_checkpoint_id)],
                ),
            }
        )
    h2h_stats = summary(
        [float(row["deep_cfr_seat_averaged_ev"]) for row in same_seed_h2h]
    )
    cross_seed_h2h = []
    for deep_seed in seeds:
        for ucv_seed in seeds:
            cross_seed_h2h.append(
                {
                    "deep_cfr_seed": int(deep_seed),
                    "ucv_escher_seed": int(ucv_seed),
                    "checkpoint_id": final_checkpoint_id,
                    **_exact_ev(
                        game,
                        policies[(ALGORITHM_ORDER[0], int(deep_seed), final_checkpoint_id)],
                        policies[(ALGORITHM_ORDER[1], int(ucv_seed), final_checkpoint_id)],
                    ),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", worker_manifest)
    write_csv(output_dir / "snapshot_inventory.csv", inventory)
    write_csv(output_dir / "checkpoint_policy_metrics.csv", metric_rows)
    write_csv(output_dir / "checkpoint_summary.csv", summary_rows)
    write_csv(output_dir / "late_window_changes_by_seed.csv", late_seed_rows)
    write_csv(output_dir / "late_window_change_summary.csv", late_summary_rows)
    write_csv(output_dir / "final_same_seed_head_to_head.csv", same_seed_h2h)
    write_csv(output_dir / "final_cross_seed_head_to_head.csv", cross_seed_h2h)
    _plot_time(
        metric_rows,
        summary_rows,
        output_dir / "exploitability_by_training_time.png",
    )
    _plot_nodes(
        metric_rows,
        summary_rows,
        output_dir / "exploitability_by_nodes_touched.png",
    )
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(worker_results),
        "num_snapshots": len(inventory),
        "num_exact_policy_evaluations": len(metric_rows),
        "num_final_same_seed_head_to_head_effects": len(same_seed_h2h),
        "num_final_cross_seed_head_to_head_matchups": len(cross_seed_h2h),
        "seeds": [int(seed) for seed in seeds],
        "checkpoint_schedule": list(schedule),
        "final_checkpoint_id": final_checkpoint_id,
        "final_same_seed_head_to_head_inference": h2h_stats,
        "inferential_note": (
            "Training seed is the inferential unit. With five seeds the minimum "
            "two-sided exact sign-flip p-value is 0.0625; emphasize effects and intervals."
        ),
        "late_window_change_summary": late_summary_rows,
    }
    write_json(output_dir / "aggregate_manifest.json", result)
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = ["aggregate_workers"]
