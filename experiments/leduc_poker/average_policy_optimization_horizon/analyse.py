"""Aggregate Experiment 41 workers and create convergence diagnostics."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (  # noqa: E402
    read_json,
    write_csv,
    write_json,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    summary,
)

from .config import (  # noqa: E402
    EXPERIMENT_NAME,
    REQUIRED_CHECKPOINT_UPDATES,
    contract_manifest,
)


METRICS = (
    "exploitability",
    "change_from_archived_policy",
    "gap_to_empirical_reservoir",
    "gap_to_exact_tabular_average",
    "full_batch_cross_entropy",
    "mean_l1",
    "reach_weighted_l1",
    "mean_kl",
    "reach_weighted_kl",
)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summarise(rows, group_fields, metric_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        units = [int(row["source_seed"]) for row in values]
        if len(units) != len(set(units)):
            raise ValueError(f"Repeated source seed inside summary group {key}")
        identity = dict(zip(group_fields, key))
        for metric in metric_fields:
            stats = summary([float(row[metric]) for row in values])
            output.append(
                {
                    **identity,
                    "metric": metric,
                    "mean": stats["mean_ev"],
                    "standard_deviation": stats["standard_deviation"],
                    "standard_error": stats["standard_error"],
                    "ci95_lower": stats["ci95_lower"],
                    "ci95_upper": stats["ci95_upper"],
                    "n_source_seeds": stats["n_seeds"],
                }
            )
    return output


def aggregate_workers(*, workers_root: Path, seeds, output_dir: Path,
                      smoke: bool) -> dict:
    expected = {int(seed) for seed in seeds}
    workers = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != EXPERIMENT_NAME:
            continue
        seed = int(result["source_seed"])
        if seed in workers:
            raise ValueError(f"Duplicate Experiment 41 worker for seed {seed}")
        if result.get("status") != "complete" or bool(result["smoke"]) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker {path}")
        workers[seed] = (path, result)
    if set(workers) != expected:
        raise ValueError(f"Worker seeds differ: {sorted(workers)} != {sorted(expected)}")
    commits = {result["repository_commit"] for _, result in workers.values()}
    if len(commits) != 1:
        raise ValueError("Experiment 41 workers used different repository commits")

    metrics, manifest, baselines, snapshots = [], [], [], []
    for seed in sorted(workers):
        path, result = workers[seed]
        root = path.parent
        rows = _read_csv(root / result["artifacts"]["metrics"])
        observed = {int(row["optimizer_update"]) for row in rows}
        expected_updates = set(
            int(value) for value in result["runtime_schedule"][
                "diagnostic_updates"
            ]
        )
        if observed != expected_updates:
            raise ValueError(
                f"Worker {seed} updates differ: {sorted(observed)} != "
                f"{sorted(expected_updates)}"
            )
        metrics.extend(rows)
        for policy_id, value in result["baselines"].items():
            baselines.append(
                {
                    "source_seed": seed,
                    "policy_id": policy_id,
                    "exploitability": float(value),
                }
            )
        for snapshot in result["snapshots"]:
            snapshots.append({"source_seed": seed, **snapshot})
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "source_sha256": result["source"]["sha256"],
                "replay_rows": result["dataset"]["replay_rows"],
                "unique_information_states": result["dataset"][
                    "unique_information_states"
                ],
                "peak_rss_mb": result["peak_rss_mb"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            }
        )

    summaries = _summarise(metrics, ("optimizer_update",), METRICS)
    baseline_summary = _summarise(
        baselines, ("policy_id",), ("exploitability",)
    )
    by_seed_update = {
        (int(row["source_seed"]), int(row["optimizer_update"])): row
        for row in metrics
    }
    updates = sorted({int(row["optimizer_update"]) for row in metrics})
    effects = []
    for seed in sorted(expected):
        initial = float(by_seed_update[(seed, 0)]["exploitability"])
        for update in updates:
            value = float(by_seed_update[(seed, update)]["exploitability"])
            effects.append(
                {
                    "source_seed": seed,
                    "optimizer_update": update,
                    "exploitability": value,
                    "change_from_update_0": value - initial,
                }
            )
    effect_summary = _summarise(
        effects, ("optimizer_update",), ("change_from_update_0",)
    )

    required = (
        tuple(sorted({int(row["optimizer_update"]) for row in metrics}))
        if smoke else (0,) + REQUIRED_CHECKPOINT_UPDATES
    )
    intervals = []
    for seed in sorted(expected):
        for previous, current in zip(required, required[1:]):
            previous_value = float(
                by_seed_update[(seed, previous)]["exploitability"]
            )
            current_value = float(
                by_seed_update[(seed, current)]["exploitability"]
            )
            intervals.append(
                {
                    "source_seed": seed,
                    "previous_update": previous,
                    "optimizer_update": current,
                    "exploitability_change": current_value - previous_value,
                    "exploitability_improvement": previous_value - current_value,
                }
            )
    interval_summary = _summarise(
        intervals, ("previous_update", "optimizer_update"),
        ("exploitability_change", "exploitability_improvement"),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fit_metrics.csv", metrics)
    write_csv(output_dir / "fit_summary.csv", summaries)
    write_csv(output_dir / "baseline_metrics.csv", baselines)
    write_csv(output_dir / "baseline_summary.csv", baseline_summary)
    write_csv(output_dir / "paired_effects_by_seed.csv", effects)
    write_csv(output_dir / "paired_effect_summary.csv", effect_summary)
    write_csv(output_dir / "interval_effects_by_seed.csv", intervals)
    write_csv(output_dir / "interval_effect_summary.csv", interval_summary)
    write_csv(output_dir / "snapshot_inventory.csv", snapshots)
    write_csv(output_dir / "worker_manifest.csv", manifest)

    def series(metric):
        return sorted(
            (row for row in summaries if row["metric"] == metric),
            key=lambda row: int(row["optimizer_update"]),
        )

    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    values = series("exploitability")
    x = np.asarray([int(row["optimizer_update"]) for row in values])
    y = np.asarray([float(row["mean"]) for row in values])
    lower = np.asarray([float(row["ci95_lower"]) for row in values])
    upper = np.asarray([float(row["ci95_upper"]) for row in values])
    ax.plot(x, y, marker="o", label="Continued neural policy fit")
    ax.fill_between(x, lower, upper, alpha=0.15)
    for row in baseline_summary:
        label = {
            "archived_neural_policy": "Archived Experiment 29 policy",
            "empirical_reservoir_policy": "Empirical reservoir policy",
            "exact_tabular_average": "Exact tabular average",
        }[row["policy_id"]]
        ax.axhline(float(row["mean"]), linestyle="--", linewidth=1.2, label=label)
    ax.set_xlabel("Additional full-batch optimiser updates")
    ax.set_ylabel("Exact exploitability (NashConv / 2; analysis only)")
    set_chart_title(
        ax, "Average-policy optimisation horizon",
        algorithm="Experiment 29 UCV average policy", game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "exploitability_by_optimizer_update.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    for metric, label in (
        ("gap_to_empirical_reservoir", "Neural minus empirical reservoir"),
        ("gap_to_exact_tabular_average", "Neural minus exact tabular average"),
    ):
        values = series(metric)
        ax.plot(
            [int(row["optimizer_update"]) for row in values],
            [float(row["mean"]) for row in values], marker="o", label=label,
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Additional full-batch optimiser updates")
    ax.set_ylabel("Exploitability gap")
    set_chart_title(
        ax, "Average-policy approximation gap",
        algorithm="Experiment 29 UCV average policy", game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "distillation_gap_by_optimizer_update.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    values = series("full_batch_cross_entropy")
    ax.plot(
        [int(row["optimizer_update"]) for row in values],
        [float(row["mean"]) for row in values], marker="o",
    )
    ax.set_xlabel("Additional full-batch optimiser updates")
    ax.set_ylabel("Uniform grouped soft-target cross-entropy")
    set_chart_title(
        ax, "Replay fitting objective",
        algorithm="Experiment 29 UCV average policy", game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_objective_by_optimizer_update.png", dpi=180)
    plt.close(fig)

    exploitability_by_update = {
        int(row["optimizer_update"]): float(row["mean"])
        for row in summaries if row["metric"] == "exploitability"
    }
    best_update = min(exploitability_by_update, key=exploitability_by_update.get)
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_source_seeds": len(expected),
        "contract": contract_manifest(),
        "descriptive_convergence": {
            "best_observed_update": best_update,
            "best_observed_mean_exploitability": exploitability_by_update[best_update],
            "final_update": max(exploitability_by_update),
            "final_mean_exploitability": exploitability_by_update[
                max(exploitability_by_update)
            ],
            "selection_warning": (
                "Best observed update is descriptive and was not used to select "
                "or stop any worker."
            ),
        },
        "outputs": {
            "metrics": "fit_metrics.csv",
            "summary": "fit_summary.csv",
            "baseline_metrics": "baseline_metrics.csv",
            "baseline_summary": "baseline_summary.csv",
            "paired_effects": "paired_effects_by_seed.csv",
            "paired_effect_summary": "paired_effect_summary.csv",
            "interval_effects": "interval_effects_by_seed.csv",
            "interval_effect_summary": "interval_effect_summary.csv",
            "snapshot_inventory": "snapshot_inventory.csv",
            "worker_manifest": "worker_manifest.csv",
            "exploitability_chart": "exploitability_by_optimizer_update.png",
            "gap_chart": "distillation_gap_by_optimizer_update.png",
            "objective_chart": "training_objective_by_optimizer_update.png",
        },
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = ["aggregate_workers"]
