"""Aggregate one of the four post-training experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

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
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (
    summary,
)  # noqa: E402

from .config import contract_manifest, method_config  # noqa: E402


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_workers(*, method_id: str, workers_root: Path,
                      seeds: Sequence[int], output_dir: Path, smoke: bool) -> dict:
    config = method_config(method_id, smoke=smoke)
    expected = {int(seed) for seed in seeds}
    workers = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("method_id") != method_id:
            continue
        seed = int(result["source_seed"])
        if seed in workers:
            raise ValueError(f"Duplicate {method_id} worker for seed {seed}")
        if result.get("status") != "complete" or bool(result["smoke"]) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker {path}")
        workers[seed] = (path, result)
    if set(workers) != expected:
        raise ValueError(f"Worker seeds differ: {sorted(workers)} != {sorted(expected)}")
    if len({result["repository_commit"] for _, result in workers.values()}) != 1:
        raise ValueError("Workers used different repository commits")

    metrics, manifest = [], []
    for seed in sorted(workers):
        path, result = workers[seed]
        metrics.extend(_read_csv(path.parent / result["artifacts"]["metrics"]))
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "source_sha256": result["source"]["sha256"],
                "peak_rss_mb": result["peak_rss_mb"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            }
        )

    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["arm_id"], int(row["update"]))].append(row)
    summaries = []
    for (arm_id, update), values in sorted(grouped.items()):
        for metric in (
            "exploitability", "mean_kl_from_blueprint",
            "ev_vs_blueprint_seat_averaged",
        ):
            stats = summary([float(row[metric]) for row in values])
            summaries.append(
                {
                    "arm_id": arm_id,
                    "update": update,
                    "metric": metric,
                    "mean": stats["mean_ev"],
                    "standard_error": stats["standard_error"],
                    "ci95_lower": stats["ci95_lower"],
                    "ci95_upper": stats["ci95_upper"],
                    "n_source_seeds": stats["n_seeds"],
                }
            )

    effects = []
    for arm_id in config["arms"]:
        by_seed = defaultdict(list)
        for row in metrics:
            if row["arm_id"] == arm_id:
                by_seed[int(row["source_seed"])].append(row)
        for seed, rows in sorted(by_seed.items()):
            ordered = sorted(rows, key=lambda row: int(row["update"]))
            baseline = float(ordered[0]["exploitability"])
            best = min(ordered, key=lambda row: float(row["exploitability"]))
            final = ordered[-1]
            effects.append(
                {
                    "arm_id": arm_id,
                    "source_seed": seed,
                    "baseline_exploitability": baseline,
                    "best_update": int(best["update"]),
                    "best_exploitability": float(best["exploitability"]),
                    "best_minus_baseline": float(best["exploitability"]) - baseline,
                    "final_update": int(final["update"]),
                    "final_exploitability": float(final["exploitability"]),
                    "final_minus_baseline": float(final["exploitability"]) - baseline,
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fine_tuning_metrics.csv", metrics)
    write_csv(output_dir / "fine_tuning_summary.csv", summaries)
    write_csv(output_dir / "seed_level_effects.csv", effects)
    write_csv(output_dir / "worker_manifest.csv", manifest)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for arm_id in config["arms"]:
        rows = sorted(
            (row for row in summaries if row["arm_id"] == arm_id
             and row["metric"] == "exploitability"),
            key=lambda row: int(row["update"]),
        )
        x = np.asarray([int(row["update"]) for row in rows])
        y = np.asarray([float(row["mean"]) for row in rows])
        lower = np.asarray([float(row["ci95_lower"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper"]) for row in rows])
        ax.plot(x, y, marker="o", label=arm_id)
        ax.fill_between(x, lower, upper, alpha=0.15)
    ax.set_xlabel("Post-training update")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        config["label"],
        algorithm="Experiment 29 UCV policy post-training",
        game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "exploitability_by_update.png", dpi=180)
    plt.close(fig)

    arm_effects = []
    for arm_id in config["arms"]:
        rows = [row for row in effects if row["arm_id"] == arm_id]
        for metric in ("best_minus_baseline", "final_minus_baseline"):
            stats = summary([float(row[metric]) for row in rows])
            arm_effects.append(
                {
                    "arm_id": arm_id, "metric": metric,
                    "mean": stats["mean_ev"],
                    "standard_error": stats["standard_error"],
                    "ci95_lower": stats["ci95_lower"],
                    "ci95_upper": stats["ci95_upper"],
                    "n_source_seeds": stats["n_seeds"],
                }
            )
    write_csv(output_dir / "arm_effect_summary.csv", arm_effects)
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "method_id": method_id,
        "num_source_seeds": len(seeds),
        "contract": contract_manifest(method_id),
        "outputs": {
            "metrics": "fine_tuning_metrics.csv",
            "summary": "fine_tuning_summary.csv",
            "seed_effects": "seed_level_effects.csv",
            "arm_effect_summary": "arm_effect_summary.csv",
            "chart": "exploitability_by_update.png",
            "manifest": "worker_manifest.csv",
        },
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = ["aggregate_workers"]
