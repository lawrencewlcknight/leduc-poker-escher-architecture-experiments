"""Aggregate Experiment 44's paired historical-policy study."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import (  # noqa: E402
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    summary,
)

from .config import (  # noqa: E402
    BOUNDED_CAPACITIES,
    CANDIDATE_ID,
    EMPIRICAL_ID,
    EXACT_ID,
    INCUMBENT_NEURAL_ID,
    ORDINARY_NEURAL_ID,
    checkpoint_schedule,
    contract_manifest,
)


PRIMARY_ORDER = (
    CANDIDATE_ID,
    INCUMBENT_NEURAL_ID,
    ORDINARY_NEURAL_ID,
    EMPIRICAL_ID,
    EXACT_ID,
)
COLOURS = {
    CANDIDATE_ID: "#2ca02c",
    INCUMBENT_NEURAL_ID: "#9467bd",
    ORDINARY_NEURAL_ID: "#7f7f7f",
    EMPIRICAL_ID: "#8c564b",
    EXACT_ID: "#17becf",
    **{
        f"historical_reservoir_{capacity}": plt.cm.Blues(
            0.35 + 0.55 * index / max(1, len(BOUNDED_CAPACITIES) - 1)
        )
        for index, capacity in enumerate(BOUNDED_CAPACITIES)
    },
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_workers(workers_root: Path, seeds: Sequence[int], smoke: bool):
    expected = {int(seed) for seed in seeds}
    schedule = checkpoint_schedule(smoke=smoke)
    found = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        seed = int(result["seed"])
        if seed in found:
            raise ValueError(f"Duplicate Experiment 44 worker for seed {seed}")
        if (
            result.get("status") != "complete"
            or result.get("candidate_id") != CANDIDATE_ID
            or bool(result.get("smoke")) != bool(smoke)
            or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
        ):
            raise ValueError(f"Incomplete or mismatched worker: {path}")
        found[seed] = (path, result)
    if set(found) != expected:
        raise ValueError(
            f"Experiment 44 workers differ; missing={sorted(expected-set(found))}, "
            f"extra={sorted(set(found)-expected)}"
        )
    if len({result["repository_commit"] for _, result in found.values()}) != 1:
        raise ValueError("Workers used different repository commits")
    return found


def _collect(workers: Mapping[int, tuple[Path, Mapping]]):
    metrics, inventory, manifests = [], [], []
    for seed in sorted(workers):
        result_path, result = workers[seed]
        root = result_path.parent
        rows = _read_csv(root / result["artifacts"]["policy_metrics"])
        if not rows:
            raise ValueError(f"Worker has no policy metrics: {root}")
        metrics.extend(rows)
        for record in result["incumbent_snapshots"]:
            path = root / record["relative_path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Missing or corrupt playable snapshot: {path}")
            inventory.append({**record, "path": str(path.resolve())})
        for record in result["historical_policy_checkpoints"]:
            path = root / record["relative_path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Missing or corrupt historical checkpoint: {path}")
            inventory.append({
                **record, "candidate_id": CANDIDATE_ID,
                "seed": int(seed), "path": str(path.resolve()),
            })
        manifests.append({
            "seed": int(seed), "repository_commit": result["repository_commit"],
            "historical_snapshot_count": int(result["historical_snapshot_count"]),
            "historical_storage_bytes": int(result["historical_storage_bytes"]),
            "peak_rss_mb": float(result["peak_rss_mb"]),
            "worker_result": str(result_path.resolve()),
        })
    return metrics, inventory, manifests


def _typed(row: Mapping) -> dict:
    return {
        **row,
        "seed": int(row["seed"]),
        "checkpoint_target_active_hours": float(row["checkpoint_target_active_hours"]),
        "actual_active_hours": float(row["actual_active_hours"]),
        "nodes_touched": int(row["nodes_touched"]),
        "completed_iteration": int(row["completed_iteration"]),
        "exploitability": float(row["exploitability"]),
        "arm_minus_incumbent": float(row["arm_minus_incumbent"]),
        "arm_minus_exact": float(row["arm_minus_exact"]),
    }


def _summaries(rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm_id"]), str(row["checkpoint_id"]))].append(row)
    result = []
    for (arm_id, checkpoint_id), values in grouped.items():
        stats = summary([float(row["exploitability"]) for row in values])
        result.append({
            "arm_id": arm_id, "arm_label": values[0]["arm_label"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_target_active_hours": float(values[0]["checkpoint_target_active_hours"]),
            "mean_actual_active_hours": float(np.mean([row["actual_active_hours"] for row in values])),
            "mean_nodes_touched": float(np.mean([row["nodes_touched"] for row in values])),
            "mean_exploitability": float(stats["mean_ev"]),
            "standard_deviation_exploitability": float(stats["standard_deviation"]),
            "standard_error_exploitability": float(stats["standard_error"]),
            "ci95_lower_exploitability": float(stats["ci95_lower"]),
            "ci95_upper_exploitability": float(stats["ci95_upper"]),
            "n_seeds": int(stats["n_seeds"]),
        })
    return sorted(result, key=lambda row: (row["checkpoint_target_active_hours"], row["arm_id"]))


def _paired(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    effects = [
        {
            "seed": row["seed"], "checkpoint_id": row["checkpoint_id"],
            "checkpoint_target_active_hours": row["checkpoint_target_active_hours"],
            "comparator_id": INCUMBENT_NEURAL_ID,
            "single_minus_incumbent": row["arm_minus_incumbent"],
        }
        for row in rows if row["arm_id"] == CANDIDATE_ID
    ]
    grouped = defaultdict(list)
    for row in effects:
        grouped[row["checkpoint_id"]].append(row)
    summaries = []
    for checkpoint_id, values in grouped.items():
        differences = [float(row["single_minus_incumbent"]) for row in values]
        stats = summary(differences)
        summaries.append({
            "checkpoint_id": checkpoint_id,
            "checkpoint_target_active_hours": values[0]["checkpoint_target_active_hours"],
            "mean_single_minus_incumbent": float(stats["mean_ev"]),
            "ci95_lower": float(stats["ci95_lower"]),
            "ci95_upper": float(stats["ci95_upper"]),
            "single_better_seed_fraction": sum(value < 0 for value in differences) / len(differences),
            "two_sided_exact_sign_flip_p": float(stats["two_sided_exact_sign_flip_p"]),
            "n_seeds": len(differences),
        })
    return effects, summaries


def _plot(rows: Sequence[Mapping], path: Path, arm_ids: Sequence[str], title: str) -> None:
    grouped = defaultdict(list)
    labels = {}
    for row in rows:
        if row["arm_id"] in arm_ids:
            grouped[(row["arm_id"], row["checkpoint_target_active_hours"])].append(
                float(row["exploitability"])
            )
            labels[row["arm_id"]] = row["arm_label"]
    fig, axis = plt.subplots(figsize=(9.2, 5.6))
    for arm_id in arm_ids:
        points = sorted(
            (hours, values) for (candidate, hours), values in grouped.items()
            if candidate == arm_id
        )
        if not points:
            continue
        x = [float(point[0]) for point in points]
        means = [float(np.mean(point[1])) for point in points]
        errors = [
            float(np.std(point[1], ddof=1) / np.sqrt(len(point[1]))) if len(point[1]) > 1 else 0.0
            for point in points
        ]
        axis.errorbar(
            x, means, yerr=errors, marker="o", capsize=3,
            color=COLOURS[arm_id], label=labels[arm_id],
        )
    set_chart_title(axis, title)
    axis.set_xlabel("Effective training time (hours)")
    axis.set_ylabel("Exploitability (NashConv / 2)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def aggregate_workers(
    *, workers_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool,
) -> dict:
    workers = _load_workers(workers_root, seeds, smoke)
    raw, inventory, worker_manifests = _collect(workers)
    rows = [_typed(row) for row in raw]
    summaries = _summaries(rows)
    effects, effect_summaries = _paired(rows)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "combined_policy_metrics.csv", rows)
    write_csv(output_dir / "policy_summary.csv", summaries)
    write_csv(output_dir / "paired_single_vs_incumbent.csv", effects)
    write_csv(output_dir / "paired_single_vs_incumbent_summary.csv", effect_summaries)
    write_csv(output_dir / "playable_snapshot_inventory.csv", inventory)
    write_csv(output_dir / "worker_manifest.csv", worker_manifests)
    _plot(
        rows, output_dir / "single_ucv_escher_vs_policy_outputs.png",
        PRIMARY_ORDER,
        "Single UCV-ESCHER versus average-policy representations",
    )
    bounded_order = (CANDIDATE_ID,) + tuple(
        f"historical_reservoir_{capacity}" for capacity in BOUNDED_CAPACITIES
    )
    _plot(
        rows, output_dir / "bounded_historical_mixture_capacity.png",
        bounded_order,
        "Bounded historical-network mixtures",
    )
    manifest = {
        "status": "complete", "smoke": bool(smoke),
        "contract": contract_manifest(),
        "repository_commit": worker_manifests[0]["repository_commit"],
        "num_workers": len(workers), "num_metric_rows": len(rows),
        "artifacts": {
            "combined_policy_metrics": "combined_policy_metrics.csv",
            "policy_summary": "policy_summary.csv",
            "paired_effects": "paired_single_vs_incumbent.csv",
            "paired_effect_summary": "paired_single_vs_incumbent_summary.csv",
            "primary_chart": "single_ucv_escher_vs_policy_outputs.png",
            "bounded_chart": "bounded_historical_mixture_capacity.png",
        },
    }
    write_json(output_dir / "aggregate_manifest.json", manifest)
    return manifest


__all__ = ["aggregate_workers"]
