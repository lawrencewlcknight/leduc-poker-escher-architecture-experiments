"""Aggregate Experiment 42 and join the immutable Experiment 41 reference."""

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
    ARM_IDS,
    EQUAL_EXAMPLE_ARM,
    EXPERIMENT_NAME,
    GROUPED_REFERENCE_ARM,
    PRACTICAL_EQUIVALENCE_MARGIN,
    contract_manifest,
)


ARM_LABELS = {
    EQUAL_EXAMPLE_ARM: "Row-wise: equal examples",
    "rowwise_equal_updates": "Row-wise: equal optimiser updates",
    GROUPED_REFERENCE_ARM: "Experiment 41 grouped full batch",
}
NEW_METRICS = (
    "exploitability",
    "change_from_archived_policy",
    "gap_to_exact_tabular_average",
    "diagnostic_row_cross_entropy",
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


def _reference_analysis(path: Path) -> Path:
    path = Path(path)
    if (path / "aggregate_summary.json").is_file():
        return path
    if (path / "analysis" / "aggregate_summary.json").is_file():
        return path / "analysis"
    raise FileNotFoundError(f"Experiment 41 analysis not found under {path}")


def aggregate_workers(*, workers_root: Path, seeds, output_dir: Path,
                      experiment_41_root: Path, smoke: bool) -> dict:
    expected = {int(seed) for seed in seeds}
    workers = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != EXPERIMENT_NAME:
            continue
        seed = int(result["source_seed"])
        if seed in workers:
            raise ValueError(f"Duplicate Experiment 42 worker for seed {seed}")
        if result.get("status") != "complete" or bool(result["smoke"]) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker {path}")
        if result["dataset"].get("grouping_performed_in_training_path") is not False:
            raise ValueError(f"Worker {seed} did not certify row-wise training")
        workers[seed] = (path, result)
    if set(workers) != expected:
        raise ValueError(f"Worker seeds differ: {sorted(workers)} != {sorted(expected)}")
    commits = {result["repository_commit"] for _, result in workers.values()}
    if len(commits) != 1:
        raise ValueError("Experiment 42 workers used different repository commits")

    metrics, manifest, baselines, snapshots = [], [], [], []
    diagnostic_updates = None
    for seed in sorted(workers):
        path, result = workers[seed]
        rows = _read_csv(path.parent / result["artifacts"]["metrics"])
        observed_arms = {row["arm_id"] for row in rows}
        if observed_arms != set(ARM_IDS):
            raise ValueError(f"Worker {seed} arms differ: {sorted(observed_arms)}")
        expected_updates = tuple(
            int(value) for value in result["runtime_schedule"][
                "diagnostic_equivalent_updates"
            ]
        )
        if diagnostic_updates is None:
            diagnostic_updates = expected_updates
        elif diagnostic_updates != expected_updates:
            raise ValueError("Experiment 42 workers used different schedules")
        for arm_id in ARM_IDS:
            observed = {
                int(row["equivalent_update"])
                for row in rows if row["arm_id"] == arm_id
            }
            if observed != set(expected_updates):
                raise ValueError(f"Worker {seed} {arm_id} updates differ")
        metrics.extend(rows)
        for policy_id, value in result["baselines"].items():
            baselines.append(
                {"source_seed": seed, "policy_id": policy_id,
                 "exploitability": float(value)}
            )
        snapshots.extend(
            {"source_seed": seed, **snapshot}
            for snapshot in result["snapshots"]
        )
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "source_sha256": result["source"]["sha256"],
                "replay_rows": result["dataset"]["replay_rows"],
                "unique_information_states_reference": result[
                    "runtime_schedule"
                ]["unique_information_states"],
                "peak_rss_mb": result["peak_rss_mb"],
                "wall_clock_seconds": result["wall_clock_seconds"],
                **{
                    f"{arm_id}_runtime_seconds": result[
                        "arm_runtime_seconds"
                    ][arm_id]
                    for arm_id in ARM_IDS
                },
            }
        )

    reference = _reference_analysis(experiment_41_root)
    reference_summary = read_json(reference / "aggregate_summary.json")
    if int(reference_summary["contract"]["experiment_id"]) != 41:
        raise ValueError("Reference analysis is not Experiment 41")
    reference_metrics = _read_csv(reference / "fit_metrics.csv")
    reference_manifest = {
        int(row["source_seed"]): row
        for row in _read_csv(reference / "worker_manifest.csv")
    }
    if set(reference_manifest) != expected:
        raise ValueError("Experiment 41 reference seeds differ")
    for row in manifest:
        ref = reference_manifest[int(row["source_seed"])]
        if row["source_sha256"] != ref["source_sha256"]:
            raise ValueError("Experiment 41 and 42 source digests differ")
        if int(row["unique_information_states_reference"]) != int(
            ref["unique_information_states"]
        ):
            raise ValueError("Frozen information-set count differs from Experiment 41")

    reference_by_key = {
        (int(row["source_seed"]), int(row["optimizer_update"])): row
        for row in reference_metrics
    }
    grouped_rows = []
    for seed in sorted(expected):
        unique_count = int(reference_manifest[seed]["unique_information_states"])
        for update in diagnostic_updates:
            row = reference_by_key.get((seed, update))
            if row is None:
                raise ValueError(f"Experiment 41 lacks seed {seed}, update {update}")
            grouped_rows.append(
                {
                    "source_seed": seed,
                    "arm_id": GROUPED_REFERENCE_ARM,
                    "equivalent_update": update,
                    "examples_seen": update * unique_count,
                    "optimizer_steps": update,
                    "exploitability": float(row["exploitability"]),
                }
            )

    new_summary = _summarise(
        metrics, ("arm_id", "equivalent_update"), NEW_METRICS
    )
    combined = [
        {
            "source_seed": int(row["source_seed"]),
            "arm_id": row["arm_id"],
            "equivalent_update": int(row["equivalent_update"]),
            "examples_seen": int(row["examples_seen"]),
            "optimizer_steps": int(row["optimizer_steps"]),
            "exploitability": float(row["exploitability"]),
        }
        for row in metrics
    ] + grouped_rows
    combined_summary = _summarise(
        combined, ("arm_id", "equivalent_update"), ("exploitability",)
    )

    final_update = max(diagnostic_updates)
    values = {
        (int(row["source_seed"]), row["arm_id"], int(row["equivalent_update"])): row
        for row in combined
    }
    archived = {
        int(row["source_seed"]): float(row["exploitability"])
        for row in baselines if row["policy_id"] == "archived_neural_policy"
    }
    final_effects = []
    for seed in sorted(expected):
        grouped = float(values[(seed, GROUPED_REFERENCE_ARM, final_update)]["exploitability"])
        denominator = archived[seed] - grouped
        for arm_id in ARM_IDS:
            row = values[(seed, arm_id, final_update)]
            value = float(row["exploitability"])
            final_effects.append(
                {
                    "source_seed": seed,
                    "arm_id": arm_id,
                    "rowwise_minus_grouped": value - grouped,
                    "improvement_from_archived": archived[seed] - value,
                    "fraction_grouped_improvement_recovered": (
                        (archived[seed] - value) / denominator
                        if denominator != 0.0 else float("nan")
                    ),
                    "final_exploitability": value,
                    "examples_seen": int(row["examples_seen"]),
                    "optimizer_steps": int(row["optimizer_steps"]),
                }
            )
    effect_summary = _summarise(
        final_effects, ("arm_id",),
        (
            "rowwise_minus_grouped",
            "improvement_from_archived",
            "fraction_grouped_improvement_recovered",
            "final_exploitability",
            "examples_seen",
            "optimizer_steps",
        ),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "rowwise_fit_metrics.csv", metrics)
    write_csv(output_dir / "rowwise_fit_summary.csv", new_summary)
    write_csv(output_dir / "combined_exploitability_metrics.csv", combined)
    write_csv(output_dir / "combined_exploitability_summary.csv", combined_summary)
    write_csv(output_dir / "baseline_metrics.csv", baselines)
    write_csv(
        output_dir / "baseline_summary.csv",
        _summarise(baselines, ("policy_id",), ("exploitability",)),
    )
    write_csv(output_dir / "paired_final_effects.csv", final_effects)
    write_csv(output_dir / "paired_final_summary.csv", effect_summary)
    write_csv(output_dir / "snapshot_inventory.csv", snapshots)
    write_csv(output_dir / "worker_manifest.csv", manifest)

    fig, ax = plt.subplots(figsize=(10.5, 6.1))
    for arm_id in (*ARM_IDS, GROUPED_REFERENCE_ARM):
        rows = sorted(
            (
                row for row in combined_summary
                if row["arm_id"] == arm_id and row["metric"] == "exploitability"
            ),
            key=lambda row: int(row["equivalent_update"]),
        )
        x = np.asarray([int(row["equivalent_update"]) for row in rows])
        y = np.asarray([float(row["mean"]) for row in rows])
        lower = np.asarray([float(row["ci95_lower"]) for row in rows])
        upper = np.asarray([float(row["ci95_upper"]) for row in rows])
        ax.plot(x, y, marker="o", label=ARM_LABELS[arm_id])
        ax.fill_between(x, lower, upper, alpha=0.10)
    ax.set_xlabel("Experiment 41-equivalent optimisation horizon")
    ax.set_ylabel("Exact exploitability (NashConv / 2; analysis only)")
    set_chart_title(
        ax, "Grouped versus row-wise policy continuation",
        algorithm="Experiment 29 UCV average policy", game_name="leduc_poker",
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "rowwise_vs_grouped_exploitability.png", dpi=180)
    plt.close(fig)

    final_resource = []
    for arm_id in ARM_IDS:
        arm_rows = [row for row in final_effects if row["arm_id"] == arm_id]
        final_resource.append(
            {
                "arm_id": arm_id,
                "mean_examples_seen": float(np.mean([row["examples_seen"] for row in arm_rows])),
                "mean_final_exploitability": float(np.mean([row["final_exploitability"] for row in arm_rows])),
            }
        )
    grouped_final = [
        row for row in combined
        if row["arm_id"] == GROUPED_REFERENCE_ARM
        and int(row["equivalent_update"]) == final_update
    ]
    final_resource.append(
        {
            "arm_id": GROUPED_REFERENCE_ARM,
            "mean_examples_seen": float(np.mean([row["examples_seen"] for row in grouped_final])),
            "mean_final_exploitability": float(np.mean([row["exploitability"] for row in grouped_final])),
        }
    )
    write_csv(output_dir / "final_resource_tradeoff.csv", final_resource)

    primary = next(
        row for row in effect_summary
        if row["arm_id"] == EQUAL_EXAMPLE_ARM
        and row["metric"] == "rowwise_minus_grouped"
    )
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_source_seeds": len(expected),
        "contract": contract_manifest(),
        "primary_result": {
            "arm_id": EQUAL_EXAMPLE_ARM,
            "mean_rowwise_minus_grouped": float(primary["mean"]),
            "ci95_lower": float(primary["ci95_lower"]),
            "ci95_upper": float(primary["ci95_upper"]),
            "practical_equivalence_margin": PRACTICAL_EQUIVALENCE_MARGIN,
            "upper_ci_within_margin": (
                float(primary["ci95_upper"]) <= PRACTICAL_EQUIVALENCE_MARGIN
            ),
        },
        "interpretation_guardrail": (
            "Exact exploitability is a Leduc-only diagnostic. Training uses "
            "ordinary replay rows and no grouping or game-tree information."
        ),
        "outputs": {
            "rowwise_metrics": "rowwise_fit_metrics.csv",
            "combined_metrics": "combined_exploitability_metrics.csv",
            "paired_effects": "paired_final_effects.csv",
            "paired_summary": "paired_final_summary.csv",
            "worker_manifest": "worker_manifest.csv",
            "snapshot_inventory": "snapshot_inventory.csv",
            "resource_tradeoff": "final_resource_tradeoff.csv",
            "exploitability_chart": "rowwise_vs_grouped_exploitability.png",
        },
    }
    write_json(output_dir / "aggregate_summary.json", result)
    return result


__all__ = ["aggregate_workers"]
