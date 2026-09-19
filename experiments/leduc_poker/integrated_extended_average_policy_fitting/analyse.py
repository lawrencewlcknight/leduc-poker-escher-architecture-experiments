"""Aggregate Experiment 43 and join the frozen Experiment 29 comparators."""

from __future__ import annotations

import csv
from collections import defaultdict
import math
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
    CANDIDATE_ID,
    CANDIDATE_LABEL,
    EMPIRICAL_POLICY_ID,
    EMPIRICAL_POLICY_LABEL,
    EXACT_POLICY_ID,
    EXACT_POLICY_LABEL,
    HISTORICAL_ALGORITHM_LABELS,
    HISTORICAL_ALGORITHM_ORDER,
    ORDINARY_POLICY_ID,
    ORDINARY_POLICY_LABEL,
    REFERENCE_EXPERIMENT_29_MANIFEST_SHA256,
    REFERENCE_EXPERIMENT_29_METRICS_SHA256,
    REFINEMENT_UPDATES,
    checkpoint_schedule,
    contract_manifest,
)


POLICY_LABELS = {
    CANDIDATE_ID: CANDIDATE_LABEL,
    ORDINARY_POLICY_ID: ORDINARY_POLICY_LABEL,
    EMPIRICAL_POLICY_ID: EMPIRICAL_POLICY_LABEL,
    EXACT_POLICY_ID: EXACT_POLICY_LABEL,
}
INTERNAL_ORDER = (
    CANDIDATE_ID, ORDINARY_POLICY_ID, EMPIRICAL_POLICY_ID, EXACT_POLICY_ID,
)
PERFORMANCE_ORDER = HISTORICAL_ALGORITHM_ORDER + (CANDIDATE_ID,)
PERFORMANCE_LABELS = {**HISTORICAL_ALGORITHM_LABELS, CANDIDATE_ID: CANDIDATE_LABEL}
COLOURS = {
    "deep_cfr": "#1f77b4",
    "unbiased_control_variate_escher": "#d62728",
    "selected_nonpredictive_ucv": "#ff7f0e",
    "promoted_ucv_cross_entropy": "#9467bd",
    CANDIDATE_ID: "#2ca02c",
    ORDINARY_POLICY_ID: "#7f7f7f",
    EMPIRICAL_POLICY_ID: "#8c564b",
    EXACT_POLICY_ID: "#17becf",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_workers(workers_root: Path, seeds: Sequence[int], smoke: bool):
    expected = {int(seed) for seed in seeds}
    found = {}
    schedule = checkpoint_schedule(smoke=smoke)
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        seed = int(result["seed"])
        if seed in found:
            raise ValueError(f"Duplicate Experiment 43 worker for seed {seed}")
        if (
            result.get("status") != "complete"
            or result.get("candidate_id") != CANDIDATE_ID
            or bool(result.get("smoke")) != bool(smoke)
            or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
        ):
            raise ValueError(f"Incomplete or mismatched Experiment 43 worker: {path}")
        found[seed] = (path, result)
    if set(found) != expected:
        raise ValueError(
            f"Experiment 43 workers differ; missing={sorted(expected-set(found))}, "
            f"extra={sorted(set(found)-expected)}"
        )
    if len({result["repository_commit"] for _, result in found.values()}) != 1:
        raise ValueError("Experiment 43 workers used different repository commits")
    return found


def _worker_metrics(workers: Mapping[int, tuple[Path, Mapping]], *, smoke: bool):
    rows, inventory, manifests, audit = [], [], [], []
    maximum = 2 if smoke else REFINEMENT_UPDATES
    for seed in sorted(workers):
        result_path, result = workers[seed]
        root = result_path.parent
        raw = _read_csv(root / result["artifacts"]["refinement_metrics"])
        if not raw:
            raise ValueError(f"Experiment 43 worker has no refinement metrics: {root}")
        for record in (
            result["ordinary_snapshots"] + result["refined_snapshots"]
            + result.get("final_audit_snapshots", [])
        ):
            path = root / record["relative_path"]
            if not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"Missing or corrupt Experiment 43 snapshot: {path}")
            inventory.append({**record, "path": str(path.resolve())})
        by_checkpoint = defaultdict(list)
        for row in raw:
            by_checkpoint[str(row["checkpoint_id"])].append(row)
            if int(row["refinement_update"]) > 0:
                audit.append(dict(row))
        for checkpoint_id, values in by_checkpoint.items():
            ordinary = next(row for row in values if int(row["refinement_update"]) == 0)
            refined = next(row for row in values if int(row["refinement_update"]) == maximum)
            common = {
                "seed": int(seed),
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": refined["checkpoint_type"],
                "checkpoint_target_active_hours": (
                    None if refined["checkpoint_target_active_hours"] in {"", "None"}
                    else float(refined["checkpoint_target_active_hours"])
                ),
                "actual_active_hours": float(refined["actual_active_hours"]),
                "nodes_touched": int(refined["nodes_touched"]),
                "completed_iteration": int(refined["completed_iteration"]),
            }
            values_by_policy = {
                CANDIDATE_ID: float(refined["exploitability"]),
                ORDINARY_POLICY_ID: float(ordinary["exploitability"]),
                EMPIRICAL_POLICY_ID: float(refined["empirical_reservoir_exploitability"]),
                EXACT_POLICY_ID: float(refined["exact_tabular_average_exploitability"]),
            }
            for policy_id, exploitability in values_by_policy.items():
                rows.append(
                    {
                        **common,
                        "policy_id": policy_id,
                        "policy_label": POLICY_LABELS[policy_id],
                        "exploitability": exploitability,
                        "candidate_minus_policy": float(refined["exploitability"]) - exploitability,
                    }
                )
        manifests.append(
            {
                "seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "worker_result": str(result_path.resolve()),
            }
        )
    return rows, inventory, manifests, audit


def _historical_metrics(experiment_29_root: Path, *, smoke: bool):
    analysis = Path(experiment_29_root).resolve() / "analysis"
    metrics_path = analysis / "combined_checkpoint_policy_metrics.csv"
    manifest_path = analysis / "aggregate_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Experiment 29 comparator artifacts incomplete under {analysis}")
    if not smoke:
        if sha256(metrics_path) != REFERENCE_EXPERIMENT_29_METRICS_SHA256:
            raise ValueError("Experiment 29 comparator metrics differ from the frozen artifact")
        if sha256(manifest_path) != REFERENCE_EXPERIMENT_29_MANIFEST_SHA256:
            raise ValueError("Experiment 29 comparator manifest differs from the frozen artifact")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or bool(manifest.get("smoke")) != bool(smoke):
        raise ValueError("Experiment 29 comparator manifest has wrong status or mode")
    rows = _read_csv(metrics_path)
    active_ids = {
        str(row["checkpoint_id"]) for row in checkpoint_schedule(smoke=smoke)
        if row["checkpoint_type"] == "active_time"
    }
    expected_seeds = {0} if smoke else set(int(seed) for seed in manifest["contract"]["production_seeds"])
    if {str(row["algorithm_id"]) for row in rows} != set(HISTORICAL_ALGORITHM_ORDER):
        raise ValueError("Experiment 29 comparator algorithm set differs")
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Experiment 29 comparator seed set differs")
    rows = [row for row in rows if str(row["checkpoint_id"]) in active_ids]
    expected = len(HISTORICAL_ALGORITHM_ORDER) * len(expected_seeds) * len(active_ids)
    if len(rows) != expected:
        raise ValueError("Experiment 29 comparator rows are incomplete")
    return [
        {
            "algorithm_id": str(row["algorithm_id"]),
            "algorithm_label": HISTORICAL_ALGORITHM_LABELS[str(row["algorithm_id"])],
            "seed": int(row["seed"]),
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint_type": "active_time",
            "checkpoint_target_active_hours": float(row["checkpoint_target_active_hours"]),
            "actual_active_hours": float(row["actual_active_hours"]),
            "nodes_touched": int(row["nodes_touched"]),
            "completed_iteration": int(row["completed_iteration"]),
            "exploitability": float(row["exploitability"]),
            "source": "experiment_29_frozen",
        }
        for row in rows
    ], manifest


def _candidate_performance(metrics: Sequence[Mapping]):
    return [
        {
            "algorithm_id": CANDIDATE_ID,
            "algorithm_label": CANDIDATE_LABEL,
            "seed": int(row["seed"]),
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_type": "active_time",
            "checkpoint_target_active_hours": float(row["checkpoint_target_active_hours"]),
            "actual_active_hours": float(row["actual_active_hours"]),
            "nodes_touched": int(row["nodes_touched"]),
            "completed_iteration": int(row["completed_iteration"]),
            "exploitability": float(row["exploitability"]),
            "source": "experiment_43_new",
        }
        for row in metrics
        if row["policy_id"] == CANDIDATE_ID
        and row["checkpoint_type"] == "active_time"
    ]


def _summaries(rows: Sequence[Mapping], *, id_field: str, order: Sequence[str], label_map):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row[id_field]), str(row["checkpoint_id"]))].append(row)
    output = []
    for item_id in order:
        checkpoints = sorted(
            {checkpoint for candidate, checkpoint in grouped if candidate == item_id},
            key=lambda value: (
                grouped[(item_id, value)][0]["checkpoint_target_active_hours"] is None,
                float(grouped[(item_id, value)][0]["checkpoint_target_active_hours"] or 0),
            ),
        )
        for checkpoint in checkpoints:
            values = grouped[(item_id, checkpoint)]
            stats = summary([float(row["exploitability"]) for row in values])
            output.append(
                {
                    id_field: item_id,
                    "label": label_map[item_id],
                    "checkpoint_id": checkpoint,
                    "checkpoint_type": values[0]["checkpoint_type"],
                    "checkpoint_target_active_hours": values[0]["checkpoint_target_active_hours"],
                    "mean_actual_active_hours": float(np.mean([float(row["actual_active_hours"]) for row in values])),
                    "mean_nodes_touched": float(np.mean([int(row["nodes_touched"]) for row in values])),
                    "mean_exploitability": float(stats["mean_ev"]),
                    "standard_deviation_exploitability": float(stats["standard_deviation"]),
                    "standard_error_exploitability": float(stats["standard_error"]),
                    "ci95_lower_exploitability": float(stats["ci95_lower"]),
                    "ci95_upper_exploitability": float(stats["ci95_upper"]),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )
    return output


def _paired_effects(internal, performance, *, smoke: bool):
    internal_index = {
        (int(row["seed"]), str(row["checkpoint_id"]), str(row["policy_id"])): row
        for row in internal
    }
    historical_index = {
        (int(row["seed"]), str(row["checkpoint_id"]), str(row["algorithm_id"])): row
        for row in performance if row["algorithm_id"] != CANDIDATE_ID
    }
    candidate_rows = [row for row in internal if row["policy_id"] == CANDIDATE_ID]
    comparators = [
        (ORDINARY_POLICY_ID, ORDINARY_POLICY_LABEL, internal_index),
        (EMPIRICAL_POLICY_ID, EMPIRICAL_POLICY_LABEL, internal_index),
        (EXACT_POLICY_ID, EXACT_POLICY_LABEL, internal_index),
    ] + [
        (algorithm_id, HISTORICAL_ALGORITHM_LABELS[algorithm_id], historical_index)
        for algorithm_id in HISTORICAL_ALGORITHM_ORDER
    ]
    effects = []
    for comparator_id, comparator_label, index in comparators:
        for row in candidate_rows:
            if comparator_id in HISTORICAL_ALGORITHM_ORDER and row["checkpoint_type"] != "active_time":
                continue
            key = (int(row["seed"]), str(row["checkpoint_id"]), comparator_id)
            if key not in index:
                continue
            effects.append(
                {
                    "seed": int(row["seed"]),
                    "checkpoint_id": row["checkpoint_id"],
                    "checkpoint_type": row["checkpoint_type"],
                    "checkpoint_target_active_hours": row["checkpoint_target_active_hours"],
                    "comparator_id": comparator_id,
                    "comparator_label": comparator_label,
                    "candidate_minus_comparator": float(row["exploitability"]) - float(index[key]["exploitability"]),
                }
            )
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row["comparator_id"], row["checkpoint_id"])].append(row)
    summaries = []
    for (comparator, checkpoint), values in grouped.items():
        differences = [float(row["candidate_minus_comparator"]) for row in values]
        stats = summary(differences)
        summaries.append(
            {
                "comparator_id": comparator,
                "comparator_label": values[0]["comparator_label"],
                "checkpoint_id": checkpoint,
                "checkpoint_type": values[0]["checkpoint_type"],
                "checkpoint_target_active_hours": values[0]["checkpoint_target_active_hours"],
                "mean_candidate_minus_comparator": float(stats["mean_ev"]),
                "ci95_lower": float(stats["ci95_lower"]),
                "ci95_upper": float(stats["ci95_upper"]),
                "candidate_better_seed_fraction": sum(value < 0 for value in differences) / len(differences),
                "two_sided_exact_sign_flip_p": float(stats["two_sided_exact_sign_flip_p"]),
                "n_seeds": len(differences),
            }
        )
    return effects, summaries


def _late_window(rows: Sequence[Mapping], *, id_field: str, smoke: bool):
    result = []
    for item_id in sorted({str(row[id_field]) for row in rows}):
        for seed in sorted({int(row["seed"]) for row in rows if str(row[id_field]) == item_id}):
            trajectory = sorted(
                [row for row in rows if str(row[id_field]) == item_id and int(row["seed"]) == seed and row["checkpoint_type"] == "active_time"],
                key=lambda row: float(row["checkpoint_target_active_hours"]),
            )
            late = trajectory if smoke else [row for row in trajectory if float(row["checkpoint_target_active_hours"]) >= 24.0]
            values = np.asarray([float(row["exploitability"]) for row in late])
            hours = np.asarray([float(row["checkpoint_target_active_hours"]) for row in late])
            adjacent = np.diff(values)
            result.append(
                {
                    id_field: item_id,
                    "seed": seed,
                    "late_window_mean_exploitability": float(np.mean(values)),
                    "late_window_adjacent_rmssd": float(np.sqrt(np.mean(np.square(adjacent))) if len(adjacent) else 0.0),
                    "late_window_max_deterioration": float(max(0.0, np.max(adjacent)) if len(adjacent) else 0.0),
                    "late_window_slope_per_hour": float(np.polyfit(hours, values, 1)[0] if len(values) > 1 else 0.0),
                    "final_exploitability": float(values[-1]),
                    "final_nodes_touched": int(trajectory[-1]["nodes_touched"]),
                }
            )
    return result


def _plot_series(rows, summaries, *, id_field, order, label_map, output, by_nodes, title):
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    for item_id in order:
        source = [row for row in rows if str(row[id_field]) == item_id and row["checkpoint_type"] == "active_time"]
        for seed in sorted({int(row["seed"]) for row in source}):
            seed_rows = sorted(
                [row for row in source if int(row["seed"]) == seed],
                key=lambda row: int(row["nodes_touched"]) if by_nodes else float(row["checkpoint_target_active_hours"]),
            )
            x = [int(row["nodes_touched"]) / 1e6 if by_nodes else float(row["checkpoint_target_active_hours"]) for row in seed_rows]
            ax.plot(x, [float(row["exploitability"]) for row in seed_rows], color=COLOURS[item_id], alpha=0.10, linewidth=0.75)
        mean_rows = sorted(
            [row for row in summaries if str(row[id_field]) == item_id and row["checkpoint_type"] == "active_time"],
            key=lambda row: float(row["mean_nodes_touched"]) if by_nodes else float(row["checkpoint_target_active_hours"]),
        )
        x = np.asarray([float(row["mean_nodes_touched"]) / 1e6 if by_nodes else float(row["checkpoint_target_active_hours"]) for row in mean_rows])
        mean = np.asarray([float(row["mean_exploitability"]) for row in mean_rows])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in mean_rows])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in mean_rows])
        ax.plot(x, mean, color=COLOURS[item_id], linewidth=2.5 if item_id == CANDIDATE_ID else 1.8, marker="o", markersize=3.5, label=label_map[item_id])
        ax.fill_between(x, lower, upper, color=COLOURS[item_id], alpha=0.08)
    ax.set_xlabel("Training nodes touched (millions)" if by_nodes else "Active training time (hours)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    set_chart_title(ax, title)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_gaps(internal, output):
    rows = [row for row in internal if row["policy_id"] == CANDIDATE_ID and row["checkpoint_type"] == "active_time"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for comparator, label in ((EMPIRICAL_POLICY_ID, "Empirical reservoir"), (EXACT_POLICY_ID, "Exact tabular average")):
        indexed = {(int(row["seed"]), row["checkpoint_id"]): row for row in internal if row["policy_id"] == comparator}
        grouped = defaultdict(list)
        for row in rows:
            key = (int(row["seed"]), row["checkpoint_id"])
            grouped[float(row["checkpoint_target_active_hours"])].append(float(row["exploitability"]) - float(indexed[key]["exploitability"]))
        hours = sorted(grouped)
        ax.plot(hours, [np.mean(grouped[hour]) for hour in hours], marker="o", linewidth=2, label=f"Refined minus {label}")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Active training time (hours)")
    ax.set_ylabel("Exploitability gap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    set_chart_title(ax, "Extended-fit UCV distillation gap over training")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_paired_final(effects, output, *, smoke: bool):
    final = max(
        [row for row in effects if row["checkpoint_type"] == "active_time"],
        key=lambda row: float(row["checkpoint_target_active_hours"]),
    )["checkpoint_id"]
    rows = [row for row in effects if row["checkpoint_id"] == final and row["comparator_id"] == ORDINARY_POLICY_ID]
    rows.sort(key=lambda row: int(row["seed"]))
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar([str(row["seed"]) for row in rows], [float(row["candidate_minus_comparator"]) for row in rows], color="#2ca02c")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Refined minus ordinary exploitability")
    set_chart_title(ax, "Paired effect of integrated policy refinement")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_final_audit(audit, output, *, smoke: bool):
    active = [row for row in audit if row["checkpoint_type"] == "active_time"]
    final_id = max(active, key=lambda row: float(row["checkpoint_target_active_hours"]))["checkpoint_id"]
    rows = [row for row in active if row["checkpoint_id"] == final_id]
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["refinement_update"])].append(float(row["exploitability"]))
    updates = sorted(grouped)
    means = [float(np.mean(grouped[update])) for update in updates]
    errors = [float(np.std(grouped[update], ddof=1) / math.sqrt(len(grouped[update]))) if len(grouped[update]) > 1 else 0.0 for update in updates]
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.errorbar(updates, means, yerr=errors, marker="o", capsize=3)
    ax.set_xlabel("Additional grouped full-batch updates")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.grid(axis="y", alpha=0.25)
    set_chart_title(ax, "Final-checkpoint policy refinement horizon")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def aggregate_workers(*, workers_root: Path, experiment_29_root: Path,
                      seeds: Sequence[int], output_dir: Path, smoke: bool) -> dict:
    workers = _load_workers(Path(workers_root).resolve(), seeds, smoke)
    internal, inventory, manifests, audit = _worker_metrics(workers, smoke=smoke)
    historical, historical_manifest = _historical_metrics(experiment_29_root, smoke=smoke)
    performance = historical + _candidate_performance(internal)
    internal_summary = _summaries(
        internal, id_field="policy_id", order=INTERNAL_ORDER, label_map=POLICY_LABELS
    )
    performance_summary = _summaries(
        performance, id_field="algorithm_id", order=PERFORMANCE_ORDER,
        label_map=PERFORMANCE_LABELS,
    )
    effects, paired_summary = _paired_effects(internal, performance, smoke=smoke)
    internal_late = _late_window(internal, id_field="policy_id", smoke=smoke)
    performance_late = _late_window(performance, id_field="algorithm_id", smoke=smoke)
    maximum = 2 if smoke else REFINEMENT_UPDATES
    final_refinements = [
        row for row in audit if int(row["refinement_update"]) == maximum
    ]
    head_to_head = [
        {
            "seed": int(row["seed"]),
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_type": row["checkpoint_type"],
            "checkpoint_target_active_hours": row["checkpoint_target_active_hours"],
            "refined_vs_ordinary_seat_averaged_ev": float(row["refined_vs_ordinary_seat_averaged_ev"]),
        }
        for row in final_refinements
    ]
    resource_rows = []
    for seed in sorted({int(row["seed"]) for row in final_refinements}):
        seed_rows = [row for row in final_refinements if int(row["seed"]) == seed]
        active_rows = [row for row in seed_rows if row["checkpoint_type"] == "active_time"]
        final_active = max(
            active_rows,
            key=lambda row: float(row["checkpoint_target_active_hours"]),
        )
        manifest = next(row for row in manifests if int(row["seed"]) == seed)
        resource_rows.append(
            {
                "seed": seed,
                "total_refinement_seconds": float(sum(float(row["fit_elapsed_seconds"]) for row in seed_rows)),
                "final_refinement_seconds": float(final_active["fit_elapsed_seconds"]),
                "peak_rss_mb": float(manifest["peak_rss_mb"]),
                "final_nodes_touched": int(final_active["nodes_touched"]),
                "final_actual_active_hours": float(final_active["actual_active_hours"]),
                "training_nodes_per_active_second": int(final_active["nodes_touched"]) / max(float(final_active["actual_active_hours"]) * 3600.0, 1e-12),
                "replay_rows": int(final_active["replay_rows"]),
                "unique_information_states": int(final_active["unique_information_states"]),
            }
        )
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", manifests)
    write_csv(output_dir / "snapshot_inventory.csv", inventory)
    write_csv(output_dir / "checkpoint_policy_metrics.csv", internal)
    write_csv(output_dir / "checkpoint_policy_summary.csv", internal_summary)
    write_csv(output_dir / "all_algorithm_checkpoint_metrics.csv", performance)
    write_csv(output_dir / "all_algorithm_checkpoint_summary.csv", performance_summary)
    write_csv(output_dir / "paired_effects_by_seed.csv", effects)
    write_csv(output_dir / "paired_effect_summary.csv", paired_summary)
    write_csv(output_dir / "internal_late_window_by_seed.csv", internal_late)
    write_csv(output_dir / "all_algorithm_late_window_by_seed.csv", performance_late)
    write_csv(output_dir / "final_checkpoint_refinement_audit.csv", audit)
    write_csv(output_dir / "ordinary_refined_head_to_head.csv", head_to_head)
    write_csv(output_dir / "resource_metrics_by_seed.csv", resource_rows)
    _plot_series(
        performance, performance_summary, id_field="algorithm_id",
        order=PERFORMANCE_ORDER, label_map=PERFORMANCE_LABELS,
        output=output_dir / "exploitability_by_training_time.png", by_nodes=False,
        title="Extended-fit UCV and frozen comparators by training time",
    )
    _plot_series(
        performance, performance_summary, id_field="algorithm_id",
        order=PERFORMANCE_ORDER, label_map=PERFORMANCE_LABELS,
        output=output_dir / "exploitability_by_nodes_touched.png", by_nodes=True,
        title="Extended-fit UCV and frozen comparators by nodes touched",
    )
    _plot_series(
        internal, internal_summary, id_field="policy_id", order=INTERNAL_ORDER,
        label_map=POLICY_LABELS,
        output=output_dir / "ordinary_vs_refined_by_training_time.png",
        by_nodes=False, title="Ordinary and refined policies on shared trajectories",
    )
    _plot_gaps(internal, output_dir / "distillation_gap_by_training_time.png")
    _plot_paired_final(effects, output_dir / "paired_final_effect.png", smoke=smoke)
    _plot_final_audit(audit, output_dir / "final_checkpoint_refinement_horizon.png", smoke=smoke)
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(workers),
        "num_playable_snapshots": len(inventory),
        "contract": contract_manifest(),
        "experiment_29_reference_root": str(Path(experiment_29_root).resolve()),
        "experiment_29_reference_contract": historical_manifest.get("contract", {}),
        "inferential_note": (
            "The ordinary and refined Experiment 43 policies are paired on the same "
            "trajectory and reservoir. Experiment 29 historical comparisons use the "
            "same five seed labels and frozen checkpoints, but are separate training "
            "trajectories. The fixed 1,700-update horizon was selected using these "
            "labels in Experiment 41, so all results remain post-selection development evidence."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers"]
