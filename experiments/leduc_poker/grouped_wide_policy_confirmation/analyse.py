"""Aggregate the fresh-seed Experiment 35 confirmation."""

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
from open_spiel.python.algorithms import exploitability  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import (  # noqa: E402
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.policies import (  # noqa: E402
    load_policy,
    validate_policy_probabilities,
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
    GAME_NAME,
    HISTORICAL_ALGORITHM_LABELS,
    HISTORICAL_ALGORITHM_ORDER,
    LEGACY_CONTROL_ID,
    LEGACY_CONTROL_LABEL,
    REFERENCE_EXPERIMENT_29_MANIFEST_SHA256,
    REFERENCE_EXPERIMENT_29_METRICS_SHA256,
    checkpoint_schedule,
    contract_manifest,
)


DIAGNOSTIC_POLICY_ORDER = (
    CANDIDATE_ID,
    LEGACY_CONTROL_ID,
    EMPIRICAL_POLICY_ID,
    EXACT_POLICY_ID,
)
PERFORMANCE_ALGORITHM_ORDER = HISTORICAL_ALGORITHM_ORDER + (CANDIDATE_ID,)
POLICY_LABELS = {
    CANDIDATE_ID: CANDIDATE_LABEL,
    LEGACY_CONTROL_ID: LEGACY_CONTROL_LABEL,
    EMPIRICAL_POLICY_ID: EMPIRICAL_POLICY_LABEL,
    EXACT_POLICY_ID: EXACT_POLICY_LABEL,
}
COLOURS = {
    CANDIDATE_ID: "#2ca02c",
    LEGACY_CONTROL_ID: "#d62728",
    EMPIRICAL_POLICY_ID: "#9467bd",
    EXACT_POLICY_ID: "#1f77b4",
    "deep_cfr": "#1f77b4",
    "unbiased_control_variate_escher": "#d62728",
    "selected_nonpredictive_ucv": "#ff7f0e",
    "promoted_ucv_cross_entropy": "#9467bd",
}
PERFORMANCE_LABELS = {**HISTORICAL_ALGORITHM_LABELS, CANDIDATE_ID: CANDIDATE_LABEL}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_workers(
    workers_root: Path, seeds: Sequence[int], smoke: bool
) -> dict[int, tuple[Path, dict]]:
    expected = {int(seed) for seed in seeds}
    found = {}
    schedule = checkpoint_schedule(smoke=smoke)
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        seed = int(result["seed"])
        if seed in found:
            raise ValueError(f"Duplicate Experiment 35 worker for seed {seed}")
        if (
            result.get("status") != "complete"
            or result.get("candidate_id") != CANDIDATE_ID
            or bool(result.get("smoke")) != bool(smoke)
            or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
        ):
            raise ValueError(f"Incomplete or mismatched Experiment 35 worker: {path}")
        found[seed] = (path, result)
    if set(found) != expected:
        raise ValueError(
            f"Experiment 35 workers differ; missing={sorted(expected-set(found))}, "
            f"extra={sorted(set(found)-expected)}"
        )
    commits = {result["repository_commit"] for _, result in found.values()}
    if len(commits) != 1:
        raise ValueError(f"Workers used different commits: {sorted(commits)}")
    return found


def _metrics(
    workers: Mapping[int, tuple[Path, Mapping]], *, smoke: bool
) -> tuple[list[dict], list[dict], list[dict]]:
    game = pyspiel.load_game(GAME_NAME)
    metrics, inventory, manifests = [], [], []
    for seed in sorted(workers):
        result_path, result = workers[seed]
        root = result_path.parent
        curves = _read_csv(root / result["artifacts"]["checkpoint_curves"])
        for record in result["snapshots"]:
            snapshot = root / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Missing or corrupt snapshot: {snapshot}")
            loaded = load_policy(game, "unbiased_control_variate_escher", snapshot)
            validate_policy_probabilities(game, loaded)
            tabular = policy.tabular_policy_from_callable(
                game, loaded.action_probabilities
            )
            snapshot_exploitability = float(
                exploitability.nash_conv(game, tabular) / 2.0
            )
            row_index = int(record["checkpoint_row_index"])
            if row_index < 0 or row_index >= len(curves):
                raise ValueError("Snapshot refers to an invalid checkpoint row")
            row = curves[row_index]
            if int(row["iteration"]) != int(record["completed_iteration"]):
                raise ValueError("Snapshot and checkpoint curve iterations differ")
            candidate = float(row["neural_average_exploitability"])
            if not np.isclose(snapshot_exploitability, candidate, atol=1e-7, rtol=1e-6):
                raise ValueError("Playable snapshot differs from recorded candidate policy")
            common = {
                "seed": int(seed),
                "checkpoint_id": record["checkpoint_id"],
                "checkpoint_type": record["checkpoint_type"],
                "checkpoint_target_active_hours": record["checkpoint_target_active_hours"],
                "checkpoint_target_nodes": record["checkpoint_target_nodes"],
                "actual_active_hours": float(record["active_seconds"]) / 3600.0,
                "nodes_touched": int(record["nodes_touched"]),
                "completed_iteration": int(record["completed_iteration"]),
            }
            values = {
                CANDIDATE_ID: candidate,
                LEGACY_CONTROL_ID: float(row["legacy_average_exploitability"]),
                EMPIRICAL_POLICY_ID: float(row["empirical_reservoir_exploitability"]),
                EXACT_POLICY_ID: float(row["exact_average_exploitability"]),
            }
            for policy_id, value in values.items():
                metrics.append(
                    {
                        **common,
                        "policy_id": policy_id,
                        "policy_label": POLICY_LABELS[policy_id],
                        "exploitability": value,
                        "candidate_minus_policy": candidate - value,
                    }
                )
            inventory.append({**record, "path": str(snapshot.resolve())})
        manifests.append(
            {
                "seed": seed,
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
                "worker_result": str(result_path.resolve()),
            }
        )
    return metrics, inventory, manifests


def _historical_metrics(
    experiment_29_root: Path, *, smoke: bool
) -> tuple[list[dict], dict]:
    analysis = Path(experiment_29_root).resolve() / "analysis"
    metrics_path = analysis / "combined_checkpoint_policy_metrics.csv"
    manifest_path = analysis / "aggregate_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Experiment 29 comparator artifacts are incomplete under {analysis}"
        )
    if not smoke:
        if sha256(metrics_path) != REFERENCE_EXPERIMENT_29_METRICS_SHA256:
            raise ValueError("Experiment 29 comparator metrics differ from the frozen artifact")
        if sha256(manifest_path) != REFERENCE_EXPERIMENT_29_MANIFEST_SHA256:
            raise ValueError("Experiment 29 comparator manifest differs from the frozen artifact")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or bool(manifest.get("smoke")) != bool(smoke):
        raise ValueError("Experiment 29 comparator manifest has the wrong status or mode")
    if not smoke and int(manifest.get("contract", {}).get("experiment_id", -1)) != 29:
        raise ValueError("Historical comparator source is not Experiment 29")

    expected_checkpoints = {
        str(row["checkpoint_id"])
        for row in checkpoint_schedule(smoke=smoke)
        if row["checkpoint_type"] == "active_time"
    }
    rows = _read_csv(metrics_path)
    observed_algorithms = {str(row.get("algorithm_id")) for row in rows}
    if observed_algorithms != set(HISTORICAL_ALGORITHM_ORDER):
        raise ValueError("Experiment 29 comparator algorithm set differs")
    observed_checkpoints = {str(row.get("checkpoint_id")) for row in rows}
    if observed_checkpoints != expected_checkpoints:
        raise ValueError("Experiment 29 comparator checkpoint schedule differs")
    expected_seeds = {0} if smoke else set(
        int(seed) for seed in manifest["contract"]["production_seeds"]
    )
    if {int(row["seed"]) for row in rows} != expected_seeds:
        raise ValueError("Experiment 29 comparator seeds differ from its manifest")
    expected_keys = {
        (algorithm_id, seed, checkpoint_id)
        for algorithm_id in HISTORICAL_ALGORITHM_ORDER
        for seed in expected_seeds
        for checkpoint_id in expected_checkpoints
    }
    observed_keys = {
        (str(row["algorithm_id"]), int(row["seed"]), str(row["checkpoint_id"]))
        for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("Experiment 29 comparator rows are incomplete or duplicated")

    normalised = []
    for row in rows:
        algorithm_id = str(row["algorithm_id"])
        expected_label = HISTORICAL_ALGORITHM_LABELS[algorithm_id]
        if str(row["algorithm_label"]) != expected_label:
            raise ValueError(f"Historical label changed for {algorithm_id}")
        normalised.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": expected_label,
                "seed": int(row["seed"]),
                "seed_cohort": "experiment_29_historical",
                "comparison_role": "historical_unpaired",
                "checkpoint_id": str(row["checkpoint_id"]),
                "checkpoint_type": "active_time",
                "checkpoint_target_active_hours": float(row["checkpoint_target_active_hours"]),
                "actual_active_hours": float(row["actual_active_hours"]),
                "nodes_touched": int(row["nodes_touched"]),
                "completed_iteration": int(row["completed_iteration"]),
                "exploitability": float(row["exploitability"]),
            }
        )
    return normalised, manifest


def _performance_metrics(
    diagnostic_metrics: Sequence[Mapping], historical_metrics: Sequence[Mapping]
) -> list[dict]:
    fresh = [
        {
            "algorithm_id": CANDIDATE_ID,
            "algorithm_label": CANDIDATE_LABEL,
            "seed": int(row["seed"]),
            "seed_cohort": "experiment_35_fresh",
            "comparison_role": "fresh_confirmation_candidate",
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint_type": "active_time",
            "checkpoint_target_active_hours": float(row["checkpoint_target_active_hours"]),
            "actual_active_hours": float(row["actual_active_hours"]),
            "nodes_touched": int(row["nodes_touched"]),
            "completed_iteration": int(row["completed_iteration"]),
            "exploitability": float(row["exploitability"]),
        }
        for row in diagnostic_metrics
        if row["policy_id"] == CANDIDATE_ID
        and row["checkpoint_type"] == "active_time"
    ]
    return [dict(row) for row in historical_metrics] + fresh


def _performance_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm_id"], row["checkpoint_id"])].append(row)
    result = []
    for algorithm_id in PERFORMANCE_ALGORITHM_ORDER:
        checkpoint_ids = sorted(
            {checkpoint for candidate, checkpoint in grouped if candidate == algorithm_id},
            key=lambda checkpoint: float(grouped[(algorithm_id, checkpoint)][0]["checkpoint_target_active_hours"]),
        )
        for checkpoint_id in checkpoint_ids:
            values = grouped[(algorithm_id, checkpoint_id)]
            observed = [float(row["exploitability"]) for row in values]
            stats = summary(observed)
            result.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": PERFORMANCE_LABELS[algorithm_id],
                    "seed_cohort": values[0]["seed_cohort"],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_target_active_hours": float(values[0]["checkpoint_target_active_hours"]),
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
    return result


def _summaries(metrics: Sequence[Mapping[str, Any]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row["policy_id"], row["checkpoint_id"])].append(row)
    rows = []
    for policy_id in DIAGNOSTIC_POLICY_ORDER:
        ids = sorted(
            {checkpoint for candidate, checkpoint in grouped if candidate == policy_id},
            key=lambda item: (
                grouped[(policy_id, item)][0]["checkpoint_target_active_hours"] is None,
                float(grouped[(policy_id, item)][0]["checkpoint_target_active_hours"] or 0.0),
            ),
        )
        for checkpoint_id in ids:
            values = grouped[(policy_id, checkpoint_id)]
            stats = summary([float(row["exploitability"]) for row in values])
            rows.append(
                {
                    "policy_id": policy_id,
                    "policy_label": POLICY_LABELS[policy_id],
                    "checkpoint_id": checkpoint_id,
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
    return rows


def _paired(metrics: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    indexed = {
        (int(row["seed"]), row["checkpoint_id"], row["policy_id"]): row
        for row in metrics
    }
    candidates = [row for row in metrics if row["policy_id"] == CANDIDATE_ID]
    effects, summaries = [], []
    for comparator in DIAGNOSTIC_POLICY_ORDER[1:]:
        grouped = defaultdict(list)
        for row in candidates:
            key = (int(row["seed"]), row["checkpoint_id"], comparator)
            effect = float(row["exploitability"]) - float(indexed[key]["exploitability"])
            effects.append(
                {
                    "seed": int(row["seed"]),
                    "checkpoint_id": row["checkpoint_id"],
                    "checkpoint_type": row["checkpoint_type"],
                    "checkpoint_target_active_hours": row["checkpoint_target_active_hours"],
                    "comparator_id": comparator,
                    "comparator_label": POLICY_LABELS[comparator],
                    "candidate_minus_comparator": effect,
                }
            )
            grouped[row["checkpoint_id"]].append(effect)
        for checkpoint_id, values in grouped.items():
            stats = summary(values)
            negative = sum(value < 0.0 for value in values)
            # Directional hypothesis was fixed before the fresh runs.  When all
            # five signs favour the candidate this exact value is 1/32.
            one_sided_sign_p = sum(
                math.comb(len(values), k) for k in range(negative, len(values) + 1)
            ) / float(2 ** len(values))
            exemplar = next(row for row in effects if row["checkpoint_id"] == checkpoint_id and row["comparator_id"] == comparator)
            summaries.append(
                {
                    "comparator_id": comparator,
                    "comparator_label": POLICY_LABELS[comparator],
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_type": exemplar["checkpoint_type"],
                    "checkpoint_target_active_hours": exemplar["checkpoint_target_active_hours"],
                    "mean_candidate_minus_comparator": float(stats["mean_ev"]),
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "candidate_better_seed_fraction": negative / float(len(values)),
                    "one_sided_exact_sign_p": one_sided_sign_p,
                    "two_sided_exact_sign_flip_p": float(stats["two_sided_exact_sign_flip_p"]),
                    "n_seeds": len(values),
                }
            )
    return effects, summaries


def _late_window(metrics: Sequence[Mapping], *, smoke: bool) -> list[dict]:
    rows = []
    for seed in sorted({int(row["seed"]) for row in metrics}):
        values = sorted(
            (
                row for row in metrics
                if int(row["seed"]) == seed
                and row["policy_id"] == CANDIDATE_ID
                and row["checkpoint_type"] == "active_time"
            ),
            key=lambda row: float(row["checkpoint_target_active_hours"]),
        )
        late = values if smoke else [row for row in values if float(row["checkpoint_target_active_hours"]) >= 24.0]
        exploits = np.asarray([float(row["exploitability"]) for row in late])
        hours = np.asarray([float(row["checkpoint_target_active_hours"]) for row in late])
        adjacent = np.diff(exploits)
        rows.append(
            {
                "seed": seed,
                "late_window_mean_exploitability": float(np.mean(exploits)),
                "late_window_adjacent_rmssd": float(np.sqrt(np.mean(np.square(adjacent))) if len(adjacent) else 0.0),
                "late_window_max_deterioration": float(max(0.0, np.max(adjacent)) if len(adjacent) else 0.0),
                "late_window_slope_per_hour": float(np.polyfit(hours, exploits, 1)[0] if len(exploits) > 1 else 0.0),
                "final_exploitability": float(exploits[-1]),
                "final_nodes_touched": int(values[-1]["nodes_touched"]),
            }
        )
    return rows


def _performance_seed_metrics(
    rows: Sequence[Mapping], *, smoke: bool
) -> list[dict]:
    result = []
    for algorithm_id in PERFORMANCE_ALGORITHM_ORDER:
        seeds = sorted(
            {int(row["seed"]) for row in rows if row["algorithm_id"] == algorithm_id}
        )
        for seed in seeds:
            trajectory = sorted(
                (
                    row for row in rows
                    if row["algorithm_id"] == algorithm_id and int(row["seed"]) == seed
                ),
                key=lambda row: float(row["checkpoint_target_active_hours"]),
            )
            late = trajectory if smoke else [
                row for row in trajectory
                if float(row["checkpoint_target_active_hours"]) >= 24.0
            ]
            values = np.asarray([float(row["exploitability"]) for row in late])
            adjacent = np.diff(values)
            result.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": PERFORMANCE_LABELS[algorithm_id],
                    "seed": seed,
                    "seed_cohort": trajectory[0]["seed_cohort"],
                    "late_window_mean_exploitability": float(np.mean(values)),
                    "late_window_adjacent_rmssd": float(np.sqrt(np.mean(np.square(adjacent))) if len(adjacent) else 0.0),
                    "late_window_max_deterioration": float(max(0.0, np.max(adjacent)) if len(adjacent) else 0.0),
                    "final_exploitability": float(values[-1]),
                    "final_nodes_touched": int(trajectory[-1]["nodes_touched"]),
                }
            )
    return result


def _welch_comparison(candidate: Sequence[float], reference: Sequence[float]) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    difference = float(np.mean(candidate) - np.mean(reference))
    candidate_variance = float(np.var(candidate, ddof=1)) if len(candidate) > 1 else 0.0
    reference_variance = float(np.var(reference, ddof=1)) if len(reference) > 1 else 0.0
    candidate_component = candidate_variance / float(len(candidate))
    reference_component = reference_variance / float(len(reference))
    standard_error = float(np.sqrt(candidate_component + reference_component))
    denominator = 0.0
    if len(candidate) > 1:
        denominator += candidate_component**2 / float(len(candidate) - 1)
    if len(reference) > 1:
        denominator += reference_component**2 / float(len(reference) - 1)
    degrees_of_freedom = (
        (candidate_component + reference_component) ** 2 / denominator
        if denominator > 0.0
        else float("inf")
    )
    if standard_error > 0.0:
        critical = float(scipy_stats.t.ppf(0.975, degrees_of_freedom))
        statistic = difference / standard_error
        p_value = float(2.0 * scipy_stats.t.sf(abs(statistic), degrees_of_freedom))
    else:
        critical = 0.0
        statistic = 0.0 if difference == 0.0 else math.copysign(float("inf"), difference)
        p_value = 1.0 if difference == 0.0 else 0.0
    return {
        "mean_candidate_minus_reference": difference,
        "standard_error": standard_error,
        "welch_degrees_of_freedom": degrees_of_freedom,
        "welch_t": statistic,
        "two_sided_welch_p": p_value,
        "ci95_lower": difference - critical * standard_error,
        "ci95_upper": difference + critical * standard_error,
        "candidate_n": len(candidate),
        "reference_n": len(reference),
    }


def _unpaired_historical_comparisons(
    seed_rows: Sequence[Mapping]
) -> list[dict]:
    metrics = (
        "final_exploitability",
        "late_window_mean_exploitability",
        "late_window_adjacent_rmssd",
        "late_window_max_deterioration",
        "final_nodes_touched",
    )
    candidate_rows = [
        row for row in seed_rows if row["algorithm_id"] == CANDIDATE_ID
    ]
    result = []
    for reference_id in HISTORICAL_ALGORITHM_ORDER:
        reference_rows = [
            row for row in seed_rows if row["algorithm_id"] == reference_id
        ]
        for metric in metrics:
            comparison = _welch_comparison(
                [float(row[metric]) for row in candidate_rows],
                [float(row[metric]) for row in reference_rows],
            )
            result.append(
                {
                    "reference_algorithm_id": reference_id,
                    "reference_algorithm_label": HISTORICAL_ALGORITHM_LABELS[reference_id],
                    "metric": metric,
                    "comparison_design": "independent five-seed cohorts",
                    **comparison,
                }
            )
    return result


def _plot_performance_trajectory(
    rows: Sequence[Mapping], summaries: Sequence[Mapping], output: Path, *, by_nodes: bool
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    for algorithm_id in PERFORMANCE_ALGORITHM_ORDER:
        algorithm_rows = [row for row in rows if row["algorithm_id"] == algorithm_id]
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                (row for row in algorithm_rows if int(row["seed"]) == seed),
                key=lambda row: int(row["nodes_touched"]) if by_nodes else float(row["checkpoint_target_active_hours"]),
            )
            x = [int(row["nodes_touched"]) / 1e6 if by_nodes else float(row["checkpoint_target_active_hours"]) for row in seed_rows]
            ax.plot(x, [float(row["exploitability"]) for row in seed_rows], color=COLOURS[algorithm_id], alpha=0.10, linewidth=0.75)
        mean_rows = sorted(
            (row for row in summaries if row["algorithm_id"] == algorithm_id),
            key=lambda row: float(row["mean_nodes_touched"]) if by_nodes else float(row["checkpoint_target_active_hours"]),
        )
        x = np.asarray([float(row["mean_nodes_touched"]) / 1e6 if by_nodes else float(row["checkpoint_target_active_hours"]) for row in mean_rows])
        mean = np.asarray([float(row["mean_exploitability"]) for row in mean_rows])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in mean_rows])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in mean_rows])
        linewidth = 2.6 if algorithm_id == CANDIDATE_ID else 1.8
        ax.plot(x, mean, color=COLOURS[algorithm_id], linewidth=linewidth, marker="o", markersize=3.5, label=PERFORMANCE_LABELS[algorithm_id])
        ax.fill_between(x, lower, upper, color=COLOURS[algorithm_id], alpha=0.08)
    ax.set_xlabel("Training nodes touched (millions)" if by_nodes else "Active training time (hours)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    set_chart_title(ax, "Fresh grouped-wide UCV and Experiment 29 comparators by " + ("nodes" if by_nodes else "training time"))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_trajectory(
    metrics: Sequence[Mapping], summaries: Sequence[Mapping], output: Path, *, by_nodes: bool
) -> None:
    rows = [row for row in metrics if row["checkpoint_type"] == "active_time"]
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    for policy_id in DIAGNOSTIC_POLICY_ORDER:
        policy_rows = [row for row in rows if row["policy_id"] == policy_id]
        for seed in sorted({int(row["seed"]) for row in policy_rows}):
            seed_rows = sorted(
                (row for row in policy_rows if int(row["seed"]) == seed),
                key=lambda row: int(row["nodes_touched"]) if by_nodes else float(row["checkpoint_target_active_hours"]),
            )
            x = [int(row["nodes_touched"]) / 1e6 if by_nodes else float(row["checkpoint_target_active_hours"]) for row in seed_rows]
            ax.plot(x, [float(row["exploitability"]) for row in seed_rows], color=COLOURS[policy_id], alpha=0.11, linewidth=0.8)
        mean_rows = sorted(
            (row for row in summaries if row["policy_id"] == policy_id and row["checkpoint_type"] == "active_time"),
            key=lambda row: float(row["mean_nodes_touched"]) if by_nodes else float(row["checkpoint_target_active_hours"]),
        )
        x = np.asarray([float(row["mean_nodes_touched"]) / 1e6 if by_nodes else float(row["checkpoint_target_active_hours"]) for row in mean_rows])
        mean = np.asarray([float(row["mean_exploitability"]) for row in mean_rows])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in mean_rows])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in mean_rows])
        ax.plot(x, mean, color=COLOURS[policy_id], linewidth=2.0, marker="o", markersize=3.5, label=POLICY_LABELS[policy_id])
        ax.fill_between(x, lower, upper, color=COLOURS[policy_id], alpha=0.09)
    ax.set_xlabel("Training nodes touched (millions)" if by_nodes else "Active training time (hours)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    set_chart_title(ax, "Fresh-seed average-policy confirmation by " + ("nodes" if by_nodes else "training time"))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_gaps(effects: Sequence[Mapping], output: Path) -> None:
    rows = [row for row in effects if row["checkpoint_type"] == "active_time"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for comparator in DIAGNOSTIC_POLICY_ORDER[1:]:
        grouped = defaultdict(list)
        for row in rows:
            if row["comparator_id"] == comparator:
                grouped[float(row["checkpoint_target_active_hours"])].append(float(row["candidate_minus_comparator"]))
        hours = sorted(grouped)
        means = [float(np.mean(grouped[hour])) for hour in hours]
        ax.plot(hours, means, marker="o", linewidth=2.0, color=COLOURS[comparator], label=f"Candidate minus {POLICY_LABELS[comparator]}")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xlabel("Active training time (hours)")
    ax.set_ylabel("Exploitability difference")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    set_chart_title(ax, "Paired average-policy gaps on fresh seeds")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _confirmation_decision(
    effects: Sequence[Mapping], metrics: Sequence[Mapping], *, smoke: bool
) -> dict:
    if smoke:
        return {"status": "smoke_only", "passed": True}
    final_id = "time_36h"
    legacy = [
        float(row["candidate_minus_comparator"])
        for row in effects
        if row["checkpoint_id"] == final_id and row["comparator_id"] == LEGACY_CONTROL_ID
    ]
    empirical = [
        float(row["candidate_minus_comparator"])
        for row in effects
        if row["checkpoint_id"] == final_id and row["comparator_id"] == EMPIRICAL_POLICY_ID
    ]
    exact = [
        float(row["candidate_minus_comparator"])
        for row in effects
        if row["checkpoint_id"] == final_id and row["comparator_id"] == EXACT_POLICY_ID
    ]
    checks = {
        "candidate_beats_legacy_on_at_least_four_seeds": sum(value < 0 for value in legacy) >= 4,
        "mean_candidate_minus_legacy_is_negative": float(np.mean(legacy)) < 0.0,
        "maximum_single_seed_legacy_deterioration_at_most_0_003": max(legacy) <= 0.003,
        "mean_neural_minus_exact_gap_at_most_0_005": float(np.mean(exact)) <= 0.005,
        "mean_neural_minus_empirical_gap_at_most_0_005": float(np.mean(empirical)) <= 0.005,
    }
    return {
        "status": "evaluated",
        "passed": all(checks.values()),
        "checks": checks,
        "final_mean_candidate_minus_legacy": float(np.mean(legacy)),
        "final_mean_candidate_minus_empirical": float(np.mean(empirical)),
        "final_mean_candidate_minus_exact": float(np.mean(exact)),
    }


def aggregate_workers(
    *, workers_root: Path, experiment_29_root: Path,
    seeds: Sequence[int], output_dir: Path, smoke: bool
) -> dict:
    workers = _load_workers(Path(workers_root).resolve(), seeds, smoke)
    metrics, inventory, manifests = _metrics(workers, smoke=smoke)
    summaries = _summaries(metrics)
    effects, paired_summaries = _paired(metrics)
    late = _late_window(metrics, smoke=smoke)
    historical, historical_manifest = _historical_metrics(
        experiment_29_root, smoke=smoke
    )
    performance = _performance_metrics(metrics, historical)
    performance_summaries = _performance_summaries(performance)
    performance_seed_rows = _performance_seed_metrics(performance, smoke=smoke)
    unpaired_comparisons = _unpaired_historical_comparisons(
        performance_seed_rows
    )
    decision = _confirmation_decision(effects, metrics, smoke=smoke)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", manifests)
    write_csv(output_dir / "snapshot_inventory.csv", inventory)
    write_csv(output_dir / "checkpoint_policy_metrics.csv", metrics)
    write_csv(output_dir / "checkpoint_policy_summary.csv", summaries)
    write_csv(output_dir / "paired_policy_effects.csv", effects)
    write_csv(output_dir / "paired_policy_summary.csv", paired_summaries)
    write_csv(output_dir / "late_window_metrics_by_seed.csv", late)
    write_csv(output_dir / "historical_checkpoint_policy_metrics.csv", historical)
    write_csv(output_dir / "all_algorithm_checkpoint_policy_metrics.csv", performance)
    write_csv(output_dir / "all_algorithm_checkpoint_summary.csv", performance_summaries)
    write_csv(output_dir / "all_algorithm_late_window_by_seed.csv", performance_seed_rows)
    write_csv(output_dir / "unpaired_historical_comparisons.csv", unpaired_comparisons)
    _plot_performance_trajectory(performance, performance_summaries, output_dir / "exploitability_by_training_time.png", by_nodes=False)
    _plot_performance_trajectory(performance, performance_summaries, output_dir / "exploitability_by_nodes_touched.png", by_nodes=True)
    _plot_trajectory(metrics, summaries, output_dir / "average_policy_diagnostics_by_training_time.png", by_nodes=False)
    _plot_trajectory(metrics, summaries, output_dir / "average_policy_diagnostics_by_nodes_touched.png", by_nodes=True)
    _plot_gaps(effects, output_dir / "paired_policy_gaps.png")
    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(workers),
        "num_playable_snapshots": len(inventory),
        "num_historical_comparator_rows": len(historical),
        "historical_algorithm_ids": list(HISTORICAL_ALGORITHM_ORDER),
        "experiment_29_reference_root": str(Path(experiment_29_root).resolve()),
        "experiment_29_reference_contract": historical_manifest.get("contract", {}),
        "contract": contract_manifest(),
        "confirmation_decision": decision,
        "inferential_note": (
            "All five production seed labels are fresh. The candidate and legacy "
            "policy fits are paired on the same training trajectory and reservoir. "
            "The pre-specified directional sign test can attain p=0.03125 only if "
            "all five seed-level effects favour the candidate; the minimum two-sided "
            "exact sign-flip p-value remains 0.0625. Deep CFR, original UCV-ESCHER, "
            "simplified UCV and the Experiment 29 revised UCV use Experiment 29's "
            "different five-seed cohort. Their comparisons with Experiment 35 are "
            "independent-sample contextual comparisons, not paired confirmation."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers"]
