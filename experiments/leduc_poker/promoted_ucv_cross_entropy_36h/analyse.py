"""Exact joined analysis for Experiment 29 and immutable comparators."""

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
    CANDIDATE_ID,
    CANDIDATE_LABEL,
    GAME_NAME,
    REFERENCE_ALGORITHM_IDS,
    REFERENCE_EXPERIMENT_21_METRICS_SHA256,
    REFERENCE_EXPERIMENT_24_METRICS_SHA256,
    checkpoint_schedule,
    contract_manifest,
)


ALGORITHM_ORDER = REFERENCE_ALGORITHM_IDS + (CANDIDATE_ID,)
ALGORITHM_LABELS = {
    "deep_cfr": "Deep CFR",
    "unbiased_control_variate_escher": "Original UCV-ESCHER",
    "selected_nonpredictive_ucv": "Simplified UCV",
    CANDIDATE_ID: CANDIDATE_LABEL,
}
COLOURS = {
    "deep_cfr": "#1f77b4",
    "unbiased_control_variate_escher": "#d62728",
    "selected_nonpredictive_ucv": "#ff7f0e",
    CANDIDATE_ID: "#2ca02c",
}
POLICY_LOADER_IDS = {
    "deep_cfr": "deep_cfr",
    "unbiased_control_variate_escher": "unbiased_control_variate_escher",
    "selected_nonpredictive_ucv": "unbiased_control_variate_escher",
    CANDIDATE_ID: "unbiased_control_variate_escher",
}


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _time_checkpoint_ids(*, smoke: bool) -> set[str]:
    return {
        str(row["checkpoint_id"])
        for row in checkpoint_schedule(smoke=smoke)
        if row["checkpoint_type"] == "active_time"
    }


def _validate_reference(
    root: Path,
    *,
    kind: str,
    seeds: Sequence[int],
    smoke: bool,
) -> tuple[list[dict], list[dict]]:
    analysis = Path(root) / "analysis"
    if kind == "experiment_21":
        metrics_path = analysis / "checkpoint_policy_metrics.csv"
        expected_algorithms = ("deep_cfr", "unbiased_control_variate_escher")
        expected_hash = REFERENCE_EXPERIMENT_21_METRICS_SHA256
    elif kind == "experiment_24":
        metrics_path = analysis / "candidate_checkpoint_policy_metrics.csv"
        expected_algorithms = ("selected_nonpredictive_ucv",)
        expected_hash = REFERENCE_EXPERIMENT_24_METRICS_SHA256
    else:  # pragma: no cover
        raise ValueError(kind)
    inventory_path = analysis / "snapshot_inventory.csv"
    manifest_path = analysis / "aggregate_manifest.json"
    if not all(path.is_file() for path in (metrics_path, inventory_path, manifest_path)):
        raise FileNotFoundError(f"Incomplete {kind} reference under {analysis}")
    if not smoke and sha256(metrics_path) != expected_hash:
        raise ValueError(f"{kind} checkpoint metrics differ from the frozen reference")
    manifest = read_json(manifest_path)
    if bool(manifest.get("smoke")) != bool(smoke):
        raise ValueError(f"{kind} smoke/production status differs")
    metrics = _read_csv(metrics_path)
    inventory = _read_csv(inventory_path)
    time_ids = _time_checkpoint_ids(smoke=smoke)
    metrics = [
        row
        for row in metrics
        if str(row.get("checkpoint_id")) in time_ids
        and str(row.get("algorithm_id", row.get("candidate_id"))) in expected_algorithms
    ]
    observed = {
        (
            str(row.get("algorithm_id", row.get("candidate_id"))),
            int(row["seed"]),
            str(row["checkpoint_id"]),
        )
        for row in metrics
    }
    expected = {
        (algorithm_id, int(seed), checkpoint_id)
        for algorithm_id in expected_algorithms
        for seed in seeds
        for checkpoint_id in time_ids
    }
    if observed != expected:
        raise ValueError(f"{kind} checkpoint rows differ from the frozen contract")
    for row in metrics:
        row["algorithm_id"] = str(row.get("algorithm_id", row.get("candidate_id")))
        row["algorithm_label"] = ALGORITHM_LABELS[row["algorithm_id"]]
    return metrics, inventory


def _candidate_metrics(
    *, workers_root: Path, seeds: Sequence[int], smoke: bool
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    schedule = checkpoint_schedule(smoke=smoke)
    expected = {(CANDIDATE_ID, int(seed)) for seed in seeds}
    results = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        key = (str(result["candidate_id"]), int(result["seed"]))
        if key in results:
            raise ValueError(f"Duplicate Experiment 29 worker result for {key}")
        if (
            result.get("status") != "complete"
            or bool(result.get("smoke")) != bool(smoke)
            or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
        ):
            raise ValueError(f"Incomplete or mismatched Experiment 29 worker: {path}")
        results[key] = (path, result)
    if set(results) != expected:
        raise ValueError(
            f"Experiment 29 worker mismatch; missing={sorted(expected-set(results))}, "
            f"extra={sorted(set(results)-expected)}"
        )
    commits = {result["repository_commit"] for _, result in results.values()}
    if len(commits) != 1:
        raise ValueError(f"Experiment 29 workers used different commits: {sorted(commits)}")

    game = pyspiel.load_game(GAME_NAME)
    inventory, metrics, curves, manifests = [], [], [], []
    policies = {}
    for key in sorted(results):
        result_path, result = results[key]
        root = result_path.parent
        curve_rows = _read_csv(root / result["artifacts"]["checkpoint_curves"])
        curve_by_iteration = {}
        for row in curve_rows:
            row.update({"candidate_id": CANDIDATE_ID, "seed": key[1]})
            curve_by_iteration[int(row["iteration"])] = row
        curves.extend(curve_rows)
        for record in result["snapshots"]:
            snapshot = root / record["relative_path"]
            if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
                raise ValueError(f"Missing or corrupt Experiment 29 snapshot: {snapshot}")
            loaded = load_policy(game, POLICY_LOADER_IDS[CANDIDATE_ID], snapshot)
            validate_policy_probabilities(game, loaded)
            tabular = policy.tabular_policy_from_callable(game, loaded.action_probabilities)
            policies[(key[1], record["checkpoint_id"])] = tabular
            nash_conv = float(exploitability.nash_conv(game, tabular))
            self_play = float(
                expected_game_score.policy_value(
                    game.new_initial_state(), [tabular, tabular]
                )[0]
            )
            diagnostic = curve_by_iteration.get(int(record["completed_iteration"]), {})
            inventory.append({**record, "path": str(snapshot.resolve())})
            metrics.append(
                {
                    "algorithm_id": CANDIDATE_ID,
                    "algorithm_label": CANDIDATE_LABEL,
                    "seed": key[1],
                    "checkpoint_id": record["checkpoint_id"],
                    "checkpoint_type": record["checkpoint_type"],
                    "checkpoint_target_active_hours": record["checkpoint_target_active_hours"],
                    "checkpoint_target_active_seconds": record["checkpoint_target_active_seconds"],
                    "checkpoint_target_nodes": record["checkpoint_target_nodes"],
                    "actual_active_hours": float(record["active_seconds"]) / 3600.0,
                    "actual_active_seconds": float(record["active_seconds"]),
                    "nodes_touched": int(record["nodes_touched"]),
                    "completed_iteration": int(record["completed_iteration"]),
                    "nash_conv": nash_conv,
                    "exploitability": nash_conv / 2.0,
                    "self_play_value_player_0": self_play,
                    "exact_average_exploitability": diagnostic.get("exact_average_exploitability", ""),
                    "neural_average_exploitability": diagnostic.get("neural_average_exploitability", ""),
                    "average_policy_distillation_gap": diagnostic.get("average_policy_distillation_gap", ""),
                    "current_strategy_exploitability": diagnostic.get("current_strategy_exploitability", ""),
                    "peak_rss_mb": float(result["peak_rss_mb"]),
                    "snapshot_sha256": record["sha256"],
                    "snapshot_path": str(snapshot.resolve()),
                }
            )
        manifests.append(
            {
                "candidate_id": CANDIDATE_ID,
                "seed": key[1],
                "worker_result": str(result_path.resolve()),
                "repository_commit": result["repository_commit"],
                "peak_rss_mb": result["peak_rss_mb"],
            }
        )
    return inventory, metrics, curves, manifests, policies


def _checkpoint_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm_id"], row["checkpoint_id"])].append(row)
    result = []
    for algorithm_id in ALGORITHM_ORDER:
        checkpoint_ids = sorted(
            {row["checkpoint_id"] for row in rows if row["algorithm_id"] == algorithm_id},
            key=lambda checkpoint_id: float(grouped[(algorithm_id, checkpoint_id)][0]["checkpoint_target_active_hours"]),
        )
        for checkpoint_id in checkpoint_ids:
            values = grouped[(algorithm_id, checkpoint_id)]
            stats = summary([float(row["exploitability"]) for row in values])
            result.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHM_LABELS[algorithm_id],
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


def _plot_trajectory(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    x_axis: str,
) -> None:
    time_axis = x_axis == "time"
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    for algorithm_id in ALGORITHM_ORDER:
        algorithm_rows = [row for row in rows if row["algorithm_id"] == algorithm_id]
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                (row for row in algorithm_rows if int(row["seed"]) == seed),
                key=lambda row: float(row["actual_active_hours"]) if time_axis else int(row["nodes_touched"]),
            )
            x = [
                float(row["actual_active_hours"]) if time_axis else int(row["nodes_touched"]) / 1_000_000.0
                for row in seed_rows
            ]
            ax.plot(x, [float(row["exploitability"]) for row in seed_rows], color=COLOURS[algorithm_id], alpha=0.12, linewidth=0.8)
        mean_rows = sorted(
            (row for row in summaries if row["algorithm_id"] == algorithm_id),
            key=lambda row: float(row["checkpoint_target_active_hours"]) if time_axis else float(row["mean_nodes_touched"]),
        )
        x = np.asarray([
            float(row["checkpoint_target_active_hours"]) if time_axis else float(row["mean_nodes_touched"]) / 1_000_000.0
            for row in mean_rows
        ])
        mean = np.asarray([float(row["mean_exploitability"]) for row in mean_rows])
        lower = np.asarray([float(row["ci95_lower_exploitability"]) for row in mean_rows])
        upper = np.asarray([float(row["ci95_upper_exploitability"]) for row in mean_rows])
        ax.plot(x, mean, color=COLOURS[algorithm_id], marker="o", markersize=3.8, linewidth=2.0, label=ALGORITHM_LABELS[algorithm_id])
        ax.fill_between(x, lower, upper, color=COLOURS[algorithm_id], alpha=0.10)
    ax.set_xlabel("Active training time (hours)" if time_axis else "Training nodes touched (millions)")
    ax.set_ylabel("Exact exploitability (NashConv / 2)")
    if time_axis:
        ax.set_xticks(np.arange(2, 37, 2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)
    set_chart_title(ax, "Exploitability by " + ("active training time" if time_axis else "training nodes touched"))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _late_metrics(rows: Sequence[Mapping[str, Any]], *, smoke: bool) -> tuple[list[dict], list[dict]]:
    seed_rows = []
    for algorithm_id in ALGORITHM_ORDER:
        seeds = sorted({int(row["seed"]) for row in rows if row["algorithm_id"] == algorithm_id})
        for seed in seeds:
            time_rows = sorted(
                (row for row in rows if row["algorithm_id"] == algorithm_id and int(row["seed"]) == seed),
                key=lambda row: float(row["checkpoint_target_active_hours"]),
            )
            late = time_rows if smoke else [row for row in time_rows if float(row["checkpoint_target_active_hours"]) >= 24.0]
            values = np.asarray([float(row["exploitability"]) for row in late])
            hours = np.asarray([float(row["checkpoint_target_active_hours"]) for row in late])
            adjacent = np.diff(values)
            slope = float(np.polyfit(hours, values, 1)[0]) if len(values) > 1 else 0.0
            seed_rows.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHM_LABELS[algorithm_id],
                    "seed": seed,
                    "late_window_mean_exploitability": float(np.mean(values)),
                    "late_window_adjacent_rmssd": float(np.sqrt(np.mean(np.square(adjacent))) if len(adjacent) else 0.0),
                    "late_window_max_deterioration": float(max(0.0, float(np.max(adjacent))) if len(adjacent) else 0.0),
                    "late_window_slope_per_hour": slope,
                    "late_window_improvement": float(values[0] - values[-1]),
                    "final_exploitability": float(values[-1]),
                    "final_nodes_touched": int(time_rows[-1]["nodes_touched"]),
                }
            )
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in seed_rows}
    candidate_seeds = sorted({int(row["seed"]) for row in seed_rows if row["algorithm_id"] == CANDIDATE_ID})
    paired = []
    for reference_id in REFERENCE_ALGORITHM_IDS:
        for metric in (
            "late_window_mean_exploitability",
            "late_window_adjacent_rmssd",
            "late_window_max_deterioration",
            "late_window_slope_per_hour",
            "late_window_improvement",
            "final_exploitability",
            "final_nodes_touched",
        ):
            effects = [
                float(indexed[(CANDIDATE_ID, seed)][metric])
                - float(indexed[(reference_id, seed)][metric])
                for seed in candidate_seeds
            ]
            stats = summary(effects)
            paired.append(
                {
                    "reference_algorithm_id": reference_id,
                    "reference_algorithm_label": ALGORITHM_LABELS[reference_id],
                    "metric": metric,
                    "mean_candidate_minus_reference": float(stats["mean_ev"]),
                    "ci95_lower": float(stats["ci95_lower"]),
                    "ci95_upper": float(stats["ci95_upper"]),
                    "positive_seed_fraction": float(stats["positive_seed_fraction"]),
                    "two_sided_exact_sign_flip_p": float(stats["two_sided_exact_sign_flip_p"]),
                    "n_seeds": int(stats["n_seeds"]),
                }
            )
    return seed_rows, paired


def _find_snapshot(root: Path, record: Mapping[str, str]) -> Path:
    matches = list((Path(root) / "workers").rglob(record["filename"]))
    if len(matches) != 1:
        raise ValueError(f"Expected one snapshot named {record['filename']}, found {len(matches)}")
    if sha256(matches[0]) != record["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {matches[0]}")
    return matches[0]


def _reference_inventory_index(
    exp21_inventory: Sequence[Mapping], exp24_inventory: Sequence[Mapping]
) -> dict:
    result = {}
    for row in exp21_inventory:
        result[(str(row["algorithm_id"]), int(row["seed"]), str(row["checkpoint_id"]))] = row
    for row in exp24_inventory:
        result[(str(row.get("candidate_id", "selected_nonpredictive_ucv")), int(row["seed"]), str(row["checkpoint_id"]))] = row
    return result


def _head_to_head(
    *,
    game,
    candidate_policies: Mapping,
    experiment_21_root: Path,
    experiment_24_root: Path,
    experiment_21_inventory: Sequence[Mapping],
    experiment_24_inventory: Sequence[Mapping],
    seeds: Sequence[int],
    smoke: bool,
) -> tuple[list[dict], list[dict]]:
    final_checkpoint = [
        row for row in checkpoint_schedule(smoke=smoke) if row["checkpoint_type"] == "active_time"
    ][-1]["checkpoint_id"]
    indexed = _reference_inventory_index(experiment_21_inventory, experiment_24_inventory)
    rows = []
    for reference_id in REFERENCE_ALGORITHM_IDS:
        reference_root = experiment_24_root if reference_id == "selected_nonpredictive_ucv" else experiment_21_root
        for seed in seeds:
            record = indexed[(reference_id, int(seed), final_checkpoint)]
            loaded = load_policy(game, POLICY_LOADER_IDS[reference_id], _find_snapshot(reference_root, record))
            validate_policy_probabilities(game, loaded)
            reference_policy = policy.tabular_policy_from_callable(game, loaded.action_probabilities)
            candidate = candidate_policies[(int(seed), final_checkpoint)]
            as_player_0 = float(expected_game_score.policy_value(game.new_initial_state(), [candidate, reference_policy])[0])
            as_player_1 = float(expected_game_score.policy_value(game.new_initial_state(), [reference_policy, candidate])[1])
            rows.append(
                {
                    "reference_algorithm_id": reference_id,
                    "reference_algorithm_label": ALGORITHM_LABELS[reference_id],
                    "seed": int(seed),
                    "checkpoint_id": final_checkpoint,
                    "candidate_ev_as_player_0": as_player_0,
                    "candidate_ev_as_player_1": as_player_1,
                    "candidate_seat_averaged_ev": 0.5 * (as_player_0 + as_player_1),
                }
            )
    summaries = []
    for reference_id in REFERENCE_ALGORITHM_IDS:
        stats = summary([
            float(row["candidate_seat_averaged_ev"])
            for row in rows
            if row["reference_algorithm_id"] == reference_id
        ])
        summaries.append(
            {
                "reference_algorithm_id": reference_id,
                "reference_algorithm_label": ALGORITHM_LABELS[reference_id],
                **stats,
            }
        )
    ordered = sorted(range(len(summaries)), key=lambda index: float(summaries[index]["two_sided_exact_sign_flip_p"]))
    running = 0.0
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (len(summaries) - rank) * float(summaries[index]["two_sided_exact_sign_flip_p"]))
        running = max(running, adjusted)
        summaries[index]["holm_adjusted_p"] = running
    return rows, summaries


def _plot_policy_diagnostics(metrics: Sequence[Mapping], output: Path) -> None:
    rows = [
        row for row in metrics
        if row["checkpoint_type"] == "active_time" and row.get("exact_average_exploitability") not in {None, ""}
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["checkpoint_target_active_hours"])].append(row)
    hours = sorted(grouped)
    exact = [np.mean([float(row["exact_average_exploitability"]) for row in grouped[hour]]) for hour in hours]
    neural = [np.mean([float(row["exploitability"]) for row in grouped[hour]]) for hour in hours]
    gap = [n - e for n, e in zip(neural, exact)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(hours, exact, marker="o", label="Exact tabular average")
    axes[0].plot(hours, neural, marker="o", label="Neural average policy")
    axes[0].set_ylabel("Exact exploitability (NashConv / 2)")
    axes[0].legend()
    axes[1].plot(hours, gap, color="#9467bd", marker="o")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Neural minus exact exploitability")
    for ax in axes:
        ax.set_xlabel("Active training time (hours)")
        ax.grid(axis="y", alpha=0.25)
    set_chart_title(axes[0], "Underlying and deployed average policies")
    set_chart_title(axes[1], "Average-policy distillation gap")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_stability(seed_rows: Sequence[Mapping], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(len(ALGORITHM_ORDER))
    late = [np.mean([float(row["late_window_mean_exploitability"]) for row in seed_rows if row["algorithm_id"] == algorithm]) for algorithm in ALGORITHM_ORDER]
    rmssd = [np.mean([float(row["late_window_adjacent_rmssd"]) for row in seed_rows if row["algorithm_id"] == algorithm]) for algorithm in ALGORITHM_ORDER]
    axes[0].bar(x, late, color=[COLOURS[item] for item in ALGORITHM_ORDER])
    axes[1].bar(x, rmssd, color=[COLOURS[item] for item in ALGORITHM_ORDER])
    for ax, ylabel, title in zip(
        axes,
        ("Mean exploitability", "Adjacent-checkpoint RMSSD"),
        ("Performance over 24--36 hours", "Late-window volatility"),
    ):
        ax.set_xticks(x, [ALGORITHM_LABELS[item] for item in ALGORITHM_ORDER], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        set_chart_title(ax, title)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def aggregate_workers(
    *,
    workers_root: Path,
    experiment_21_root: Path,
    experiment_24_root: Path,
    seeds: Sequence[int],
    output_dir: Path,
    smoke: bool,
) -> dict:
    workers_root = Path(workers_root).resolve()
    experiment_21_root = Path(experiment_21_root).resolve()
    experiment_24_root = Path(experiment_24_root).resolve()
    output_dir = Path(output_dir).resolve()
    inventory, candidate_metrics, curves, worker_manifest, candidate_policies = _candidate_metrics(
        workers_root=workers_root, seeds=seeds, smoke=smoke
    )
    experiment_21_metrics, experiment_21_inventory = _validate_reference(
        experiment_21_root, kind="experiment_21", seeds=seeds, smoke=smoke
    )
    experiment_24_metrics, experiment_24_inventory = _validate_reference(
        experiment_24_root, kind="experiment_24", seeds=seeds, smoke=smoke
    )
    time_candidate = [row for row in candidate_metrics if row["checkpoint_type"] == "active_time"]
    combined = [dict(row) for row in experiment_21_metrics + experiment_24_metrics] + time_candidate
    summaries = _checkpoint_summaries(combined)
    seed_metrics, paired_metrics = _late_metrics(combined, smoke=smoke)
    game = pyspiel.load_game(GAME_NAME)
    h2h, h2h_summary = _head_to_head(
        game=game,
        candidate_policies=candidate_policies,
        experiment_21_root=experiment_21_root,
        experiment_24_root=experiment_24_root,
        experiment_21_inventory=experiment_21_inventory,
        experiment_24_inventory=experiment_24_inventory,
        seeds=seeds,
        smoke=smoke,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", worker_manifest)
    write_csv(output_dir / "snapshot_inventory.csv", inventory)
    write_csv(output_dir / "candidate_checkpoint_policy_metrics.csv", candidate_metrics)
    write_csv(output_dir / "candidate_node_endpoint_metrics.csv", [row for row in candidate_metrics if row["checkpoint_type"] == "nodes"])
    write_csv(output_dir / "combined_checkpoint_policy_metrics.csv", combined)
    write_csv(output_dir / "combined_checkpoint_summary.csv", summaries)
    write_csv(output_dir / "late_window_metrics_by_seed.csv", seed_metrics)
    write_csv(output_dir / "paired_candidate_comparisons.csv", paired_metrics)
    write_csv(output_dir / "final_same_seed_head_to_head.csv", h2h)
    write_csv(output_dir / "final_head_to_head_summary.csv", h2h_summary)
    write_csv(output_dir / "training_checkpoint_curves.csv", curves)
    _plot_trajectory(combined, summaries, output_dir / "exploitability_by_training_time.png", x_axis="time")
    _plot_trajectory(combined, summaries, output_dir / "exploitability_by_nodes_touched.png", x_axis="nodes")
    _plot_policy_diagnostics(candidate_metrics, output_dir / "average_policy_diagnostics.png")
    _plot_stability(seed_metrics, output_dir / "late_window_performance_stability.png")

    result = {
        "status": "complete",
        "smoke": bool(smoke),
        "num_workers": len(seeds),
        "num_candidate_snapshots": len(inventory),
        "num_combined_time_rows": len(combined),
        "num_final_same_seed_head_to_head_effects": len(h2h),
        "contract": contract_manifest(),
        "experiment_21_reference_root": str(experiment_21_root),
        "experiment_24_reference_root": str(experiment_24_root),
        "paired_candidate_comparisons": paired_metrics,
        "final_head_to_head_summary": h2h_summary,
        "inferential_note": (
            "Experiment 29 reuses five previously examined Experiment 21 seeds for "
            "paired post-selection development comparisons. It is not a fresh held-out "
            "confirmation; the minimum two-sided exact sign-flip p-value is 0.0625."
        ),
    }
    write_json(output_dir / "aggregate_summary.json", result)
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers"]
