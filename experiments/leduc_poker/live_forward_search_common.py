"""Shared execution and aggregation for Leduc live-resolving experiments."""

from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import matplotlib
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (
    summary,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402

from escher_poker.live_resolving import (  # noqa: E402
    StitchedResolvedPolicy,
    collect_public_roots,
    solve_public_roots,
)
from escher_poker.policy_snapshots import LoadedESCHERPolicy  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_source_snapshot(source_root: Path, seed: int, checkpoint_id: str) -> Path:
    filename = f"promoted_ucv_cross_entropy_seed_{int(seed)}_{checkpoint_id}.pkl"
    matches = list(Path(source_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one Experiment 29 snapshot named {filename}; "
            f"found {len(matches)} under {source_root}"
        )
    return matches[0]


def exact_metrics(game, candidate, blueprint) -> dict[str, float]:
    tabular = policy.tabular_policy_from_callable(game, candidate.action_probabilities)
    blueprint_tabular = policy.tabular_policy_from_callable(
        game, blueprint.action_probabilities
    )
    nash_conv = float(exploitability.nash_conv(game, tabular))
    as_player_zero = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [tabular, blueprint_tabular]
        )[0]
    )
    as_player_one = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [blueprint_tabular, tabular]
        )[1]
    )
    return {
        "nash_conv": nash_conv,
        "exploitability": nash_conv / 2.0,
        "resolved_ev_vs_blueprint_player_0": as_player_zero,
        "resolved_ev_vs_blueprint_player_1": as_player_one,
        "resolved_ev_vs_blueprint_seat_averaged": 0.5
        * (as_player_zero + as_player_one),
    }


def _root_summary(rows: Sequence[Mapping]) -> dict[str, float]:
    if not rows:
        return {
            "total_online_nodes": 0,
            "mean_online_nodes_per_public_root": 0.0,
            "median_online_nodes_per_public_root": 0.0,
            "max_online_nodes_per_public_root": 0,
            "total_resolve_seconds": 0.0,
            "mean_resolve_seconds_per_public_root": 0.0,
            "median_resolve_seconds_per_public_root": 0.0,
            "p95_resolve_seconds_per_public_root": 0.0,
            "max_resolve_seconds_per_public_root": 0.0,
        }
    nodes = np.asarray([float(row["nodes_touched"]) for row in rows])
    seconds = np.asarray([float(row["solve_seconds"]) for row in rows])
    return {
        "total_online_nodes": int(nodes.sum()),
        "mean_online_nodes_per_public_root": float(nodes.mean()),
        "median_online_nodes_per_public_root": float(np.median(nodes)),
        "max_online_nodes_per_public_root": int(nodes.max()),
        "total_resolve_seconds": float(seconds.sum()),
        "mean_resolve_seconds_per_public_root": float(seconds.mean()),
        "median_resolve_seconds_per_public_root": float(np.median(seconds)),
        "p95_resolve_seconds_per_public_root": float(np.quantile(seconds, 0.95)),
        "max_resolve_seconds_per_public_root": float(seconds.max()),
    }


def run_resolver_worker(
    *,
    experiment_name: str,
    resolver_kind: str | Sequence[str],
    seed: int,
    blueprint,
    source_record: Mapping,
    budgets: Sequence[int],
    resolver_seeds: Sequence[int],
    exploration: float,
    worker_dir: Path,
    smoke: bool,
    repository_commit: str,
    contract: Mapping,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game("leduc_poker")
    roots = collect_public_roots(game, blueprint)
    blueprint_metrics = exact_metrics(game, blueprint, blueprint)
    curve_rows = []
    root_rows = []
    policy_records = []
    resolver_kinds = (
        (resolver_kind,) if isinstance(resolver_kind, str) else tuple(resolver_kind)
    )
    for replicate, resolver_seed in enumerate(resolver_seeds):
        curve_rows.append(
            {
                "experiment_name": experiment_name,
                "resolver_kind": "blueprint",
                "source_seed": int(seed),
                "resolver_replicate": int(replicate),
                "resolver_seed": int(resolver_seed),
                "effort_level": 0,
                "target_updates": 0,
                "num_public_roots": len(roots),
                **_root_summary(()),
                **blueprint_metrics,
            }
        )
        for active_kind in resolver_kinds:
            for effort_level, budget in enumerate(budgets, start=1):
                started = time.perf_counter()
                table, diagnostics = solve_public_roots(
                    roots,
                    resolver_kind=active_kind,
                    target_updates=int(budget),
                    seed=int(resolver_seed),
                    exploration=float(exploration),
                )
                stitched = StitchedResolvedPolicy(game, blueprint, table)
                metric = exact_metrics(game, stitched, blueprint)
                elapsed = time.perf_counter() - started
                policy_path = (
                    worker_dir
                    / "resolved_policies"
                    / f"{active_kind}_replicate_{replicate:02d}_effort_{effort_level:02d}.json"
                )
                write_json(
                    policy_path,
                    {
                        "schema_version": 1,
                        "experiment_name": experiment_name,
                        "resolver_kind": active_kind,
                        "source_seed": int(seed),
                        "resolver_replicate": int(replicate),
                        "resolver_seed": int(resolver_seed),
                        "effort_level": int(effort_level),
                        "target_updates": int(budget),
                        "action_probabilities": table,
                    },
                )
                policy_record = {
                    "resolver_kind": active_kind,
                    "resolver_replicate": int(replicate),
                    "effort_level": int(effort_level),
                    "target_updates": int(budget),
                    "relative_path": str(policy_path.relative_to(worker_dir)),
                    "sha256": sha256(policy_path),
                    "size_bytes": int(policy_path.stat().st_size),
                }
                policy_records.append(policy_record)
                for row in diagnostics:
                    root_rows.append(
                        {
                            "experiment_name": experiment_name,
                            "source_seed": int(seed),
                            "resolver_replicate": int(replicate),
                            "effort_level": int(effort_level),
                            **row,
                        }
                    )
                curve_rows.append(
                    {
                        "experiment_name": experiment_name,
                        "resolver_kind": active_kind,
                        "source_seed": int(seed),
                        "resolver_replicate": int(replicate),
                        "resolver_seed": int(resolver_seed),
                        "effort_level": int(effort_level),
                        "target_updates": int(budget),
                        "num_public_roots": len(roots),
                        "whole_policy_evaluation_seconds": float(elapsed),
                        "resolved_policy_path": policy_record["relative_path"],
                        "resolved_policy_sha256": policy_record["sha256"],
                        **_root_summary(diagnostics),
                        **metric,
                    }
                )

    write_csv(worker_dir / "resolver_curves.csv", curve_rows)
    write_csv(worker_dir / "public_root_diagnostics.csv", root_rows)
    result = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "resolver_kind": list(resolver_kinds),
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": str(repository_commit),
        "contract": dict(contract),
        "source": dict(source_record),
        "num_public_roots": len(roots),
        "num_private_root_histories": int(sum(len(rows) for rows in roots.values())),
        "artifacts": {
            "resolver_curves": "resolver_curves.csv",
            "public_root_diagnostics": "public_root_diagnostics.csv",
        },
        "resolved_policies": policy_records,
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(
        worker_dir / "SUCCESS.json",
        {"status": "complete", "source_seed": int(seed)},
    )
    return result


def load_workers(
    workers_root: Path,
    *,
    experiment_name: str,
    seeds: Sequence[int],
    smoke: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    expected = {int(seed) for seed in seeds}
    results = {}
    for path in Path(workers_root).rglob("worker_result.json"):
        result = read_json(path)
        if result.get("experiment_name") != experiment_name:
            continue
        seed = int(result["source_seed"])
        if seed in results:
            raise ValueError(f"Duplicate {experiment_name} source seed {seed}")
        if result.get("status") != "complete" or bool(result["smoke"]) != bool(smoke):
            raise ValueError(f"Incomplete or mismatched worker {path}")
        for relative in result["artifacts"].values():
            if not (path.parent / relative).is_file():
                raise FileNotFoundError(path.parent / relative)
        for record in result.get("resolved_policies", ()):
            resolved = path.parent / record["relative_path"]
            if not resolved.is_file() or sha256(resolved) != record["sha256"]:
                raise ValueError(f"Missing or corrupt resolved policy {resolved}")
        results[seed] = (path, result)
    if set(results) != expected:
        raise ValueError(f"Worker seeds differ: {sorted(results)} != {sorted(expected)}")
    commits = {result["repository_commit"] for _, result in results.values()}
    if len(commits) != 1:
        raise ValueError("Workers used different repository commits")
    curves, roots, manifest = [], [], []
    for seed in sorted(results):
        path, result = results[seed]
        curves.extend(read_csv(path.parent / result["artifacts"]["resolver_curves"]))
        roots.extend(
            read_csv(path.parent / result["artifacts"]["public_root_diagnostics"])
        )
        manifest.append(
            {
                "source_seed": seed,
                "repository_commit": result["repository_commit"],
                "source_sha256": result["source"].get("sha256", "synthetic"),
                "source_path": result["source"].get("path", "synthetic"),
                "worker_result": str(path.resolve()),
            }
        )
    return curves, roots, manifest


def seed_level_curves(curves: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in curves:
        grouped[
            (
                int(row["source_seed"]),
                str(row["resolver_kind"]),
                int(row["effort_level"]),
            )
        ].append(row)
    result = []
    numeric = (
        "target_updates",
        "total_online_nodes",
        "mean_online_nodes_per_public_root",
        "median_online_nodes_per_public_root",
        "max_online_nodes_per_public_root",
        "total_resolve_seconds",
        "mean_resolve_seconds_per_public_root",
        "median_resolve_seconds_per_public_root",
        "p95_resolve_seconds_per_public_root",
        "max_resolve_seconds_per_public_root",
        "exploitability",
        "nash_conv",
        "resolved_ev_vs_blueprint_seat_averaged",
    )
    for (seed, resolver_kind, effort), rows in sorted(grouped.items()):
        result.append(
            {
                "experiment_name": rows[0]["experiment_name"],
                "resolver_kind": resolver_kind,
                "source_seed": seed,
                "effort_level": effort,
                "n_resolver_replicates": len(rows),
                **{
                    key: float(np.mean([float(row[key]) for row in rows]))
                    for key in numeric
                },
                "within_blueprint_exploitability_sd": float(
                    np.std([float(row["exploitability"]) for row in rows], ddof=1)
                )
                if len(rows) > 1
                else 0.0,
            }
        )
    return result


def aggregate_curve(seed_rows: Sequence[Mapping]) -> list[dict]:
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["resolver_kind"]), int(row["effort_level"]))].append(row)
    result = []
    for (resolver_kind, effort), rows in sorted(grouped.items()):
        record = {
            "experiment_name": rows[0]["experiment_name"],
            "resolver_kind": resolver_kind,
            "effort_level": effort,
            "target_updates": float(np.mean([float(row["target_updates"]) for row in rows])),
            "n_blueprint_seeds": len(rows),
            "n_resolver_replicates_per_blueprint": int(
                round(np.mean([float(row["n_resolver_replicates"]) for row in rows]))
            ),
        }
        for metric in (
            "exploitability",
            "resolved_ev_vs_blueprint_seat_averaged",
            "total_online_nodes",
            "mean_online_nodes_per_public_root",
            "median_resolve_seconds_per_public_root",
            "p95_resolve_seconds_per_public_root",
            "within_blueprint_exploitability_sd",
        ):
            stats = summary([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = stats["mean_ev"]
            record[f"{metric}_standard_error"] = stats["standard_error"]
            record[f"{metric}_ci95_lower"] = stats["ci95_lower"]
            record[f"{metric}_ci95_upper"] = stats["ci95_upper"]
        result.append(record)
    return result


def estimator_variance_summary(root_rows: Sequence[Mapping]) -> list[dict]:
    """Pool node-estimator moments, then summarize across blueprint seeds."""
    grouped = defaultdict(list)
    for row in root_rows:
        if row.get("control_variate_samples", "") in {"", None}:
            continue
        grouped[
            (
                int(row["source_seed"]),
                str(row["resolver_kind"]),
                int(row["effort_level"]),
            )
        ].append(row)
    seed_rows = []
    for (seed, resolver_kind, effort), rows in sorted(grouped.items()):
        counts = np.asarray(
            [float(row["control_variate_samples"]) for row in rows], dtype=np.float64
        )
        means = np.asarray(
            [float(row["control_variate_correction_mean"]) for row in rows],
            dtype=np.float64,
        )
        variances = np.asarray(
            [float(row["control_variate_correction_variance"]) for row in rows],
            dtype=np.float64,
        )
        total = float(counts.sum())
        pooled_mean = float(np.sum(counts * means) / total)
        pooled_variance = float(
            max(0.0, np.sum(counts * (variances + means * means)) / total - pooled_mean**2)
        )
        seed_rows.append(
            {
                "source_seed": seed,
                "resolver_kind": resolver_kind,
                "effort_level": effort,
                "num_correction_samples": int(total),
                "pooled_correction_mean": pooled_mean,
                "pooled_correction_variance": pooled_variance,
            }
        )
    final = []
    aggregate_groups = defaultdict(list)
    for row in seed_rows:
        aggregate_groups[(row["resolver_kind"], row["effort_level"])].append(row)
    for (resolver_kind, effort), rows in sorted(aggregate_groups.items()):
        stats = summary([row["pooled_correction_variance"] for row in rows])
        final.append(
            {
                "resolver_kind": resolver_kind,
                "effort_level": effort,
                "mean_pooled_correction_variance": stats["mean_ev"],
                "standard_error": stats["standard_error"],
                "ci95_lower": stats["ci95_lower"],
                "ci95_upper": stats["ci95_upper"],
                "n_blueprint_seeds": stats["n_seeds"],
                "mean_num_correction_samples": float(
                    np.mean([row["num_correction_samples"] for row in rows])
                ),
            }
        )
    return final


def _plot_curve(rows: Sequence[Mapping], *, x: str, output: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    colours = {
        "blueprint": "#7f7f7f",
        "cfr_plus": "#1f77b4",
        "external_sampling": "#ff7f0e",
        "ucv_external_sampling": "#d62728",
    }
    labels = {
        "blueprint": "Frozen blueprint",
        "cfr_plus": "Tabular CFR+",
        "external_sampling": "Sampled, no control variate",
        "ucv_external_sampling": "UCV sampled",
    }
    for resolver_kind in dict.fromkeys(str(row["resolver_kind"]) for row in rows):
        selected = [row for row in rows if row["resolver_kind"] == resolver_kind]
        x_values = np.asarray([float(row[x]) for row in selected], dtype=np.float64)
        y_values = np.asarray([float(row["exploitability_mean"]) for row in selected])
        lower = np.asarray([float(row["exploitability_ci95_lower"]) for row in selected])
        upper = np.asarray([float(row["exploitability_ci95_upper"]) for row in selected])
        colour = colours.get(resolver_kind, "#2ca02c")
        axis.plot(
            x_values, y_values, marker="o", color=colour,
            label=labels.get(resolver_kind, resolver_kind),
        )
        axis.fill_between(x_values, lower, upper, color=colour, alpha=0.18)
    axis.set_xlabel(
        "Total nodes traversed across all public-root solves"
        if x == "total_online_nodes_mean"
        else "Median online solve time per public root (seconds)"
    )
    axis.set_ylabel("Exact exploitability (NashConv / 2)")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def aggregate_experiment(
    *,
    workers_root: Path,
    output_dir: Path,
    experiment_name: str,
    seeds: Sequence[int],
    smoke: bool,
    contract: Mapping,
) -> dict:
    curves, roots, manifest = load_workers(
        workers_root,
        experiment_name=experiment_name,
        seeds=seeds,
        smoke=smoke,
    )
    seed_rows = seed_level_curves(curves)
    aggregate_rows = aggregate_curve(seed_rows)
    variance_rows = estimator_variance_summary(roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "resolver_curves.csv", curves)
    write_csv(output_dir / "public_root_diagnostics.csv", roots)
    write_csv(output_dir / "seed_level_curves.csv", seed_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate_rows)
    write_csv(output_dir / "estimator_variance_summary.csv", variance_rows)
    write_csv(output_dir / "worker_manifest.csv", manifest)
    _plot_curve(
        aggregate_rows,
        x="total_online_nodes_mean",
        output=output_dir / "exploitability_by_online_nodes.png",
        title="Leduc live resolving: exploitability by online search work",
    )
    _plot_curve(
        aggregate_rows,
        x="median_resolve_seconds_per_public_root_mean",
        output=output_dir / "exploitability_by_decision_latency.png",
        title="Leduc live resolving: exploitability by median decision latency",
    )
    result = {
        "status": "complete",
        "experiment_name": experiment_name,
        "smoke": bool(smoke),
        "num_blueprint_seeds": len(seeds),
        "num_curve_rows": len(curves),
        "contract": dict(contract),
        "artifacts": {
            "resolver_curves": "resolver_curves.csv",
            "public_root_diagnostics": "public_root_diagnostics.csv",
            "seed_level_curves": "seed_level_curves.csv",
            "aggregate_summary": "aggregate_summary.csv",
            "estimator_variance_summary": "estimator_variance_summary.csv",
            "worker_manifest": "worker_manifest.csv",
            "exploitability_by_online_nodes": "exploitability_by_online_nodes.png",
            "exploitability_by_decision_latency": "exploitability_by_decision_latency.png",
        },
    }
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


def add_cfr_comparison(
    *,
    ucv_analysis_dir: Path,
    experiment_31_root: Path,
) -> dict:
    """Join Experiment 31 without treating nested UCV runs as blueprint seeds."""
    cfr_path = Path(experiment_31_root) / "analysis" / "aggregate_summary.csv"
    ucv_path = Path(ucv_analysis_dir) / "aggregate_summary.csv"
    if not cfr_path.is_file():
        raise FileNotFoundError(f"Experiment 31 aggregate not found: {cfr_path}")
    cfr = read_csv(cfr_path)
    ucv = read_csv(ucv_path)
    series = (
        ("Frozen blueprint", [row for row in cfr if row["resolver_kind"] == "blueprint"]),
        ("Tabular CFR+", [row for row in cfr if row["resolver_kind"] == "cfr_plus"]),
        (
            "Sampled, no control variate",
            [row for row in ucv if row["resolver_kind"] == "external_sampling"],
        ),
        (
            "UCV sampled",
            [row for row in ucv if row["resolver_kind"] == "ucv_external_sampling"],
        ),
    )
    joined = []
    for label, rows in series:
        for row in rows:
            joined.append({"resolver_label": label, **row})
    write_csv(Path(ucv_analysis_dir) / "cfr_ucv_comparison.csv", joined)

    for x, filename, xlabel in (
        (
            "total_online_nodes_mean",
            "cfr_ucv_exploitability_by_online_nodes.png",
            "Total nodes traversed across all public-root solves",
        ),
        (
            "median_resolve_seconds_per_public_root_mean",
            "cfr_ucv_exploitability_by_decision_latency.png",
            "Median online solve time per public root (seconds)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 5.2))
        colours = ("#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728")
        for (label, rows), colour in zip(series, colours):
            if not rows:
                continue
            xs = np.asarray([float(row[x]) for row in rows])
            ys = np.asarray([float(row["exploitability_mean"]) for row in rows])
            lower = np.asarray(
                [float(row["exploitability_ci95_lower"]) for row in rows]
            )
            upper = np.asarray(
                [float(row["exploitability_ci95_upper"]) for row in rows]
            )
            axis.plot(xs, ys, marker="o", label=label, color=colour)
            axis.fill_between(xs, lower, upper, color=colour, alpha=0.15)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Exact exploitability (NashConv / 2)")
        axis.set_title("Leduc live resolving: full-tree CFR+ versus UCV sampling")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(Path(ucv_analysis_dir) / filename, dpi=200)
        plt.close(figure)
    return {
        "comparison_csv": "cfr_ucv_comparison.csv",
        "comparison_by_nodes": "cfr_ucv_exploitability_by_online_nodes.png",
        "comparison_by_latency": "cfr_ucv_exploitability_by_decision_latency.png",
    }


def load_blueprint(game, source_root: Path, seed: int, checkpoint_id: str):
    path = find_source_snapshot(source_root, seed, checkpoint_id)
    return LoadedESCHERPolicy(game, path), {
        "source_experiment_id": 29,
        "source_checkpoint_id": checkpoint_id,
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


__all__ = [
    "add_cfr_comparison",
    "aggregate_experiment",
    "load_blueprint",
    "read_csv",
    "read_json",
    "run_resolver_worker",
    "seed_level_curves",
    "sha256",
    "write_csv",
    "write_json",
]
