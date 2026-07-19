"""Run Experiment 7: two VR-Deep algorithms and Experiment 6 to 15M nodes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path("outputs") / ".matplotlib_cache").resolve()),
)
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes import (  # noqa: E402
    run as experiment_2,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes import (  # noqa: E402
    run as experiment_6,
)

from .config import (  # noqa: E402
    ALGORITHMS,
    CANDIDATE_ALGORITHM_ID,
    CANDIDATE_CONFIG,
    DEFAULT_ALGORITHM_IDS,
    DEFAULT_SEEDS,
    EXPECTED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS,
    EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
    EXPERIMENT_ID,
    MAX_NUM_ITERATIONS,
    MEASURED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS,
    MEASURED_RUNTIME_PER_SEED_HOURS,
    MEASURED_SEQUENTIAL_RUNTIME_HOURS,
    PARALLEL_BATCH_TIMEOUT_SECONDS,
    SEQUENTIAL_BATCH_TIMEOUT_SECONDS,
    TARGET_NODES,
    UPSTREAM,
    VR_CONFIG,
)


LOGGER = logging.getLogger("unbiased_escher_vs_vr_deep_cfr_15m_nodes")
RESULT_SOURCE = "experiment_7_new_run"


def _parse_csv_ints(value: str | None, default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_algorithms(value: str | None) -> List[str]:
    if value is None:
        return list(DEFAULT_ALGORITHM_IDS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithm ids: {', '.join(unknown)}")
    if not selected:
        raise ValueError("At least one algorithm id is required")
    return selected


def _apply_overrides(args, vr_config: Dict[str, Any], candidate_config: Dict[str, Any]):
    shared_overrides = {
        "num_traversals": args.traversals,
        "max_num_iterations": args.max_iterations,
        "advantage_network_train_steps": args.advantage_train_steps,
        "ave_policy_network_train_steps": args.policy_train_steps,
        "baseline_network_train_steps": args.q_train_steps,
        "advantage_batch_size": args.batch_size,
        "ave_policy_batch_size": args.batch_size,
        "baseline_batch_size": args.batch_size,
        "advantage_buffer_size": args.buffer_size,
        "ave_policy_buffer_size": args.buffer_size,
        "baseline_buffer_size": args.buffer_size,
    }
    for key, value in shared_overrides.items():
        if value is not None:
            vr_config[key] = value
            candidate_config[key] = value
    candidate_overrides = {
        "calibration_train_steps": args.calibration_train_steps,
        "calibration_batch_size": args.batch_size,
        "calibration_buffer_size": args.buffer_size,
    }
    for key, value in candidate_overrides.items():
        if value is not None:
            candidate_config[key] = value
    if args.early_evaluation_nodes is not None:
        thresholds = (int(args.early_evaluation_nodes),)
        vr_config["early_evaluation_node_thresholds"] = thresholds
        candidate_config["early_evaluation_node_thresholds"] = thresholds


def _mark_result(result: Dict[str, Any], target_nodes: int) -> Dict[str, Any]:
    final_nodes = float(result["summary"]["final_nodes_touched"])
    if final_nodes < target_nodes:
        raise RuntimeError(
            f"Run stopped at {final_nodes:.0f} nodes before target {target_nodes}"
        )
    result["summary"]["result_source"] = RESULT_SOURCE
    result["summary"].setdefault(
        "final_nash_conv_recomputed",
        2.0 * float(result["summary"]["final_exploitability"]),
    )
    for row in result["curves"]:
        row["result_source"] = RESULT_SOURCE
    return result


def _run_algorithm(
    algorithm_id: str,
    seed: int,
    target_nodes: int,
    vr_config: Dict[str, Any],
    candidate_config: Dict[str, Any],
) -> Dict[str, Any]:
    if algorithm_id == CANDIDATE_ALGORITHM_ID:
        return _mark_result(
            experiment_6._run_candidate(seed, candidate_config, target_nodes),
            target_nodes,
        )
    return _mark_result(
        experiment_2._run_vr(algorithm_id, seed, vr_config, target_nodes),
        target_nodes,
    )


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_algorithm(
        str(payload["algorithm_id"]),
        int(payload["seed"]),
        int(payload["target_nodes_touched"]),
        payload["vr_config"],
        payload["candidate_config"],
    )
    shared._write_json(output_path, result)
    return 0


def _run_subprocess(
    run_dir: Path,
    algorithm_id: str,
    seed: int,
    target_nodes: int,
    vr_config: Dict[str, Any],
    candidate_config: Dict[str, Any],
) -> Dict[str, Any]:
    stem = f"{algorithm_id}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    shared._write_json(
        input_path,
        {
            "algorithm_id": algorithm_id,
            "seed": seed,
            "target_nodes_touched": target_nodes,
            "vr_config": vr_config,
            "candidate_config": candidate_config,
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run",
        "--worker-input-json",
        str(input_path),
        "--worker-output-json",
        str(output_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"{stem} failed; see {log_path}")
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate(summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for algorithm_id in ALGORITHMS:
        rows = [row for row in summary_rows if row["algorithm_id"] == algorithm_id]
        if not rows:
            continue
        numeric_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        aggregate[algorithm_id] = {
            field: shared._stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return aggregate


def _paired_differences(summary_rows: Sequence[Mapping[str, Any]]):
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for baseline_id in DEFAULT_ALGORITHM_IDS[:2]:
        for seed in sorted({int(row["seed"]) for row in summary_rows}):
            baseline = indexed.get((baseline_id, seed))
            candidate = indexed.get((CANDIDATE_ALGORITHM_ID, seed))
            if baseline is None or candidate is None:
                continue
            rows.append(
                {
                    "baseline_algorithm_id": baseline_id,
                    "baseline_algorithm_label": ALGORITHMS[baseline_id][
                        "algorithm_label"
                    ],
                    "seed": seed,
                    "exploitability_difference": (
                        candidate["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "nodes_difference": (
                        candidate["final_nodes_touched"]
                        - baseline["final_nodes_touched"]
                    ),
                    "wall_clock_seconds_difference": (
                        candidate["final_wall_clock_seconds"]
                        - baseline["final_wall_clock_seconds"]
                    ),
                }
            )
    return rows


def _mean_curve(rows, algorithm_id: str, x_key: str):
    selected = [
        row
        for row in rows
        if row["algorithm_id"] == algorithm_id
        and not bool(row.get("is_final_policy_evaluation", False))
    ]
    checkpoints = sorted({int(row["checkpoint_index"]) for row in selected})
    xs, means, ses = [], [], []
    for checkpoint in checkpoints:
        at_checkpoint = [
            row for row in selected if int(row["checkpoint_index"]) == checkpoint
        ]
        x = np.asarray([row[x_key] for row in at_checkpoint], dtype=float)
        y = np.asarray([row["exploitability"] for row in at_checkpoint], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            xs.append(float(np.mean(x[finite])))
            means.append(float(np.mean(y[finite])))
            ses.append(float(shared._stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _plot_exploitability(run_dir: Path, curve_rows, *, x_key: str) -> None:
    colors = {
        "vr_deep_dcfr_plus": "#1f77b4",
        "vr_deep_pdcfr_plus": "#2ca02c",
        CANDIDATE_ALGORITHM_ID: "#9467bd",
    }
    is_time = x_key == "wall_clock_seconds"
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algorithm_id, spec in ALGORITHMS.items():
        algorithm_rows = [
            row
            for row in curve_rows
            if row["algorithm_id"] == algorithm_id
            and not bool(row.get("is_final_policy_evaluation", False))
        ]
        if not algorithm_rows:
            continue
        divisor = 3600.0 if is_time else 1.0
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                [row for row in algorithm_rows if int(row["seed"]) == seed],
                key=lambda row: row[x_key],
            )
            ax.plot(
                [row[x_key] / divisor for row in seed_rows],
                [row["exploitability"] for row in seed_rows],
                color=colors[algorithm_id],
                linewidth=1,
                alpha=0.16,
            )
        x, mean, se = _mean_curve(curve_rows, algorithm_id, x_key)
        x = x / divisor
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            color=colors[algorithm_id],
            label=spec["algorithm_label"],
        )
        ax.fill_between(
            x,
            mean - se,
            mean + se,
            color=colors[algorithm_id],
            alpha=0.14,
        )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Wall-clock training time (hours)" if is_time else "Nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    suffix = "wall-clock time" if is_time else "nodes touched"
    set_chart_title(ax, f"Experiment 7 15M-node comparison by {suffix}")
    ax.legend()
    fig.tight_layout()
    filename = (
        "exploitability_by_wall_clock.png"
        if is_time
        else "exploitability_by_nodes.png"
    )
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final(run_dir: Path, summary_rows) -> None:
    labels, means, ses = [], [], []
    for algorithm_id, spec in ALGORITHMS.items():
        values = [
            row["final_exploitability"]
            for row in summary_rows
            if row["algorithm_id"] == algorithm_id
        ]
        if not values:
            continue
        stats = shared._stats(values)
        labels.append(spec["algorithm_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 7 15M-node final exploitability")
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_aggregate_results(run_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    indexed = {}
    for run_dir in run_dirs:
        paths = sorted(run_dir.rglob("worker_results/*.json"))
        if not paths:
            raise ValueError(f"No worker results found under {run_dir}")
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            key = (
                str(result["summary"]["algorithm_id"]),
                int(result["summary"]["seed"]),
            )
            if key in indexed:
                raise ValueError(f"Duplicate aggregate result for {key}")
            indexed[key] = result
    return [indexed[key] for key in sorted(indexed)]


def _finalize(run_dir: Path, results, failures, metadata) -> None:
    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    paired_rows = _paired_differences(summary_rows)
    aggregate = _aggregate(summary_rows)
    shared._write_json(run_dir / "experiment_metadata.json", metadata)
    shared._write_csv(run_dir / "seed_summary.csv", summary_rows)
    shared._write_csv(run_dir / "checkpoint_curves.csv", curve_rows)
    shared._write_csv(run_dir / "paired_differences.csv", paired_rows)
    shared._write_json(run_dir / "aggregate_summary.json", aggregate)
    shared._write_json(
        run_dir / "summary.json",
        {"seed_summary": summary_rows, "aggregate": aggregate, "failures": failures},
    )
    if curve_rows:
        _plot_exploitability(run_dir, curve_rows, x_key="nodes_touched")
        _plot_exploitability(run_dir, curve_rows, x_key="wall_clock_seconds")
        _plot_final(run_dir, summary_rows)


def _metadata(seeds, algorithms, target_nodes, vr_config, candidate_config):
    return {
        "experiment_id": EXPERIMENT_ID,
        "seeds": seeds,
        "algorithm_ids": algorithms,
        "algorithms": ALGORITHMS,
        "target_nodes_touched": target_nodes,
        "vr_config": vr_config,
        "candidate_config": candidate_config,
        "upstream": UPSTREAM,
        "measured_runtime_per_seed_hours": MEASURED_RUNTIME_PER_SEED_HOURS,
        "measured_sequential_runtime_hours": MEASURED_SEQUENTIAL_RUNTIME_HOURS,
        "expected_sequential_runtime_hours": EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
        "sequential_batch_timeout_seconds": SEQUENTIAL_BATCH_TIMEOUT_SECONDS,
        "measured_parallel_by_algorithm_runtime_hours": (
            MEASURED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS
        ),
        "expected_parallel_by_algorithm_runtime_hours": (
            EXPECTED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS
        ),
        "parallel_batch_timeout_seconds": PARALLEL_BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "node_matching": (
                "Every algorithm stops after the first complete outer iteration "
                "crossing the common 15,000,000-node target."
            ),
            "configuration": (
                "VR arms reuse Experiment 2 paper settings; the candidate reuses "
                "the Experiment 6 architecture. Only the iteration safety cap is "
                "raised to 120."
            ),
            "evaluation": (
                "All arms evaluate the untrained policy, at about 10,000 nodes, "
                "and after every outer iteration. Evaluation nodes are excluded."
            ),
            "subsetting": (
                "Algorithms and seeds may be run in independent jobs; worker JSON "
                "outputs can be merged with --aggregate-run-dir."
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/unbiased_escher_vs_vr_deep_cfr_15m_nodes",
    )
    parser.add_argument("--seeds")
    parser.add_argument("--algorithms")
    parser.add_argument("--target-nodes", type=int, default=TARGET_NODES)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--calibration-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument(
        "--aggregate-run-dir",
        action="append",
        type=Path,
        default=[],
        help="Read worker results under this run directory; repeat as needed.",
    )
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_json)
    if args.target_nodes <= 0:
        raise ValueError("target-nodes must be positive")

    seeds = _parse_csv_ints(args.seeds, DEFAULT_SEEDS)
    algorithms = _parse_algorithms(args.algorithms)
    vr_config = deepcopy(VR_CONFIG)
    candidate_config = deepcopy(CANDIDATE_CONFIG)
    _apply_overrides(args, vr_config, candidate_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root)
        / f"unbiased_escher_vs_vr_deep_cfr_15m_nodes_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metadata = _metadata(
        seeds,
        algorithms,
        args.target_nodes,
        vr_config,
        candidate_config,
    )
    if args.aggregate_run_dir:
        results = _load_aggregate_results(args.aggregate_run_dir)
        aggregate_targets = {
            int(result["summary"]["target_nodes_touched"]) for result in results
        }
        if len(aggregate_targets) != 1:
            raise ValueError(
                "Aggregate inputs must use one common target-node budget"
            )
        metadata["aggregate_source_run_dirs"] = [
            str(path) for path in args.aggregate_run_dir
        ]
        metadata["target_nodes_touched"] = aggregate_targets.pop()
        metadata["algorithm_ids"] = sorted(
            {result["summary"]["algorithm_id"] for result in results}
        )
        metadata["seeds"] = sorted(
            {int(result["summary"]["seed"]) for result in results}
        )
        _finalize(run_dir, results, [], metadata)
        LOGGER.info("Aggregated outputs saved to %s", run_dir.resolve())
        return 0

    results, failures = [], []
    schedule = [(algorithm_id, seed) for algorithm_id in algorithms for seed in seeds]
    for algorithm_id, seed in schedule:
        try:
            LOGGER.info(
                "Running Experiment 7 %s seed %s to %s nodes",
                algorithm_id,
                seed,
                args.target_nodes,
            )
            result = _run_subprocess(
                run_dir,
                algorithm_id,
                seed,
                args.target_nodes,
                vr_config,
                candidate_config,
            )
            results.append(result)
            shared._write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {
                    "algorithm_id": algorithm_id,
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            shared._write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("%s seed %s failed: %s", algorithm_id, seed, exc)
            if not args.continue_on_error:
                return 2

    _finalize(run_dir, results, failures, metadata)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
