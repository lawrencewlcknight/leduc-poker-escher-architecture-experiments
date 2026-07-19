"""Run Experiment 12 and compare it with immutable Experiment 6 results."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List

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
    DEFAULT_FINAL_WINDOW,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes import (  # noqa: E402
    run as experiment_6,
)

from .config import (  # noqa: E402
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_SEEDS,
    EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_6_SOURCE,
    EXPERIMENT_ID,
    PARALLEL_MULTI_ACTION_CONFIG,
    REFERENCE_ALGORITHM_ID,
    REFERENCE_ALGORITHM_LABEL,
    REFERENCE_CURVE_ROWS,
    REFERENCE_CURVES,
    REFERENCE_CURVES_SHA256,
    REFERENCE_SUMMARIES,
    REFERENCE_SUMMARIES_SHA256,
    REFERENCE_SUMMARY_ROWS,
)


LOGGER = logging.getLogger("parallel_multi_action_residual_escher_5x_nodes")
RESULT_SOURCE = "experiment_12_new_run"
REFERENCE_SOURCE = "saved_experiment_6"
ALGORITHMS = {
    REFERENCE_ALGORITHM_ID: {"algorithm_label": REFERENCE_ALGORITHM_LABEL},
    ALGORITHM_ID: {"algorithm_label": ALGORITHM_LABEL},
}
COLORS = {
    REFERENCE_ALGORITHM_ID: "#9467bd",
    ALGORITHM_ID: "#ff7f0e",
}
MULTI_ACTION_DIAGNOSTIC_FIELDS = (
    "subset_information_set_count",
    "sampled_subset_size_mean",
    "sampled_subset_size_max",
    "expected_subset_size_mean",
    "multi_action_information_set_fraction",
    "raw_action_inclusion_probability_mean",
    "raw_action_inclusion_probability_min",
    "raw_action_inclusion_probability_max",
    "action_inclusion_probability_mean",
    "action_inclusion_probability_min",
    "action_inclusion_probability_max",
    "raw_empty_subset_probability_mean",
    "predicted_regret_noise_scale_mean",
    "subset_diagonal_variance_proxy_mean",
    "coupled_return_pair_squared_difference_mean",
    "common_random_number_group_count",
    "actual_parallel_action_batch_count",
    "ideal_parallel_node_speedup",
    "ideal_parallelisable_node_fraction",
)
DIAGNOSTIC_FIELDS = tuple(
    dict.fromkeys((*experiment_6.DIAGNOSTIC_FIELDS, *MULTI_ACTION_DIAGNOSTIC_FIELDS))
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float(value) -> float:
    return np.nan if value in {None, ""} else float(value)


def _load_reference_curves(path: Path):
    digest = _sha256(path)
    if digest != REFERENCE_CURVES_SHA256:
        raise ValueError(
            f"Experiment 6 curve checksum mismatch: expected "
            f"{REFERENCE_CURVES_SHA256}, found {digest}"
        )
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in ("seed", "checkpoint_index", "iteration", "episode"):
                row[field] = int(float(row[field]))
            for field in (
                "nodes_touched",
                "wall_clock_seconds",
                "exploitability",
                "average_policy_value",
                "policy_value_error",
            ):
                row[field] = _parse_float(row.get(field))
            row["is_initial_policy_evaluation"] = shared._parse_bool(
                row.get("is_initial_policy_evaluation", False)
            )
            row["is_final_policy_evaluation"] = shared._parse_bool(
                row.get("is_final_policy_evaluation", False)
            )
            row["algorithm_label"] = REFERENCE_ALGORITHM_LABEL
            row["result_source"] = REFERENCE_SOURCE
            rows.append(row)
    if len(rows) != REFERENCE_CURVE_ROWS:
        raise ValueError(
            f"Expected {REFERENCE_CURVE_ROWS} Experiment 6 curves, found {len(rows)}"
        )
    if {row["algorithm_id"] for row in rows} != {REFERENCE_ALGORITHM_ID}:
        raise ValueError("Experiment 6 curves contain an unexpected algorithm")
    if {int(row["seed"]) for row in rows} != set(DEFAULT_SEEDS):
        raise ValueError("Experiment 6 curves must contain seeds 0, 1 and 2")
    return rows


def _load_reference_summaries(path: Path):
    digest = _sha256(path)
    if digest != REFERENCE_SUMMARIES_SHA256:
        raise ValueError(
            f"Experiment 6 summary checksum mismatch: expected "
            f"{REFERENCE_SUMMARIES_SHA256}, found {digest}"
        )
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = dict(raw)
            for key, value in list(row.items()):
                if key in {"algorithm_id", "algorithm_label", "result_source"}:
                    continue
                try:
                    row[key] = _parse_float(value)
                except ValueError:
                    pass
            row["seed"] = int(row["seed"])
            row["algorithm_label"] = REFERENCE_ALGORITHM_LABEL
            row["result_source"] = REFERENCE_SOURCE
            rows.append(row)
    if len(rows) != REFERENCE_SUMMARY_ROWS:
        raise ValueError(
            f"Expected {REFERENCE_SUMMARY_ROWS} Experiment 6 summaries, found {len(rows)}"
        )
    if {row["algorithm_id"] for row in rows} != {REFERENCE_ALGORITHM_ID}:
        raise ValueError("Experiment 6 summaries contain an unexpected algorithm")
    return rows


def _run_candidate(seed: int, config: Dict[str, Any], target_nodes: int):
    import torch

    from parallel_multi_action_escher import ParallelMultiActionResidualEscher
    from vr_deep_cfr.logger import Logger

    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    solver = ParallelMultiActionResidualEscher(**kwargs)
    solver.target_nodes_touched = int(target_nodes)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )
    raw_checkpoints = solver.solve()

    curves = []
    for checkpoint_index, raw in enumerate(raw_checkpoints):
        value = float(raw["average_policy_value"])
        row = {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_label": ALGORITHM_LABEL,
            "seed": int(seed),
            "checkpoint_index": int(checkpoint_index),
            "iteration": int(raw["iteration"]),
            "episode": int(raw["episode"]),
            "nodes_touched": float(raw["nodes_touched"]),
            "wall_clock_seconds": float(raw["wall_clock_seconds"]),
            "exploitability": float(raw["exp"]),
            "average_policy_value": value,
            "policy_value_error": abs(value - LEDUC_GAME_VALUE_PLAYER_0),
            "average_policy_loss": _parse_float(raw.get("average_policy_loss")),
            "regret_loss_player_0": _parse_float(raw.get("regret_loss_0")),
            "regret_loss_player_1": _parse_float(raw.get("regret_loss_1")),
            "baseline_loss_player_0": _parse_float(raw.get("baseline_loss_0")),
            "baseline_loss_player_1": _parse_float(raw.get("baseline_loss_1")),
            "checkpoint_kind": str(raw.get("checkpoint_kind", "outer_iteration")),
            "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
            "is_initial_policy_evaluation": (
                raw.get("checkpoint_kind") == "initial_untrained_policy"
            ),
            "is_final_policy_evaluation": False,
            "result_source": RESULT_SOURCE,
        }
        for field in DIAGNOSTIC_FIELDS:
            row[field] = _parse_float(raw.get(field))
        curves.append(row)

    final = curves[-1]
    training_curves = [
        row for row in curves if not row["is_initial_policy_evaluation"]
    ]
    exploitabilities = [row["exploitability"] for row in training_curves]
    nodes = [row["nodes_touched"] for row in training_curves]
    wall_times = [row["wall_clock_seconds"] for row in training_curves]
    node_delta = float(final["nodes_touched"] - target_nodes)
    summary = {
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "seed": int(seed),
        "final_exploitability": float(final["exploitability"]),
        "best_exploitability": float(np.min(exploitabilities)),
        "final_window_mean_exploitability": float(
            np.mean(
                exploitabilities[
                    -min(DEFAULT_FINAL_WINDOW, len(exploitabilities)) :
                ]
            )
        ),
        "final_policy_value": float(final["average_policy_value"]),
        "final_policy_value_error": float(final["policy_value_error"]),
        "final_nash_conv_recomputed": 2.0 * float(final["exploitability"]),
        "final_nodes_touched": float(final["nodes_touched"]),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "num_iterations_completed": int(final["iteration"]),
        "num_intermediate_points": len(curves),
        "exploitability_normalised_auc_nodes": shared._normalised_auc(
            nodes,
            exploitabilities,
        ),
        "nodes_to_exploitability_threshold": shared._first_x_to_threshold(
            nodes,
            exploitabilities,
            EXPLOITABILITY_THRESHOLD,
        ),
        "seconds_to_exploitability_threshold": shared._first_x_to_threshold(
            wall_times,
            exploitabilities,
            EXPLOITABILITY_THRESHOLD,
        ),
        "target_nodes_touched": float(target_nodes),
        "node_budget_delta": node_delta,
        "node_budget_relative_delta": node_delta / float(target_nodes),
        "final_average_policy_buffer_size": len(solver.ave_policy_trainer.buffer),
        "final_advantage_buffer_size_player_0": len(
            solver.regret_trainers[0].buffer
        ),
        "final_advantage_buffer_size_player_1": len(
            solver.regret_trainers[1].buffer
        ),
        "final_history_value_buffer_size": int(
            sum(solver.q_value_trainer.fold_sizes())
        ),
        "result_source": RESULT_SOURCE,
    }
    for field in DIAGNOSTIC_FIELDS:
        summary[f"final_{field}"] = float(final[field])

    if float(final["policy_weighted_advantage_abs_mean"]) > 1e-10:
        raise RuntimeError("Control-variate advantages were not policy-centred")
    if min(solver.q_value_trainer.fold_sizes()) <= 0:
        raise RuntimeError("An Experiment 6 Q fold received no transitions")
    minimum_floor = float(config["sampling_uniform_floor_mass"]) / float(
        solver.action_size
    )
    if float(final["action_inclusion_probability_min"]) < minimum_floor - 1e-12:
        raise RuntimeError("Action-subset sampler violated its support floor")
    if float(final["sampled_subset_size_mean"]) < 1.0:
        raise RuntimeError("Action-subset sampler produced an empty subset")
    if not np.isfinite(final["ideal_parallel_node_speedup"]):
        raise RuntimeError("Parallel rollout speedup diagnostic was not finite")
    del solver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"seed": int(seed), "summary": summary, "curves": curves}


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_candidate(
        int(payload["seed"]),
        payload["config"],
        int(payload["target_nodes_touched"]),
    )
    shared._write_json(output_path, result)
    return 0


def _run_subprocess(run_dir, seed, config, target_nodes):
    stem = f"{ALGORITHM_ID}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    shared._write_json(
        input_path,
        {"seed": seed, "config": config, "target_nodes_touched": target_nodes},
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run",
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


def _aggregate(summary_rows):
    aggregate = {}
    for algorithm_id in ALGORITHMS:
        rows = [row for row in summary_rows if row["algorithm_id"] == algorithm_id]
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


def _paired_differences(summary_rows):
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for seed in DEFAULT_SEEDS:
        baseline = indexed.get((REFERENCE_ALGORITHM_ID, seed))
        candidate = indexed.get((ALGORITHM_ID, seed))
        if baseline is None or candidate is None:
            continue
        rows.append(
            {
                "seed": seed,
                "exploitability_difference_vs_experiment_6": (
                    candidate["final_exploitability"]
                    - baseline["final_exploitability"]
                ),
                "normalised_auc_difference_vs_experiment_6": (
                    candidate["exploitability_normalised_auc_nodes"]
                    - baseline["exploitability_normalised_auc_nodes"]
                ),
                "nodes_difference_vs_experiment_6": (
                    candidate["final_nodes_touched"]
                    - baseline["final_nodes_touched"]
                ),
                "wall_clock_seconds_difference_vs_experiment_6": (
                    candidate["final_wall_clock_seconds"]
                    - baseline["final_wall_clock_seconds"]
                ),
                "wall_clock_ratio_vs_experiment_6": (
                    candidate["final_wall_clock_seconds"]
                    / baseline["final_wall_clock_seconds"]
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
        current = [
            row for row in selected if int(row["checkpoint_index"]) == checkpoint
        ]
        x = np.asarray([row[x_key] for row in current], dtype=float)
        y = np.asarray([row["exploitability"] for row in current], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            xs.append(float(np.mean(x[finite])))
            means.append(float(np.mean(y[finite])))
            ses.append(float(shared._stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _plot_exploitability(run_dir, rows, *, x_key: str):
    is_time = x_key == "wall_clock_seconds"
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algorithm_id, spec in ALGORITHMS.items():
        algorithm_rows = [row for row in rows if row["algorithm_id"] == algorithm_id]
        divisor = 3600.0 if is_time else 1.0
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                [row for row in algorithm_rows if int(row["seed"]) == seed],
                key=lambda row: row[x_key],
            )
            ax.plot(
                [row[x_key] / divisor for row in seed_rows],
                [row["exploitability"] for row in seed_rows],
                color=COLORS[algorithm_id],
                linewidth=1,
                alpha=0.16,
            )
        x, mean, se = _mean_curve(rows, algorithm_id, x_key)
        ax.plot(
            x / divisor,
            mean,
            marker="o",
            linewidth=2.2,
            color=COLORS[algorithm_id],
            label=spec["algorithm_label"],
        )
        ax.fill_between(
            x / divisor,
            mean - se,
            mean + se,
            color=COLORS[algorithm_id],
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
    dimension = "wall-clock time" if is_time else "nodes touched"
    set_chart_title(
        ax,
        f"Experiment 12 parallel multi-action ESCHER vs Experiment 6 by {dimension}",
    )
    ax.legend()
    fig.tight_layout()
    filename = (
        "combined_exploitability_by_wall_clock.png"
        if is_time
        else "combined_exploitability_by_nodes.png"
    )
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_final(run_dir, summaries):
    labels, means, ses = [], [], []
    for algorithm_id, spec in ALGORITHMS.items():
        values = [
            row["final_exploitability"]
            for row in summaries
            if row["algorithm_id"] == algorithm_id
        ]
        stats = shared._stats(values)
        labels.append(spec["algorithm_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        np.arange(len(labels)),
        means,
        yerr=ses,
        color=[COLORS[key] for key in ALGORITHMS],
        capsize=5,
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=8, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 12 and Experiment 6: final exploitability")
    fig.tight_layout()
    fig.savefig(run_dir / "combined_final_exploitability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_multi_action_diagnostics(run_dir, rows):
    candidate = [
        row
        for row in rows
        if row["algorithm_id"] == ALGORITHM_ID
        and not row.get("is_initial_policy_evaluation", False)
    ]
    if not candidate:
        return
    checkpoints = sorted({int(row["checkpoint_index"]) for row in candidate})
    x, realised_size, expected_size, multi_fraction = [], [], [], []
    speedup, parallelisable, parallel_batches = [], [], []
    for checkpoint in checkpoints:
        current = [
            row for row in candidate if int(row["checkpoint_index"]) == checkpoint
        ]
        x.append(float(np.mean([row["nodes_touched"] for row in current])))
        realised_size.append(
            shared._stats(
                row["sampled_subset_size_mean"]
                for row in current
            )["mean"]
        )
        expected_size.append(
            shared._stats(
                row["expected_subset_size_mean"] for row in current
            )["mean"]
        )
        multi_fraction.append(
            shared._stats(
                row["multi_action_information_set_fraction"] for row in current
            )["mean"]
        )
        speedup.append(
            shared._stats(
                row["ideal_parallel_node_speedup"] for row in current
            )["mean"]
        )
        parallelisable.append(
            shared._stats(
                row["ideal_parallelisable_node_fraction"] for row in current
            )["mean"]
        )
        parallel_batches.append(
            shared._stats(
                row["actual_parallel_action_batch_count"] for row in current
            )["mean"]
        )

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, realised_size, marker="o", label="Realised subset size")
    ax.plot(x, expected_size, marker="o", label="Expected subset size")
    ax.plot(x, multi_fraction, marker="o", label="Multi-action fraction")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Subset diagnostic")
    set_chart_title(ax, "Experiment 12 adaptive action-subset diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "multi_action_subset_by_nodes.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, speedup, marker="o", label="Ideal node speedup")
    ax.plot(x, parallelisable, marker="o", label="Parallelisable node fraction")
    ax.plot(x, parallel_batches, marker="o", label="Executed parallel batches")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Parallel rollout diagnostic")
    set_chart_title(ax, "Experiment 12 action-rollout parallelism")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "multi_action_parallelism_by_nodes.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _parse_seeds(value):
    if value is None:
        return list(DEFAULT_SEEDS)
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _apply_overrides(args, config):
    overrides = {
        "num_traversals": args.traversals,
        "max_num_iterations": args.max_iterations,
        "advantage_network_train_steps": args.advantage_train_steps,
        "ave_policy_network_train_steps": args.policy_train_steps,
        "baseline_network_train_steps": args.q_train_steps,
        "calibration_train_steps": args.calibration_train_steps,
        "advantage_batch_size": args.batch_size,
        "ave_policy_batch_size": args.batch_size,
        "baseline_batch_size": args.batch_size,
        "calibration_batch_size": args.batch_size,
        "advantage_buffer_size": args.buffer_size,
        "ave_policy_buffer_size": args.buffer_size,
        "baseline_buffer_size": args.buffer_size,
        "calibration_buffer_size": args.buffer_size,
        "subset_rollout_cost_scale": args.subset_rollout_cost_scale,
        "parallel_action_workers": args.parallel_action_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (
            int(args.early_evaluation_nodes),
        )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/parallel_multi_action_residual_escher_5x_nodes",
    )
    parser.add_argument("--reference-curves", type=Path, default=REFERENCE_CURVES)
    parser.add_argument(
        "--reference-summaries",
        type=Path,
        default=REFERENCE_SUMMARIES,
    )
    parser.add_argument("--seeds")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--target-nodes", type=int)
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--calibration-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--subset-rollout-cost-scale", type=float)
    parser.add_argument("--parallel-action-workers", type=int)
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_json)

    seeds = _parse_seeds(args.seeds)
    if any(seed not in EXPERIMENT_2_NODE_TARGETS for seed in seeds):
        raise ValueError("Experiment 12 supports paired seeds 0, 1 and 2")
    if args.target_nodes is not None and args.target_nodes <= 0:
        raise ValueError("target-nodes must be positive")
    config = deepcopy(PARALLEL_MULTI_ACTION_CONFIG)
    _apply_overrides(args, config)
    reference_curves = _load_reference_curves(args.reference_curves)
    reference_summaries = _load_reference_summaries(args.reference_summaries)
    reference_curves = [
        row for row in reference_curves if int(row["seed"]) in seeds
    ]
    reference_summaries = [
        row for row in reference_summaries if int(row["seed"]) in seeds
    ]
    targets = {
        seed: int(args.target_nodes or EXPERIMENT_2_NODE_TARGETS[seed])
        for seed in seeds
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root)
        / f"parallel_multi_action_residual_escher_5x_nodes_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "seeds": seeds,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "training_config": config,
        "paired_node_targets": targets,
        "experiment_6_source": EXPERIMENT_6_SOURCE,
        "reference_curves_file": str(args.reference_curves),
        "reference_curves_sha256": _sha256(args.reference_curves),
        "reference_summaries_file": str(args.reference_summaries),
        "reference_summaries_sha256": _sha256(args.reference_summaries),
        "expected_sequential_runtime_hours": EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
        "configured_batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "single_change": (
                "Experiment 6's one-action traverser sample is replaced by an "
                "adaptive nonempty action subset."
            ),
            "inclusion": (
                "Independent adaptive Bernoulli draws are conditioned on a "
                "nonempty subset; exact conditional inclusion marginals are used."
            ),
            "uncertainty": (
                "The subset expands with the predicted standard deviation of "
                "the centred control-residual correction."
            ),
            "common_random_numbers": (
                "Sibling action rollouts clone separate chance, opponent and "
                "nested-subset random streams."
            ),
            "parallelism": (
                "The first multi-action frontier of every traversal executes "
                "on up to three workers; training events merge deterministically."
            ),
            "unbiasedness": (
                "Every included residual is divided by its exact marginal "
                "inclusion probability before unchanged policy centering."
            ),
            "comparison": (
                "Immutable Experiment 6 candidate curves and summaries are "
                "checksum validated and are not retrained."
            ),
        },
    }
    shared._write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    for seed in seeds:
        try:
            LOGGER.info("Running Experiment 12 seed %s to %s nodes", seed, targets[seed])
            result = _run_subprocess(run_dir, seed, config, targets[seed])
            results.append(result)
            shared._write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {"seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
            )
            shared._write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("Experiment 12 seed %s failed: %s", seed, exc)
            if not args.continue_on_error:
                return 2

    candidate_summaries = [result["summary"] for result in results]
    candidate_curves = [row for result in results for row in result["curves"]]
    combined_summaries = [*reference_summaries, *candidate_summaries]
    combined_curves = [*reference_curves, *candidate_curves]
    paired = _paired_differences(combined_summaries)
    aggregate = _aggregate(combined_summaries)
    shared._write_csv(run_dir / "candidate_seed_summary.csv", candidate_summaries)
    shared._write_csv(run_dir / "candidate_checkpoint_curves.csv", candidate_curves)
    shared._write_csv(run_dir / "combined_seed_summary.csv", combined_summaries)
    shared._write_csv(run_dir / "combined_checkpoint_curves.csv", combined_curves)
    shared._write_csv(run_dir / "paired_differences_vs_experiment_6.csv", paired)
    shared._write_json(run_dir / "aggregate_summary.json", aggregate)
    shared._write_json(
        run_dir / "summary.json",
        {
            "candidate_seed_summary": candidate_summaries,
            "combined_aggregate": aggregate,
            "failures": failures,
        },
    )
    if combined_curves:
        _plot_exploitability(run_dir, combined_curves, x_key="nodes_touched")
        _plot_exploitability(run_dir, combined_curves, x_key="wall_clock_seconds")
        _plot_final(run_dir, combined_summaries)
        _plot_multi_action_diagnostics(run_dir, combined_curves)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
