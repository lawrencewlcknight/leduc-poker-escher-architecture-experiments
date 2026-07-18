"""Run Experiment 6 and compare it with immutable Experiment 2 curves."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
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
    run as exp3,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes import (  # noqa: E402
    run as exp4,
)

from .config import (  # noqa: E402
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_2_SOURCE,
    REFERENCE_CURVES,
    UNBIASED_CONFIG,
)


LOGGER = logging.getLogger("unbiased_control_variate_escher_5x_nodes")

REFERENCE_ALGORITHM_IDS = (
    "escher_exp28",
    "vr_deep_dcfr_plus",
    "vr_deep_pdcfr_plus",
)
ALGORITHMS = {
    "escher_exp28": {"algorithm_label": "ESCHER (Experiment 28, 5x nodes)"},
    "vr_deep_dcfr_plus": {"algorithm_label": "VR-DeepDCFR+"},
    "vr_deep_pdcfr_plus": {"algorithm_label": "VR-DeepPDCFR+"},
    ALGORITHM_ID: {"algorithm_label": ALGORITHM_LABEL},
}

DIAGNOSTIC_FIELDS = (
    "calibration_loss",
    "unbiased_estimator_sample_count",
    "control_variate_beta_mean",
    "control_variate_beta_min",
    "control_variate_beta_max",
    "predicted_residual_variance_mean",
    "q_ensemble_disagreement_mean",
    "q_residual_abs_mean",
    "control_residual_abs_mean",
    "importance_correction_abs_mean",
    "policy_weighted_advantage_abs_mean",
    "full_support_sampling_min_probability",
    "calibration_target_version",
    "q_ensemble_target_version_min",
    "q_ensemble_target_version_max",
    "prediction_gate_player_0",
    "prediction_gate_player_1",
    "prediction_gate_next_player_0",
    "prediction_gate_next_player_1",
    "predictor_relative_skill_player_0",
    "predictor_relative_skill_player_1",
    "predictor_holdout_mse_player_0",
    "predictor_holdout_mse_player_1",
    "predictor_zero_mse_player_0",
    "predictor_zero_mse_player_1",
    "q_fold_0_replay_size",
    "q_fold_1_replay_size",
    "q_fold_2_replay_size",
)


def _parse_float(value) -> float:
    return np.nan if value in {None, ""} else float(value)


def _run_candidate(seed: int, config: Dict[str, Any], target_nodes: int):
    import torch

    from unbiased_escher import UnbiasedControlVariateEscher
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
    solver = UnbiasedControlVariateEscher(**kwargs)
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
            "result_source": "experiment_6_new_run",
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
        "final_nodes_touched": float(final["nodes_touched"]),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "num_iterations_completed": int(final["iteration"]),
        "num_intermediate_points": len(curves),
        "exploitability_normalised_auc_nodes": exp3._normalised_auc(
            nodes,
            exploitabilities,
        ),
        "nodes_to_exploitability_threshold": exp3._first_x_to_threshold(
            nodes,
            exploitabilities,
            EXPLOITABILITY_THRESHOLD,
        ),
        "seconds_to_exploitability_threshold": exp3._first_x_to_threshold(
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
        "result_source": "experiment_6_new_run",
    }
    for field in DIAGNOSTIC_FIELDS:
        summary[f"final_{field}"] = float(final[field])
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
    exp3._write_json(output_path, result)
    return 0


def _run_subprocess(run_dir, seed, config, target_nodes):
    stem = f"{ALGORITHM_ID}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    exp3._write_json(
        input_path,
        {"seed": seed, "config": config, "target_nodes_touched": target_nodes},
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run",
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


def _reference_summaries(reference_rows):
    return exp4._reference_summaries(reference_rows)


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
            field: exp3._stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return aggregate


def _paired_differences(summary_rows):
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for baseline_id in REFERENCE_ALGORITHM_IDS:
        for seed in DEFAULT_SEEDS:
            baseline = indexed.get((baseline_id, seed))
            candidate = indexed.get((ALGORITHM_ID, seed))
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
                }
            )
    return rows


def _plot_combined_exploitability(run_dir, curve_rows):
    colors = {
        "escher_exp28": "#8c564b",
        "vr_deep_dcfr_plus": "#1f77b4",
        "vr_deep_pdcfr_plus": "#2ca02c",
        ALGORITHM_ID: "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algorithm_id, spec in ALGORITHMS.items():
        algorithm_rows = [
            row
            for row in curve_rows
            if row["algorithm_id"] == algorithm_id
            and not bool(row.get("is_final_policy_evaluation", False))
        ]
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                [row for row in algorithm_rows if int(row["seed"]) == seed],
                key=lambda row: row["nodes_touched"],
            )
            ax.plot(
                [row["nodes_touched"] for row in seed_rows],
                [row["exploitability"] for row in seed_rows],
                color=colors[algorithm_id],
                linewidth=1,
                alpha=0.16,
            )
        x, mean, se = exp3._mean_curve(curve_rows, algorithm_id)
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
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Experiment 6 unbiased control-variate ESCHER vs Experiment 2",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "combined_exploitability_by_nodes.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_final(run_dir, summaries):
    labels, means, ses = [], [], []
    for algorithm_id, spec in ALGORITHMS.items():
        values = [
            row["final_exploitability"]
            for row in summaries
            if row["algorithm_id"] == algorithm_id
        ]
        stats = exp3._stats(values)
        labels.append(spec["algorithm_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 6 and Experiment 2: final exploitability")
    fig.tight_layout()
    fig.savefig(
        run_dir / "combined_final_exploitability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _parse_seeds(value):
    if value is None:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (
            args.early_evaluation_nodes,
        )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/unbiased_control_variate_escher_5x_nodes",
    )
    parser.add_argument("--reference-curves", type=Path, default=REFERENCE_CURVES)
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
        raise ValueError("Experiment 6 supports paired seeds 0, 1 and 2")
    config = deepcopy(UNBIASED_CONFIG)
    _apply_overrides(args, config)
    reference_rows = exp4._load_reference_curves(args.reference_curves)
    reference_for_seeds = [
        row for row in reference_rows if int(row["seed"]) in seeds
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root)
        / f"unbiased_control_variate_escher_5x_nodes_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    targets = {
        seed: int(args.target_nodes or EXPERIMENT_2_NODE_TARGETS[seed])
        for seed in seeds
    }
    metadata = {
        "experiment_id": 6,
        "seeds": seeds,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "training_config": config,
        "paired_node_targets": targets,
        "experiment_2_source": EXPERIMENT_2_SOURCE,
        "reference_curves_file": str(args.reference_curves),
        "reference_curves_sha256": exp4._sha256(args.reference_curves),
        "configured_batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "estimator": (
                "Always-unbiased beta*Q control variate with sampled residual "
                "importance correction and policy centering."
            ),
            "cross_fitting": (
                "Every trajectory is assigned to one disjoint critic replay fold "
                "and evaluated by the other critic folds."
            ),
            "adaptation": (
                "Frozen held-out residual calibration selects beta and full-support "
                "variance-directed sampling before the return is observed."
            ),
            "prediction_gate": (
                "Predictive PDCFR+ is mixed with conservative DCFR+ using the "
                "previous held-out predictor skill estimate."
            ),
            "baseline_reuse": (
                "Experiment 2 ESCHER and VR rows are immutable saved results."
            ),
        },
    }
    exp3._write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    for seed in seeds:
        try:
            LOGGER.info("Running Experiment 6 seed %s, target %s", seed, targets[seed])
            result = _run_subprocess(run_dir, seed, config, targets[seed])
            results.append(result)
            exp3._write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {"seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
            )
            exp3._write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("Experiment 6 seed %s failed: %s", seed, exc)
            if not args.continue_on_error:
                return 2

    new_summaries = [result["summary"] for result in results]
    new_curves = [row for result in results for row in result["curves"]]
    combined_curves = [*reference_for_seeds, *new_curves]
    combined_summaries = [
        *_reference_summaries(reference_for_seeds),
        *new_summaries,
    ]
    paired = _paired_differences(combined_summaries)
    aggregate = _aggregate(combined_summaries)

    exp3._write_csv(run_dir / "candidate_seed_summary.csv", new_summaries)
    exp3._write_csv(run_dir / "candidate_checkpoint_curves.csv", new_curves)
    exp3._write_csv(run_dir / "combined_checkpoint_curves.csv", combined_curves)
    exp3._write_csv(run_dir / "combined_seed_summary.csv", combined_summaries)
    exp3._write_csv(run_dir / "paired_differences.csv", paired)
    exp3._write_json(run_dir / "aggregate_summary.json", aggregate)
    exp3._write_json(
        run_dir / "summary.json",
        {
            "candidate_seed_summary": new_summaries,
            "combined_aggregate": aggregate,
            "failures": failures,
        },
    )
    if combined_curves:
        _plot_combined_exploitability(run_dir, combined_curves)
        _plot_final(run_dir, combined_summaries)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
