"""Run Experiment 15 and compare it with Experiments 6, 9 and 13."""

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
from typing import Any, Dict, List, Sequence

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
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from experiments.leduc_poker import fixed_beta_reservoir_shared as common  # noqa: E402
from experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes import (  # noqa: E402
    run as experiment_9,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes import (  # noqa: E402
    run as experiment_6,
)

from .config import (  # noqa: E402
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    BATCH_TIMEOUT_SECONDS,
    CANDIDATE_CONFIG,
    DEFAULT_SEEDS,
    EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_6_ALGORITHM_ID,
    EXPERIMENT_6_ALGORITHM_LABEL,
    EXPERIMENT_9_ALGORITHM_ID,
    EXPERIMENT_9_ALGORITHM_LABEL,
    EXPERIMENT_9_CURVE_ROWS,
    EXPERIMENT_9_CURVES,
    EXPERIMENT_9_CURVES_SHA256,
    EXPERIMENT_9_SUMMARIES,
    EXPERIMENT_9_SUMMARIES_SHA256,
    EXPERIMENT_9_SUMMARY_ROWS,
    EXPERIMENT_13_ALGORITHM_ID,
    EXPERIMENT_13_ALGORITHM_LABEL,
    EXPERIMENT_13_CURVE_ROWS,
    EXPERIMENT_13_CURVES,
    EXPERIMENT_13_CURVES_SHA256,
    EXPERIMENT_13_SUMMARIES,
    EXPERIMENT_13_SUMMARIES_SHA256,
    EXPERIMENT_13_SUMMARY_ROWS,
    EXPERIMENT_ID,
    REFERENCE_SOURCES,
)


LOGGER = logging.getLogger("fixed_beta_fast_slow_escher_5x_nodes")
RESULT_SOURCE = "experiment_15_new_run"
REFERENCE_SOURCE_9 = "saved_experiments_6_and_9"
REFERENCE_SOURCE_13 = "saved_experiment_13"
ALGORITHM_IDS = (
    EXPERIMENT_6_ALGORITHM_ID,
    EXPERIMENT_9_ALGORITHM_ID,
    EXPERIMENT_13_ALGORITHM_ID,
    ALGORITHM_ID,
)
ALGORITHM_LABELS = {
    EXPERIMENT_6_ALGORITHM_ID: EXPERIMENT_6_ALGORITHM_LABEL,
    EXPERIMENT_9_ALGORITHM_ID: EXPERIMENT_9_ALGORITHM_LABEL,
    EXPERIMENT_13_ALGORITHM_ID: EXPERIMENT_13_ALGORITHM_LABEL,
    ALGORITHM_ID: ALGORITHM_LABEL,
}
COLORS = {
    EXPERIMENT_6_ALGORITHM_ID: "#9467bd",
    EXPERIMENT_9_ALGORITHM_ID: "#d62728",
    EXPERIMENT_13_ALGORITHM_ID: "#7f7f7f",
    ALGORITHM_ID: "#2ca02c",
}
FAST_SLOW_DIAGNOSTIC_FIELDS = tuple(
    dict.fromkeys(
        (
            *experiment_9.FAST_SLOW_DIAGNOSTIC_FIELDS,
            "control_replay_rng_isolated",
            "control_replay_rng_seed",
        )
    )
)
DIAGNOSTIC_FIELDS = tuple(
    dict.fromkeys((*experiment_6.DIAGNOSTIC_FIELDS, *FAST_SLOW_DIAGNOSTIC_FIELDS))
)


def _run_candidate(seed: int, config: Dict[str, Any], target_nodes: int):
    import torch

    from fixed_beta_fast_slow_escher import (
        FixedBetaFastSlowControlCriticEscher,
    )
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
    solver = FixedBetaFastSlowControlCriticEscher(**kwargs)
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
            "average_policy_loss": common.parse_float(
                raw.get("average_policy_loss")
            ),
            "regret_loss_player_0": common.parse_float(raw.get("regret_loss_0")),
            "regret_loss_player_1": common.parse_float(raw.get("regret_loss_1")),
            "baseline_loss_player_0": common.parse_float(
                raw.get("baseline_loss_0")
            ),
            "baseline_loss_player_1": common.parse_float(
                raw.get("baseline_loss_1")
            ),
            "checkpoint_kind": str(
                raw.get("checkpoint_kind", "outer_iteration")
            ),
            "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
            "is_initial_policy_evaluation": (
                raw.get("checkpoint_kind") == "initial_untrained_policy"
            ),
            "is_final_policy_evaluation": False,
            "result_source": RESULT_SOURCE,
        }
        for field in DIAGNOSTIC_FIELDS:
            row[field] = common.parse_float(raw.get(field))
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
        "final_slow_q_buffer_size": int(
            sum(solver.q_value_trainer.fold_sizes())
        ),
        "final_fast_q_buffer_size": int(
            sum(solver.q_value_trainer.fast_fold_sizes())
        ),
        "result_source": RESULT_SOURCE,
    }
    for field in DIAGNOSTIC_FIELDS:
        summary[f"final_{field}"] = float(final[field])

    estimator_rows = [
        row
        for row in curves
        if float(row.get("unbiased_estimator_sample_count", 0.0)) > 0.0
    ]
    if not estimator_rows:
        raise RuntimeError("The run produced no unbiased estimator samples")
    for row in estimator_rows:
        if not np.isclose(row["control_variate_beta_min"], 1.0):
            raise RuntimeError("The candidate did not keep beta fixed at one")
        if not np.isclose(row["control_variate_beta_max"], 1.0):
            raise RuntimeError("The candidate did not keep beta fixed at one")
        if not np.isclose(row["control_replay_rng_isolated"], 1.0):
            raise RuntimeError("Control replay used the process-wide Python RNG")
    if float(final["policy_weighted_advantage_abs_mean"]) > 1e-10:
        raise RuntimeError("Control-variate advantages were not policy-centred")
    if min(solver.q_value_trainer.fast_fold_sizes()) <= 0:
        raise RuntimeError("A fast critic fold received no data")
    if min(solver.q_value_trainer.fold_sizes()) <= 0:
        raise RuntimeError("A slow critic fold received no data")
    if not solver.q_value_trainer.isolated_replay_rng:
        raise RuntimeError("Control replay RNG isolation was not enabled")

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
        "experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes.run",
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


def _parse_seeds(value: str | None):
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
        "fast_q_train_steps": args.fast_q_train_steps,
        "calibration_train_steps": args.calibration_train_steps,
        "rho_train_steps": args.rho_train_steps,
        "advantage_batch_size": args.batch_size,
        "ave_policy_batch_size": args.batch_size,
        "baseline_batch_size": args.batch_size,
        "calibration_batch_size": args.batch_size,
        "rho_batch_size": args.batch_size,
        "advantage_buffer_size": args.buffer_size,
        "ave_policy_buffer_size": args.buffer_size,
        "baseline_buffer_size": args.buffer_size,
        "calibration_buffer_size": args.buffer_size,
        "fast_q_buffer_size": args.fast_q_buffer_size,
        "rho_buffer_size": args.rho_buffer_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (
            int(args.early_evaluation_nodes),
        )


def _load_references(seeds: Sequence[int]):
    labels = ALGORITHM_LABELS
    curves_9 = common.load_reference_curves(
        EXPERIMENT_9_CURVES,
        expected_sha256=EXPERIMENT_9_CURVES_SHA256,
        expected_rows=EXPERIMENT_9_CURVE_ROWS,
        expected_algorithm_ids=(
            EXPERIMENT_6_ALGORITHM_ID,
            EXPERIMENT_9_ALGORITHM_ID,
        ),
        expected_seeds=DEFAULT_SEEDS,
        result_source=REFERENCE_SOURCE_9,
        label_overrides=labels,
    )
    summaries_9 = common.load_reference_summaries(
        EXPERIMENT_9_SUMMARIES,
        expected_sha256=EXPERIMENT_9_SUMMARIES_SHA256,
        expected_rows=EXPERIMENT_9_SUMMARY_ROWS,
        expected_algorithm_ids=(
            EXPERIMENT_6_ALGORITHM_ID,
            EXPERIMENT_9_ALGORITHM_ID,
        ),
        expected_seeds=DEFAULT_SEEDS,
        result_source=REFERENCE_SOURCE_9,
        label_overrides=labels,
    )
    curves_13 = common.load_reference_curves(
        EXPERIMENT_13_CURVES,
        expected_sha256=EXPERIMENT_13_CURVES_SHA256,
        expected_rows=EXPERIMENT_13_CURVE_ROWS,
        expected_algorithm_ids=(EXPERIMENT_13_ALGORITHM_ID,),
        expected_seeds=DEFAULT_SEEDS,
        result_source=REFERENCE_SOURCE_13,
        label_overrides=labels,
    )
    summaries_13 = common.load_reference_summaries(
        EXPERIMENT_13_SUMMARIES,
        expected_sha256=EXPERIMENT_13_SUMMARIES_SHA256,
        expected_rows=EXPERIMENT_13_SUMMARY_ROWS,
        expected_algorithm_ids=(EXPERIMENT_13_ALGORITHM_ID,),
        expected_seeds=DEFAULT_SEEDS,
        result_source=REFERENCE_SOURCE_13,
        label_overrides=labels,
    )
    selected = set(int(seed) for seed in seeds)
    curves = [
        row
        for row in (*curves_9, *curves_13)
        if int(row["seed"]) in selected
    ]
    summaries = [
        row
        for row in (*summaries_9, *summaries_13)
        if int(row["seed"]) in selected
    ]
    return curves, summaries


def _plot_critic_diagnostics(run_dir: Path, rows):
    candidate = [
        row
        for row in rows
        if row["algorithm_id"] == ALGORITHM_ID
        and not bool(row.get("is_initial_policy_evaluation", False))
    ]
    if not candidate:
        return
    checkpoints = sorted({int(row["checkpoint_index"]) for row in candidate})
    nodes, rho, rho_se, fast, slow, mixture = [], [], [], [], [], []
    for checkpoint in checkpoints:
        current = [
            row
            for row in candidate
            if int(row["checkpoint_index"]) == checkpoint
        ]
        nodes.append(float(np.mean([row["nodes_touched"] for row in current])))
        rho_stats = shared._stats(row["fast_slow_rho_mean"] for row in current)
        rho.append(rho_stats["mean"])
        rho_se.append(rho_stats["se"])
        fast.append(
            shared._stats(
                row["fast_critic_sampled_mse"] for row in current
            )["mean"]
        )
        slow.append(
            shared._stats(
                row["slow_critic_sampled_mse"] for row in current
            )["mean"]
        )
        mixture.append(
            shared._stats(
                row["mixture_critic_sampled_mse"] for row in current
            )["mean"]
        )

    x = np.asarray(nodes)
    mean_rho = np.asarray(rho)
    se_rho = np.asarray(rho_se)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, mean_rho, marker="o", color=COLORS[ALGORITHM_ID])
    ax.fill_between(
        x,
        mean_rho - se_rho,
        mean_rho + se_rho,
        color=COLORS[ALGORITHM_ID],
        alpha=0.14,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Mean fast-critic weight rho")
    set_chart_title(ax, "Experiment 15 held-out fast/slow mixture")
    fig.tight_layout()
    fig.savefig(run_dir / "rho_by_nodes.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(x, fast, marker="o", label="Fast critic")
    ax.plot(x, slow, marker="o", label="Slow critic")
    ax.plot(x, mixture, marker="o", label="Controlled mixture")
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Sampled return prediction MSE")
    set_chart_title(ax, "Experiment 15 held-out critic error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "critic_sampled_mse_by_nodes.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/fixed_beta_fast_slow_escher_5x_nodes",
    )
    parser.add_argument("--seeds")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--target-nodes", type=int)
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--fast-q-train-steps", type=int)
    parser.add_argument("--calibration-train-steps", type=int)
    parser.add_argument("--rho-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--fast-q-buffer-size", type=int)
    parser.add_argument("--rho-buffer-size", type=int)
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
        raise ValueError("Experiment 15 supports paired seeds 0, 1 and 2")
    if args.target_nodes is not None and args.target_nodes <= 0:
        raise ValueError("target-nodes must be positive")
    config = deepcopy(CANDIDATE_CONFIG)
    _apply_overrides(args, config)
    reference_curves, reference_summaries = _load_references(seeds)
    targets = {
        seed: int(args.target_nodes or EXPERIMENT_2_NODE_TARGETS[seed])
        for seed in seeds
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root)
        / f"fixed_beta_fast_slow_escher_5x_nodes_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "seeds": seeds,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "training_config": config,
        "paired_node_targets": targets,
        "reference_sources": REFERENCE_SOURCES,
        "reference_hashes": {
            "experiment_9_curves": common.sha256(EXPERIMENT_9_CURVES),
            "experiment_9_summaries": common.sha256(EXPERIMENT_9_SUMMARIES),
            "experiment_13_curves": common.sha256(EXPERIMENT_13_CURVES),
            "experiment_13_summaries": common.sha256(
                EXPERIMENT_13_SUMMARIES
            ),
        },
        "expected_sequential_runtime_hours": EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
        "configured_batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "estimator": (
                "Always-unbiased residual correction with beta fixed exactly "
                "at one."
            ),
            "control_critic": (
                "Experiment 9's complete three-fold fast/slow critic and "
                "one-iteration-lagged held-out rho controller."
            ),
            "slow_replay": (
                "Uniform lifetime reservoir for each cross-fitted slow critic."
            ),
            "fast_replay": (
                "Current-outer-iteration circular replay for each fast critic."
            ),
            "rng_isolation": (
                "Reservoir replacement and all fast, slow and rho replay "
                "minibatch sampling use deterministic component-local Python "
                "RNG streams."
            ),
            "comparison": (
                "Checksum-validated Experiment 6, 9 and 13 outputs are reused "
                "without retraining."
            ),
        },
    }
    shared._write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    for seed in seeds:
        try:
            LOGGER.info(
                "Running Experiment 15 seed %s to %s nodes",
                seed,
                targets[seed],
            )
            result = _run_subprocess(run_dir, seed, config, targets[seed])
            results.append(result)
            shared._write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {
                    "seed": seed,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            shared._write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("Experiment 15 seed %s failed: %s", seed, exc)
            if not args.continue_on_error:
                return 2

    candidate_summaries = [result["summary"] for result in results]
    candidate_curves = [row for result in results for row in result["curves"]]
    combined_summaries = [*reference_summaries, *candidate_summaries]
    combined_curves = [*reference_curves, *candidate_curves]
    paired = common.paired_differences(
        combined_summaries,
        candidate_algorithm_id=ALGORITHM_ID,
        reference_algorithm_ids=ALGORITHM_IDS[:-1],
        algorithm_labels=ALGORITHM_LABELS,
        seeds=seeds,
    )
    aggregate = common.aggregate(combined_summaries, ALGORITHM_IDS)

    shared._write_csv(run_dir / "candidate_seed_summary.csv", candidate_summaries)
    shared._write_csv(run_dir / "candidate_checkpoint_curves.csv", candidate_curves)
    shared._write_csv(run_dir / "combined_seed_summary.csv", combined_summaries)
    shared._write_csv(run_dir / "combined_checkpoint_curves.csv", combined_curves)
    shared._write_csv(run_dir / "paired_differences.csv", paired)
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
        title = "Experiment 15 fixed-beta full fast/slow ESCHER"
        common.plot_exploitability(
            run_dir,
            combined_curves,
            x_key="nodes_touched",
            algorithm_ids=ALGORITHM_IDS,
            algorithm_labels=ALGORITHM_LABELS,
            colors=COLORS,
            title=title,
        )
        common.plot_exploitability(
            run_dir,
            combined_curves,
            x_key="wall_clock_seconds",
            algorithm_ids=ALGORITHM_IDS,
            algorithm_labels=ALGORITHM_LABELS,
            colors=COLORS,
            title=title,
        )
        common.plot_final(
            run_dir,
            combined_summaries,
            algorithm_ids=ALGORITHM_IDS,
            algorithm_labels=ALGORITHM_LABELS,
            colors=COLORS,
            title="Experiment 15 with Experiments 6, 9 and 13",
        )
        _plot_critic_diagnostics(run_dir, candidate_curves)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
