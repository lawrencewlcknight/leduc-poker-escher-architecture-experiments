"""Run Experiment 13 and compare it with immutable Experiment 6 results."""

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
from typing import Any, Dict, List, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path("outputs") / ".matplotlib_cache").resolve()),
)
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from experiments.leduc_poker.adaptive_residual_predictive_escher import (  # noqa: E402
    run as shared,
)
from experiments.leduc_poker import fixed_beta_reservoir_shared as common  # noqa: E402

from .config import (  # noqa: E402
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    BATCH_TIMEOUT_SECONDS,
    CANDIDATE_CONFIG,
    DEFAULT_SEEDS,
    EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_6_SOURCE,
    EXPERIMENT_ID,
    REFERENCE_ALGORITHM_ID,
    REFERENCE_ALGORITHM_LABEL,
    REFERENCE_CURVE_ROWS,
    REFERENCE_CURVES,
    REFERENCE_CURVES_SHA256,
    REFERENCE_SUMMARIES,
    REFERENCE_SUMMARIES_SHA256,
    REFERENCE_SUMMARY_ROWS,
)


LOGGER = logging.getLogger("fixed_beta_reservoir_escher_5x_nodes")
RESULT_SOURCE = "experiment_13_new_run"
REFERENCE_SOURCE = "saved_experiment_6"
ALGORITHM_IDS = (REFERENCE_ALGORITHM_ID, ALGORITHM_ID)
ALGORITHM_LABELS = {
    REFERENCE_ALGORITHM_ID: REFERENCE_ALGORITHM_LABEL,
    ALGORITHM_ID: ALGORITHM_LABEL,
}
COLORS = {
    REFERENCE_ALGORITHM_ID: "#9467bd",
    ALGORITHM_ID: "#d62728",
}


def _run_candidate(seed: int, config: Dict[str, Any], target_nodes: int):
    return common.run_candidate(
        seed=seed,
        config=config,
        target_nodes=target_nodes,
        algorithm_id=ALGORITHM_ID,
        algorithm_label=ALGORITHM_LABEL,
        result_source=RESULT_SOURCE,
    )


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
        "experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.run",
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
            int(args.early_evaluation_nodes),
        )


def _load_aggregate_results(run_dirs: Sequence[Path]):
    indexed = {}
    for run_dir in run_dirs:
        paths = sorted(run_dir.rglob("worker_results/*.json"))
        if not paths:
            raise ValueError(f"No worker results found under {run_dir}")
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                result = json.load(handle)
            key = int(result["summary"]["seed"])
            if key in indexed:
                raise ValueError(f"Duplicate aggregate result for seed {key}")
            indexed[key] = result
    return [indexed[key] for key in sorted(indexed)]


def _finalize(
    run_dir: Path,
    *,
    results,
    failures,
    metadata,
    reference_curves,
    reference_summaries,
    seeds,
):
    candidate_summaries = [result["summary"] for result in results]
    candidate_curves = [row for result in results for row in result["curves"]]
    combined_summaries = [*reference_summaries, *candidate_summaries]
    combined_curves = [*reference_curves, *candidate_curves]
    paired = common.paired_differences(
        combined_summaries,
        candidate_algorithm_id=ALGORITHM_ID,
        reference_algorithm_ids=(REFERENCE_ALGORITHM_ID,),
        algorithm_labels=ALGORITHM_LABELS,
        seeds=seeds,
    )
    aggregate = common.aggregate(combined_summaries, ALGORITHM_IDS)

    shared._write_json(run_dir / "experiment_metadata.json", metadata)
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
        title = "Experiment 13 fixed-beta reservoir ESCHER vs Experiment 6"
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
            title="Experiment 13 and Experiment 6: final exploitability",
        )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/fixed_beta_reservoir_escher_5x_nodes",
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
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument(
        "--aggregate-run-dir",
        action="append",
        type=Path,
        default=[],
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

    seeds = _parse_seeds(args.seeds)
    if any(seed not in EXPERIMENT_2_NODE_TARGETS for seed in seeds):
        raise ValueError("Experiment 13 supports paired seeds 0, 1 and 2")
    if args.target_nodes is not None and args.target_nodes <= 0:
        raise ValueError("target-nodes must be positive")
    config = deepcopy(CANDIDATE_CONFIG)
    _apply_overrides(args, config)
    reference_curves = common.load_reference_curves(
        args.reference_curves,
        expected_sha256=REFERENCE_CURVES_SHA256,
        expected_rows=REFERENCE_CURVE_ROWS,
        expected_algorithm_ids=(REFERENCE_ALGORITHM_ID,),
        expected_seeds=DEFAULT_SEEDS,
        result_source=REFERENCE_SOURCE,
        label_overrides=ALGORITHM_LABELS,
    )
    reference_summaries = common.load_reference_summaries(
        args.reference_summaries,
        expected_sha256=REFERENCE_SUMMARIES_SHA256,
        expected_rows=REFERENCE_SUMMARY_ROWS,
        expected_algorithm_ids=(REFERENCE_ALGORITHM_ID,),
        expected_seeds=DEFAULT_SEEDS,
        result_source=REFERENCE_SOURCE,
        label_overrides=ALGORITHM_LABELS,
    )
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
        / f"fixed_beta_reservoir_escher_5x_nodes_{timestamp}"
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
        "reference_curves_sha256": common.sha256(args.reference_curves),
        "reference_summaries_file": str(args.reference_summaries),
        "reference_summaries_sha256": common.sha256(args.reference_summaries),
        "expected_sequential_runtime_hours": EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
        "configured_batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "estimator": "Always-unbiased residual correction with beta fixed exactly at one.",
            "critic": (
                "Three persistent frozen-target cross-fitted critics, each "
                "trained from a uniform lifetime transition reservoir."
            ),
            "retained_components": (
                "Experiment 6 residual calibration, adaptive full-support "
                "sampling, gated predictor, regret learner and weighted "
                "average-policy learner."
            ),
            "removed_components": "No fast critic and no rho controller from Experiment 9.",
            "comparison": (
                "Checksum-validated Experiment 6 curves and summaries are "
                "reused without retraining."
            ),
        },
    }

    failures = []
    if args.aggregate_run_dir:
        results = _load_aggregate_results(args.aggregate_run_dir)
        result_seeds = sorted(int(result["summary"]["seed"]) for result in results)
        result_targets = {
            int(result["summary"]["seed"]): int(
                result["summary"]["target_nodes_touched"]
            )
            for result in results
        }
        expected_targets = {
            seed: int(args.target_nodes or EXPERIMENT_2_NODE_TARGETS[seed])
            for seed in result_seeds
        }
        if result_targets != expected_targets:
            raise ValueError(
                "Aggregate inputs do not use the expected paired node budgets"
            )
        metadata["aggregate_source_run_dirs"] = [
            str(path) for path in args.aggregate_run_dir
        ]
        metadata["seeds"] = result_seeds
        metadata["paired_node_targets"] = result_targets
        _finalize(
            run_dir,
            results=results,
            failures=failures,
            metadata=metadata,
            reference_curves=[
                row for row in reference_curves if int(row["seed"]) in result_seeds
            ],
            reference_summaries=[
                row for row in reference_summaries if int(row["seed"]) in result_seeds
            ],
            seeds=result_seeds,
        )
        return 0

    results = []
    for seed in seeds:
        try:
            LOGGER.info("Running Experiment 13 seed %s to %s nodes", seed, targets[seed])
            result = _run_subprocess(run_dir, seed, config, targets[seed])
            results.append(result)
            shared._write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {"seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
            )
            shared._write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("Experiment 13 seed %s failed: %s", seed, exc)
            if not args.continue_on_error:
                return 2

    _finalize(
        run_dir,
        results=results,
        failures=failures,
        metadata=metadata,
        reference_curves=reference_curves,
        reference_summaries=reference_summaries,
        seeds=seeds,
    )
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
