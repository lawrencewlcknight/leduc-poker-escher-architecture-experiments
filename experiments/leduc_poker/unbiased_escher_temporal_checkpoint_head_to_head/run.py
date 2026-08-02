"""CLI for Experiment 16 temporal checkpoint head-to-head analysis."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_architecture_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from escher_poker.json_utils import json_safe  # noqa: E402

from .analyse import run_analysis  # noqa: E402
from .config import (  # noqa: E402
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    CANDIDATE_CONFIG,
    CHECKPOINT_NODE_THRESHOLDS,
    CHECKPOINT_SCHEDULE,
    DEFAULT_SEEDS,
    EQUIVALENCE_EPSILON,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    validate_config,
)
from .train import _run_worker, run_training  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _parse_ints(value: Optional[str], default: Sequence[int]) -> List[int]:
    if value is None:
        return [int(item) for item in default]
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("At least one integer is required")
    return parsed


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)


def _config_sha256(config: Mapping[str, object]) -> str:
    encoded = json.dumps(
        json_safe(dict(config)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _apply_overrides(args, config: dict) -> None:
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


def _checkpoint_thresholds(args) -> tuple[int, ...]:
    if args.checkpoint_node_thresholds is not None:
        return tuple(_parse_ints(args.checkpoint_node_thresholds, ()))
    if args.target_nodes is None:
        return tuple(CHECKPOINT_NODE_THRESHOLDS)
    target = int(args.target_nodes)
    if target < len(CHECKPOINT_SCHEDULE):
        raise ValueError("target-nodes must be at least the number of checkpoints")
    thresholds = [
        max(1, round(target * stage / len(CHECKPOINT_SCHEDULE)))
        for stage in CHECKPOINT_SCHEDULE
    ]
    thresholds[-1] = target
    return tuple(thresholds)


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format)
    handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=("all", "train", "analyse"),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/unbiased_escher_temporal_checkpoint_head_to_head",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds")
    parser.add_argument("--target-nodes", type=int)
    parser.add_argument("--checkpoint-node-thresholds")
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
        "--equivalence-epsilon",
        type=float,
        default=None,
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_json)

    stored_metadata = None
    if args.phase == "analyse" and args.run_dir:
        metadata_path = args.run_dir / "experiment_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as handle:
                stored_metadata = json.load(handle)

    config = deepcopy(
        stored_metadata["training_config"]
        if stored_metadata and "training_config" in stored_metadata
        else CANDIDATE_CONFIG
    )
    if not stored_metadata:
        _apply_overrides(args, config)
    seeds = (
        [int(seed) for seed in stored_metadata["seeds"]]
        if stored_metadata and args.seeds is None
        else _parse_ints(args.seeds, DEFAULT_SEEDS)
    )
    thresholds = (
        tuple(int(value) for value in stored_metadata["checkpoint_node_thresholds"])
        if stored_metadata and args.checkpoint_node_thresholds is None
        and args.target_nodes is None
        else _checkpoint_thresholds(args)
    )
    epsilon = float(
        args.equivalence_epsilon
        if args.equivalence_epsilon is not None
        else (
            stored_metadata.get("equivalence_epsilon", EQUIVALENCE_EPSILON)
            if stored_metadata
            else EQUIVALENCE_EPSILON
        )
    )
    validate_config(config, CHECKPOINT_SCHEDULE, thresholds)

    if args.run_dir:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (
            Path(args.output_root)
            / f"{EXPERIMENT_NAME}_{timestamp}"
        ).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir, args.verbose)

    metadata = dict(stored_metadata or {})
    metadata.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "algorithm_id": ALGORITHM_ID,
            "algorithm_label": ALGORITHM_LABEL,
            "baseline_experiment": 7,
            "training_config": config,
            "experiment_7_candidate_config_sha256": _config_sha256(
                CANDIDATE_CONFIG
            ),
            "active_training_config_sha256": _config_sha256(config),
            "seeds": seeds,
            "checkpoint_schedule": list(CHECKPOINT_SCHEDULE),
            "checkpoint_node_thresholds": list(thresholds),
            "equivalence_epsilon": epsilon,
            "phase": args.phase,
            "training_protocol": (
                "one uninterrupted Experiment 7 candidate trajectory per seed"
            ),
            "snapshot_protocol": (
                "first complete outer iteration crossing each node threshold"
            ),
            "head_to_head_evaluation": (
                "exact OpenSpiel expected value averaged over both seats"
            ),
            "statistical_unit": "independent training seed",
        }
    )
    _write_json(run_dir / "experiment_metadata.json", metadata)
    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Seeds: %s", seeds)
    LOGGER.info("Node thresholds: %s", thresholds)

    if args.phase in {"all", "train"}:
        outcome = run_training(
            config=config,
            seeds=seeds,
            checkpoint_node_thresholds=thresholds,
            run_dir=run_dir,
            continue_on_error=bool(args.continue_on_error),
        )
        metadata["completed_seeds"] = outcome["completed_seeds"]
        metadata["failed_seeds"] = outcome["failed"]
        _write_json(run_dir / "experiment_metadata.json", metadata)
        if set(outcome["completed_seeds"]) != set(seeds):
            LOGGER.error(
                "Completed seeds %s do not match requested seeds %s",
                outcome["completed_seeds"],
                seeds,
            )
            return 1

    if args.phase in {"all", "analyse"}:
        snapshots_dir = run_dir / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.glob("*.pkl")):
            LOGGER.error("No snapshots found in %s", snapshots_dir)
            return 2
        outputs = run_analysis(
            config=config,
            checkpoint_schedule=CHECKPOINT_SCHEDULE,
            checkpoint_node_thresholds=thresholds,
            equivalence_epsilon=epsilon,
            run_dir=run_dir,
            snapshots_dir=snapshots_dir,
        )
        metadata["analysis_outputs"] = {
            key: str(value) for key, value in outputs.items()
        }
        _write_json(run_dir / "experiment_metadata.json", metadata)

    LOGGER.info("All outputs saved to %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
