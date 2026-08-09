"""CLI for Experiment 17: six final policies across five training seeds."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/escher_architecture_exp17_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from .analyse import run_analysis  # noqa: E402
from .config import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_ORDER,
    DEFAULT_SEEDS,
    EQUIVALENCE_EPSILON,
    EXISTING_SNAPSHOT_ALGORITHMS,
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    TARGET_NODES,
    UPSTREAM,
    VR_ALGORITHMS,
    VR_CONFIG,
    validate_contract,
)
from .io_utils import write_csv, write_json  # noqa: E402
from .policies import (  # noqa: E402
    local_audited_snapshot_directories,
    read_snapshot_metadata,
    select_final_snapshots,
    snapshot_directories,
)
from .train_vr import run_worker, train_all_vr  # noqa: E402


LOGGER = logging.getLogger(__name__)


def _parse_seeds(value: Optional[str]) -> list[int]:
    if value is None:
        return [int(seed) for seed in DEFAULT_SEEDS]
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def _apply_vr_overrides(args, config: dict) -> None:
    overrides = {
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
    for key, value in overrides.items():
        if value is not None:
            config[key] = int(value)
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (
            int(args.early_evaluation_nodes),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="all",
        choices=("all", "prepare", "train", "analyse"),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/six_algorithm_final_policy_head_to_head",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        help=(
            "Directory containing deep_cfr/, dream/, escher/, and "
            "unbiased_control_variate_escher/ snapshot subdirectories"
        ),
    )
    parser.add_argument("--seeds")
    parser.add_argument("--target-nodes", type=int, default=TARGET_NODES)
    parser.add_argument("--equivalence-epsilon", type=float, default=EQUIVALENCE_EPSILON)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=log_format)
    handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def _select_sources(args) -> Mapping[str, Path]:
    if args.snapshot_root is not None:
        return snapshot_directories(args.snapshot_root.resolve())
    return local_audited_snapshot_directories()


def _inventory_from_run(run_dir: Path, seeds: Sequence[int]) -> list[dict]:
    inventory = []
    required = {int(seed) for seed in seeds}
    for algorithm_id in ALGORITHM_ORDER:
        directory = run_dir / "snapshots" / algorithm_id
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing archived snapshots: {directory}")
        by_seed = {seed: [] for seed in required}
        for suffix in ("*.pt", "*.pkl"):
            for path in directory.glob(suffix):
                metadata = read_snapshot_metadata(algorithm_id, path)
                if metadata["seed"] in by_seed:
                    by_seed[metadata["seed"]].append(metadata)
        for seed in sorted(required):
            candidates = by_seed[seed]
            if not candidates:
                raise FileNotFoundError(
                    f"Missing {algorithm_id} snapshot for seed {seed} in {directory}"
                )
            inventory.append(
                max(
                    candidates,
                    key=lambda row: (
                        -1 if row["nodes_touched"] is None else row["nodes_touched"],
                        row["checkpoint"],
                    ),
                )
            )
    return inventory


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return run_worker(args.worker_input_json, args.worker_output_json)

    seeds = _parse_seeds(args.seeds)
    config = deepcopy(VR_CONFIG)
    _apply_vr_overrides(args, config)
    validate_contract(
        seeds,
        args.target_nodes,
        config,
        require_five_seeds=not args.smoke,
    )
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (Path(args.output_root) / f"{EXPERIMENT_NAME}_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir, args.verbose)

    source_directories = _select_sources(args)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "game": "leduc_poker",
        "phase": args.phase,
        "smoke": bool(args.smoke),
        "seeds": seeds,
        "target_nodes": int(args.target_nodes),
        "algorithms": ALGORITHMS,
        "algorithm_order": list(ALGORITHM_ORDER),
        "existing_snapshot_algorithms": list(EXISTING_SNAPSHOT_ALGORITHMS),
        "trained_algorithms": list(VR_ALGORITHMS),
        "source_snapshot_directories": {
            key: str(value) for key, value in source_directories.items()
        },
        "vr_config": config,
        "vr_upstream": UPSTREAM,
        "evaluation_protocol": "exact OpenSpiel expected value in both seats",
        "sampled_games": 0,
        "primary_statistical_unit": "paired independent training seed",
        "multiplicity_family": "15 unordered algorithm pairs",
    }
    write_json(run_dir / "experiment_metadata.json", metadata)
    LOGGER.info("Run directory: %s", run_dir)

    if args.phase in {"all", "prepare"}:
        existing_inventory = select_final_snapshots(
            source_directories,
            seeds,
            run_dir / "snapshots",
        )
        write_csv(run_dir / "existing_snapshot_inventory.csv", existing_inventory)
        LOGGER.info("Archived %s existing final policies", len(existing_inventory))
        if args.phase == "prepare":
            return 0

    if args.phase in {"all", "train"}:
        outcome = train_all_vr(
            seeds=seeds,
            config=config,
            target_nodes=int(args.target_nodes),
            run_dir=run_dir,
            continue_on_error=bool(args.continue_on_error),
        )
        metadata["vr_training_failures"] = outcome["failures"]
        write_json(run_dir / "experiment_metadata.json", metadata)
        if outcome["failures"]:
            return 1
        if args.phase == "train":
            return 0

    if args.phase in {"all", "analyse"}:
        inventory = _inventory_from_run(run_dir, seeds)
        write_csv(run_dir / "snapshot_inventory.csv", inventory)
        outputs = run_analysis(
            snapshot_inventory=inventory,
            seeds=seeds,
            run_dir=run_dir,
            equivalence_epsilon=float(args.equivalence_epsilon),
        )
        metadata["analysis_outputs"] = outputs
        write_json(run_dir / "experiment_metadata.json", metadata)

    LOGGER.info("Experiment 17 complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
