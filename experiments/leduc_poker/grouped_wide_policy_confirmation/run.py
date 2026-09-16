"""CLI for Experiment 35 training, aggregation, and smoke validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import (
    read_json,
    sha256,
    write_csv,
    write_json,
)

from .config import (
    CANDIDATE_ID,
    HISTORICAL_ALGORITHM_LABELS,
    HISTORICAL_ALGORITHM_ORDER,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    validate_contract,
)
from .worker import run_worker


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _worker_dir(root: Path, index: int, seed: int) -> Path:
    return Path(root) / "workers" / f"task_{index:03d}_{CANDIDATE_ID}_seed_{seed}"


def _validated_existing(
    worker_dir: Path, *, seed: int, smoke: bool, schedule: Sequence[Mapping]
) -> dict | None:
    path = worker_dir / "worker_result.json"
    if not path.is_file():
        return None
    result = read_json(path)
    if (
        result.get("status") != "complete"
        or result.get("candidate_id") != CANDIDATE_ID
        or int(result.get("seed", -1)) != int(seed)
        or bool(result.get("smoke")) != bool(smoke)
        or result.get("repository_commit") != _repository_commit()
        or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
    ):
        return None
    for record in result.get("snapshots", ()):
        snapshot = worker_dir / record["relative_path"]
        if not snapshot.is_file() or sha256(snapshot) != record["sha256"]:
            return None
    return result


def _run_task(
    *, task_index: int, seeds: Sequence[int], output_root: Path,
    smoke: bool, resume: bool,
) -> dict:
    tasks = task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} outside [0, {len(tasks)-1}]")
    _, seed = tasks[task_index]
    schedule = checkpoint_schedule(smoke=smoke)
    worker_dir = _worker_dir(output_root, task_index, seed)
    if resume:
        existing = _validated_existing(
            worker_dir, seed=seed, smoke=smoke, schedule=schedule
        )
        if existing is not None:
            LOGGER.info("Reusing validated Experiment 35 seed %s", seed)
            return existing
    return run_worker(
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        smoke=smoke,
        resume=resume,
    )


def _normalise_smoke_seeds(args) -> None:
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS


def _cmd_worker(args) -> None:
    _normalise_smoke_seeds(args)
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    result = _run_task(
        task_index=args.task_index,
        seeds=args.seeds,
        output_root=args.output_root,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, default=str))


def _cmd_aggregate(args) -> None:
    from .analyse import aggregate_workers

    _normalise_smoke_seeds(args)
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    result = aggregate_workers(
        workers_root=Path(args.output_root) / "workers",
        experiment_29_root=args.experiment_29_root,
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _write_smoke_reference(output_root: Path) -> Path:
    """Create a schema-complete synthetic join fixture for local/cloud smoke."""
    worker_result_path = next((output_root / "workers").rglob("worker_result.json"))
    result = read_json(worker_result_path)
    curves_path = worker_result_path.parent / result["artifacts"]["checkpoint_curves"]
    import csv

    with open(curves_path, newline="", encoding="utf-8") as handle:
        curves = list(csv.DictReader(handle))
    rows = []
    offsets = {
        "deep_cfr": 0.04,
        "unbiased_control_variate_escher": 0.03,
        "selected_nonpredictive_ucv": 0.02,
        "promoted_ucv_cross_entropy": 0.01,
    }
    for record in result["snapshots"]:
        if record["checkpoint_type"] != "active_time":
            continue
        curve = curves[int(record["checkpoint_row_index"])]
        baseline = float(curve["exploitability"])
        for algorithm_id in HISTORICAL_ALGORITHM_ORDER:
            rows.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": HISTORICAL_ALGORITHM_LABELS[algorithm_id],
                    "seed": 0,
                    "checkpoint_id": record["checkpoint_id"],
                    "checkpoint_type": "active_time",
                    "checkpoint_target_active_hours": record["checkpoint_target_active_hours"],
                    "actual_active_hours": float(record["active_seconds"]) / 3600.0,
                    "nodes_touched": int(record["nodes_touched"]),
                    "completed_iteration": int(record["completed_iteration"]),
                    "exploitability": baseline + offsets[algorithm_id],
                }
            )
    reference = output_root / "experiment_29_reference" / "analysis"
    write_csv(reference / "combined_checkpoint_policy_metrics.csv", rows)
    write_json(
        reference / "aggregate_manifest.json",
        {
            "status": "complete",
            "smoke": True,
            "contract": {"experiment_id": 29, "production_seeds": [0]},
            "fixture_role": "Experiment 35 join-path smoke only",
        },
    )
    return reference.parent


def _cmd_smoke(args) -> None:
    from .analyse import aggregate_workers

    output_root = Path(args.output_root).resolve()
    schedule = checkpoint_schedule(smoke=True)
    validate_contract(seeds=SMOKE_SEEDS, schedule=schedule, smoke=True)
    _run_task(
        task_index=0,
        seeds=SMOKE_SEEDS,
        output_root=output_root,
        smoke=True,
        resume=args.resume,
    )
    experiment_29_root = _write_smoke_reference(output_root)
    result = aggregate_workers(
        workers_root=output_root / "workers",
        experiment_29_root=experiment_29_root,
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args) -> None:
    _normalise_smoke_seeds(args)
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    print(
        json.dumps(
            {
                "tasks": [
                    {"task_index": index, "candidate_id": candidate, "seed": seed}
                    for index, (candidate, seed) in enumerate(task_schedule(args.seeds))
                ],
                "checkpoints": list(schedule),
            },
            indent=2,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--no-resume", dest="resume", action="store_false")
    worker.set_defaults(func=_cmd_worker, resume=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--experiment-29-root", type=Path, required=True)
    aggregate.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(func=_cmd_aggregate)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)

    schedule = subparsers.add_parser("schedule")
    schedule.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    schedule.add_argument("--smoke", action="store_true")
    schedule.set_defaults(func=_cmd_schedule)
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
