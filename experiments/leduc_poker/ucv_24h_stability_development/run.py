"""CLI for Experiment 23 workers, exact aggregation, and smoke testing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .common import read_json, sha256
from .config import (
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    VARIANT_ORDER,
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


def _worker_dir(root: Path, index: int, variant_id: str, seed: int) -> Path:
    return Path(root) / "workers" / f"task_{index:03d}_{variant_id}_seed_{seed}"


def _validated_existing(
    worker_dir: Path,
    *,
    variant_id: str,
    seed: int,
    smoke: bool,
    schedule: Sequence[Mapping],
) -> dict | None:
    path = worker_dir / "worker_result.json"
    if not path.is_file():
        return None
    result = read_json(path)
    if (
        result.get("status") != "complete"
        or result.get("variant_id") != variant_id
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
    for relative in result.get("artifacts", {}).values():
        if not (worker_dir / relative).is_file():
            return None
    return result


def _run_task(
    *,
    task_index: int,
    seeds: Sequence[int],
    output_root: Path,
    smoke: bool,
    resume: bool,
) -> dict:
    tasks = task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} outside [0, {len(tasks)-1}]")
    variant_id, seed = tasks[task_index]
    schedule = checkpoint_schedule(smoke=smoke)
    worker_dir = _worker_dir(output_root, task_index, variant_id, seed)
    if resume:
        existing = _validated_existing(
            worker_dir,
            variant_id=variant_id,
            seed=seed,
            smoke=smoke,
            schedule=schedule,
        )
        if existing is not None:
            LOGGER.info("Reusing validated %s seed %s", variant_id, seed)
            return existing
    return run_worker(
        variant_id=variant_id,
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        smoke=smoke,
    )


def _cmd_worker(args) -> None:
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
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

    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    result = aggregate_workers(
        workers_root=Path(args.output_root) / "workers",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args) -> None:
    from .analyse import aggregate_workers

    schedule = checkpoint_schedule(smoke=True)
    validate_contract(seeds=SMOKE_SEEDS, schedule=schedule, smoke=True)
    for index, (variant_id, seed) in enumerate(task_schedule(SMOKE_SEEDS)):
        LOGGER.info(
            "Smoke worker %s/%s: %s seed %s",
            index + 1,
            len(VARIANT_ORDER),
            variant_id,
            seed,
        )
        _run_task(
            task_index=index,
            seeds=SMOKE_SEEDS,
            output_root=args.output_root,
            smoke=True,
            resume=args.resume,
        )
    result = aggregate_workers(
        workers_root=Path(args.output_root) / "workers",
        seeds=SMOKE_SEEDS,
        output_dir=Path(args.output_root) / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args) -> None:
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    print(
        json.dumps(
            {
                "tasks": [
                    {"task_index": index, "variant_id": variant, "seed": seed}
                    for index, (variant, seed) in enumerate(task_schedule(args.seeds))
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
