"""CLI for Experiments 36--39 post-training studies."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil

from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import read_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    checkpoint_schedule as experiment_29_schedule,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.worker import (
    run_worker as run_experiment_29_worker,
)

from .analyse import aggregate_workers
from .config import METHODS, PRODUCTION_SEEDS, SMOKE_SEEDS, task_schedule
from .worker import repository_commit, run_worker


LOGGER = logging.getLogger(__name__)


def _parse_seeds(value):
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return result


def worker_dir(root, method_id, index, seed):
    return Path(root) / "workers" / f"task_{index:03d}_{method_id}_seed_{seed}"


def run_task(*, method_id, task_index, seeds, output_root, source_root,
             smoke, resume):
    tasks = task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} is outside the schedule")
    seed = tasks[task_index]
    destination = worker_dir(output_root, method_id, task_index, seed)
    existing_path = destination / "worker_result.json"
    if resume and existing_path.is_file():
        existing = read_json(existing_path)
        if (
            existing.get("status") == "complete"
            and existing.get("method_id") == method_id
            and existing.get("repository_commit") == repository_commit()
        ):
            return existing
    return run_worker(
        method_id=method_id, seed=seed, source_root=source_root,
        worker_dir=destination, smoke=smoke,
    )


def cmd_worker(args):
    seeds = SMOKE_SEEDS if args.smoke and args.seeds == PRODUCTION_SEEDS else args.seeds
    print(json.dumps(run_task(
        method_id=args.method, task_index=args.task_index, seeds=seeds,
        output_root=args.output_root, source_root=args.source_root,
        smoke=args.smoke, resume=args.resume,
    ), indent=2))


def cmd_aggregate(args):
    seeds = SMOKE_SEEDS if args.smoke and args.seeds == PRODUCTION_SEEDS else args.seeds
    result = aggregate_workers(
        method_id=args.method,
        workers_root=Path(args.output_root) / "workers",
        seeds=seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def cmd_smoke(args):
    output_root = Path(args.output_root).resolve()
    source_root = output_root / "experiment_29_source"
    source_worker = source_root / "workers" / (
        "task_000_promoted_ucv_cross_entropy_seed_0"
    )
    if source_worker.exists() and not args.resume:
        shutil.rmtree(source_worker)
    run_experiment_29_worker(
        seed=0, schedule=experiment_29_schedule(smoke=True),
        worker_dir=source_worker, smoke=True, resume=args.resume,
    )
    run_task(
        method_id=args.method, task_index=0, seeds=SMOKE_SEEDS,
        output_root=output_root, source_root=source_root,
        smoke=True, resume=args.resume,
    )
    result = aggregate_workers(
        method_id=args.method, workers_root=output_root / "workers",
        seeds=SMOKE_SEEDS, output_dir=output_root / "analysis", smoke=True,
    )
    print(json.dumps(result, indent=2))


def cmd_schedule(args):
    seeds = SMOKE_SEEDS if args.smoke else args.seeds
    print(json.dumps({"method_id": args.method, "tasks": [
        {"task_index": index, "source_seed": seed}
        for index, seed in enumerate(task_schedule(seeds))
    ]}, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", choices=tuple(METHODS), required=True)
    sub = result.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--source-root", type=Path, required=True)
    worker.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--no-resume", dest="resume", action="store_false")
    worker.set_defaults(func=cmd_worker, resume=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(func=cmd_aggregate)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=cmd_smoke, resume=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    schedule.add_argument("--smoke", action="store_true")
    schedule.set_defaults(func=cmd_schedule)
    return result


def main():
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()

