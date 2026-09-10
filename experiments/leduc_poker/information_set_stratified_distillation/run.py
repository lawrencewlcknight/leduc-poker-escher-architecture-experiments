"""CLI for Experiment 28 workers, aggregation and smoke validation."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    read_json,
    sha256,
    write_csv,
)
from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    checkpoint_schedule as experiment_25_schedule,
)
from experiments.leduc_poker.ucv_residual_target_factorial.worker import (
    run_worker as run_experiment_25_worker,
)

from .analyse import aggregate_workers
from .config import PRODUCTION_SEEDS, SMOKE_SEEDS, task_schedule, validate_contract
from .worker import run_worker


LOGGER = logging.getLogger(__name__)


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _worker_dir(root: Path, index: int, seed: int) -> Path:
    return Path(root) / "workers" / f"task_{index:03d}_stratified_policy_seed_{seed}"


def _run_task(
    *,
    task_index: int,
    seeds,
    output_root: Path,
    source_root: Path,
    smoke: bool,
    resume: bool,
):
    tasks = task_schedule(seeds)
    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(f"Task index {task_index} outside [0, {len(tasks) - 1}]")
    seed = tasks[task_index]
    worker_dir = _worker_dir(output_root, task_index, seed)
    existing_path = worker_dir / "worker_result.json"
    if resume and existing_path.is_file():
        existing = read_json(existing_path)
        if (
            existing.get("status") == "complete"
            and int(existing.get("source_seed", -1)) == seed
            and existing.get("repository_commit") == _repository_commit()
        ):
            valid = all(
                (worker_dir / row["relative_path"]).is_file()
                and sha256(worker_dir / row["relative_path"]) == row["sha256"]
                for row in existing.get("policies", ())
            )
            if valid:
                LOGGER.info("Reusing completed Experiment 28 seed %s", seed)
                return existing
    return run_worker(
        seed=seed,
        source_root=source_root,
        worker_dir=worker_dir,
        smoke=smoke,
    )


def _cmd_worker(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    result = _run_task(
        task_index=args.task_index,
        seeds=args.seeds,
        output_root=args.output_root,
        source_root=args.source_root,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


def _cmd_aggregate(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    result = aggregate_workers(
        workers_root=Path(args.output_root) / "workers",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args):
    output_root = Path(args.output_root).resolve()
    source_root = output_root / "experiment_25_source"
    source_worker = (
        source_root / "workers" / "task_002_averaged_critic_target_seed_0"
    )
    if source_worker.exists() and not args.resume:
        shutil.rmtree(source_worker)
    LOGGER.info("Creating tiny Experiment 25-compatible source states")
    source_result = run_experiment_25_worker(
        variant_id=AVERAGED_TARGET_ONLY,
        seed=0,
        schedule=experiment_25_schedule(smoke=True),
        worker_dir=source_worker,
        smoke=True,
        resume=args.resume,
    )
    write_csv(
        source_root / "analysis" / "training_state_inventory.csv",
        [
            {**row, "seed": 0, "variant_id": AVERAGED_TARGET_ONLY}
            for row in source_result["training_states"]
        ],
    )
    validate_contract(seeds=SMOKE_SEEDS, smoke=True)
    _run_task(
        task_index=0,
        seeds=SMOKE_SEEDS,
        output_root=output_root,
        source_root=source_root,
        smoke=True,
        resume=args.resume,
    )
    result = aggregate_workers(
        workers_root=output_root / "workers",
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    print(
        json.dumps(
            {
                "tasks": [
                    {"task_index": index, "source_seed": seed}
                    for index, seed in enumerate(task_schedule(args.seeds))
                ]
            },
            indent=2,
        )
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--source-root", type=Path, required=True)
    worker.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--no-resume", dest="resume", action="store_false")
    worker.set_defaults(func=_cmd_worker, resume=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(func=_cmd_aggregate)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    schedule.add_argument("--smoke", action="store_true")
    schedule.set_defaults(func=_cmd_schedule)
    return parser


def main():
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
