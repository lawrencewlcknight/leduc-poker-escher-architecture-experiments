"""CLI for Experiment 21 workers, exact aggregation, and smoke testing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from .common import read_json, sha256
from .config import (
    ALGORITHM_ORDER,
    DEEP_CFR,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    validate_contract,
)
from .worker import run_ucv_worker


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEEP_CFR_REPO = (
    REPOSITORY_ROOT.parents[1]
    / "leduc_poker_deep_cfr"
    / "leduc-poker-deep-cfr-experiments"
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return seeds


def _git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def _complete_worker(
    worker_dir: Path,
    *,
    algorithm_id: str,
    seed: int,
    expected_commit: str,
    schedule: Sequence[Mapping],
    smoke: bool,
) -> dict | None:
    result_path = worker_dir / "worker_result.json"
    if not result_path.is_file():
        return None
    result = read_json(result_path)
    records = list(result.get("snapshots", ()))
    if result.get("status") != "complete" or len(records) != len(schedule):
        return None
    if (
        result.get("algorithm_id") != algorithm_id
        or int(result.get("seed", -1)) != int(seed)
        or result.get("repository_commit") != expected_commit
        or bool(result.get("smoke")) != bool(smoke)
        or tuple(result.get("checkpoint_schedule", ())) != tuple(schedule)
    ):
        return None
    expected_ids = [str(row["checkpoint_id"]) for row in schedule]
    if [str(row["checkpoint_id"]) for row in records] != expected_ids:
        return None
    for record in records:
        path = worker_dir / record["relative_path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            return None
    return result


def _run_one_worker(
    *,
    algorithm_id: str,
    seed: int,
    worker_dir: Path,
    deep_cfr_repo: Path,
    schedule: Sequence[Mapping],
    smoke: bool,
    resume: bool,
) -> dict:
    source_repository = (
        Path(deep_cfr_repo).resolve() if algorithm_id == DEEP_CFR else REPOSITORY_ROOT
    )
    expected_commit = _git_commit(source_repository)
    if resume:
        existing = _complete_worker(
            worker_dir,
            algorithm_id=algorithm_id,
            seed=seed,
            expected_commit=expected_commit,
            schedule=schedule,
            smoke=smoke,
        )
        if existing is not None:
            LOGGER.info("Reusing validated worker %s seed %s", algorithm_id, seed)
            return existing
    if algorithm_id == DEEP_CFR:
        command = [
            sys.executable,
            str(Path(__file__).with_name("deep_worker.py")),
            "--deep-cfr-repo",
            str(Path(deep_cfr_repo).resolve()),
            "--seed",
            str(seed),
            "--schedule-json",
            json.dumps(list(schedule), separators=(",", ":")),
            "--worker-dir",
            str(worker_dir),
        ]
        if smoke:
            command.append("--smoke")
        subprocess.run(command, check=True)
        result = _complete_worker(
            worker_dir,
            algorithm_id=algorithm_id,
            seed=seed,
            expected_commit=expected_commit,
            schedule=schedule,
            smoke=smoke,
        )
        if result is None:
            raise RuntimeError("Deep CFR subprocess returned without a valid archive")
        return result
    return run_ucv_worker(
        seed=seed,
        schedule=schedule,
        worker_dir=worker_dir,
        smoke=smoke,
    )


def _worker_dir(output_root: Path, task_index: int, algorithm_id: str, seed: int) -> Path:
    return Path(output_root) / "workers" / f"task_{task_index:03d}_{algorithm_id}_seed_{seed}"


def _cmd_worker(args) -> None:
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    tasks = task_schedule(args.seeds)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ValueError(f"Task index {args.task_index} outside [0, {len(tasks) - 1}]")
    algorithm_id, seed = tasks[args.task_index]
    result = _run_one_worker(
        algorithm_id=algorithm_id,
        seed=seed,
        worker_dir=_worker_dir(
            args.output_root, args.task_index, algorithm_id, seed
        ),
        deep_cfr_repo=args.deep_cfr_repo,
        schedule=schedule,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps({"task_index": args.task_index, **result}, indent=2, default=str))


def _cmd_aggregate(args) -> None:
    # Keep scheduling and individual worker commands independent of plotting
    # dependencies.  The aggregate stage is the only stage that needs them.
    from .analyse import aggregate_workers

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

    seeds = SMOKE_SEEDS
    schedule = checkpoint_schedule(smoke=True)
    validate_contract(seeds=seeds, schedule=schedule, smoke=True)
    output_root = Path(args.output_root).resolve()
    for task_index, (algorithm_id, seed) in enumerate(task_schedule(seeds)):
        LOGGER.info(
            "Smoke worker %s/%s: %s seed %s",
            task_index + 1,
            len(ALGORITHM_ORDER),
            algorithm_id,
            seed,
        )
        _run_one_worker(
            algorithm_id=algorithm_id,
            seed=seed,
            worker_dir=_worker_dir(output_root, task_index, algorithm_id, seed),
            deep_cfr_repo=args.deep_cfr_repo,
            schedule=schedule,
            smoke=True,
            resume=args.resume,
        )
    result = aggregate_workers(
        workers_root=output_root / "workers",
        seeds=seeds,
        output_dir=output_root / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args) -> None:
    schedule = checkpoint_schedule(smoke=args.smoke)
    validate_contract(seeds=args.seeds, schedule=schedule, smoke=args.smoke)
    print(
        json.dumps(
            {
                "tasks": [
                    {"task_index": index, "algorithm_id": algorithm_id, "seed": seed}
                    for index, (algorithm_id, seed) in enumerate(
                        task_schedule(args.seeds)
                    )
                ],
                "checkpoints": list(schedule),
            },
            indent=2,
        )
    )


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="Run one stable array task")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--deep-cfr-repo", type=Path, default=DEFAULT_DEEP_CFR_REPO)
    worker.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--no-resume", dest="resume", action="store_false")
    worker.set_defaults(func=_cmd_worker, resume=True)

    aggregate = subparsers.add_parser("aggregate", help="Validate and analyse all workers")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(func=_cmd_aggregate)

    smoke = subparsers.add_parser("smoke", help="Run both tiny workers and analysis")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--deep-cfr-repo", type=Path, default=DEFAULT_DEEP_CFR_REPO)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)

    schedule_parser = subparsers.add_parser("schedule", help="Print tasks and checkpoints")
    schedule_parser.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    schedule_parser.add_argument("--smoke", action="store_true")
    schedule_parser.set_defaults(func=_cmd_schedule)
    return parser


def main() -> None:
    args = _base_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
