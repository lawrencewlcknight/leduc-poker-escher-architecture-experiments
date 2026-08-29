"""CLI for workers, exact aggregation, schedule inspection, and smoke testing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

from .analyse import aggregate_workers
from .common import read_json, sha256
from .config import (
    ALGORITHM_ORDER,
    DEEP_CFR,
    HELDOUT_SEEDS,
    SMOKE_SEEDS,
    TARGET_ACTIVE_SECONDS,
    TARGET_NODES,
    task_schedule,
    validate_contract,
)
from .worker import run_architecture_worker


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
    target_nodes: int,
    target_seconds: float,
    smoke: bool,
) -> dict | None:
    result_path = worker_dir / "worker_result.json"
    if not result_path.is_file():
        return None
    result = read_json(result_path)
    if result.get("status") != "complete" or len(result.get("snapshots", [])) != 2:
        return None
    if (
        result.get("algorithm_id") != algorithm_id
        or int(result.get("seed", -1)) != int(seed)
        or result.get("repository_commit") != expected_commit
        or int(result.get("target_nodes", -1)) != int(target_nodes)
        or float(result.get("target_active_seconds", -1)) != float(target_seconds)
        or bool(result.get("smoke")) != bool(smoke)
    ):
        return None
    for record in result["snapshots"]:
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
    target_nodes: int,
    target_seconds: float,
    smoke: bool,
    resume: bool,
) -> dict:
    expected_commit = _git_commit(
        Path(deep_cfr_repo).resolve() if algorithm_id == DEEP_CFR else REPOSITORY_ROOT
    )
    if resume:
        existing = _complete_worker(
            worker_dir,
            algorithm_id=algorithm_id,
            seed=seed,
            expected_commit=expected_commit,
            target_nodes=target_nodes,
            target_seconds=target_seconds,
            smoke=smoke,
        )
        if existing is not None:
            LOGGER.info("Reusing validated complete worker %s seed %s", algorithm_id, seed)
            return existing
    if algorithm_id == DEEP_CFR:
        command = [
            sys.executable,
            str(Path(__file__).with_name("deep_worker.py")),
            "--deep-cfr-repo",
            str(Path(deep_cfr_repo).resolve()),
            "--seed",
            str(seed),
            "--target-nodes",
            str(target_nodes),
            "--target-seconds",
            str(target_seconds),
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
            target_nodes=target_nodes,
            target_seconds=target_seconds,
            smoke=smoke,
        )
        if result is None:
            raise RuntimeError("Deep CFR subprocess returned without a valid worker archive")
        return result
    return run_architecture_worker(
        algorithm_id=algorithm_id,
        seed=seed,
        target_nodes=target_nodes,
        target_seconds=target_seconds,
        worker_dir=worker_dir,
        smoke=smoke,
    )


def _worker_dir(output_root: Path, task_index: int, algorithm_id: str, seed: int) -> Path:
    return Path(output_root) / "workers" / f"task_{task_index:03d}_{algorithm_id}_seed_{seed}"


def _cmd_worker(args) -> None:
    validate_contract(
        seeds=args.seeds,
        target_nodes=args.target_nodes,
        target_seconds=args.target_seconds,
        smoke=args.smoke,
    )
    schedule = task_schedule(args.seeds)
    if args.task_index < 0 or args.task_index >= len(schedule):
        raise ValueError(
            f"Task index {args.task_index} outside [0, {len(schedule) - 1}]"
        )
    algorithm_id, seed = schedule[args.task_index]
    worker_dir = _worker_dir(args.output_root, args.task_index, algorithm_id, seed)
    result = _run_one_worker(
        algorithm_id=algorithm_id,
        seed=seed,
        worker_dir=worker_dir,
        deep_cfr_repo=args.deep_cfr_repo,
        target_nodes=args.target_nodes,
        target_seconds=args.target_seconds,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps({"task_index": args.task_index, **result}, indent=2, default=str))


def _cmd_aggregate(args) -> None:
    validate_contract(
        seeds=args.seeds,
        target_nodes=TARGET_NODES,
        target_seconds=TARGET_ACTIVE_SECONDS,
        smoke=args.smoke,
    )
    result = aggregate_workers(
        workers_root=Path(args.output_root) / "workers",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args) -> None:
    seeds = SMOKE_SEEDS
    validate_contract(seeds=seeds, target_nodes=1, target_seconds=0, smoke=True)
    output_root = Path(args.output_root).resolve()
    schedule = task_schedule(seeds)
    for task_index, (algorithm_id, seed) in enumerate(schedule):
        LOGGER.info(
            "Smoke worker %s/%s: %s seed %s",
            task_index + 1,
            len(schedule),
            algorithm_id,
            seed,
        )
        _run_one_worker(
            algorithm_id=algorithm_id,
            seed=seed,
            worker_dir=_worker_dir(output_root, task_index, algorithm_id, seed),
            deep_cfr_repo=args.deep_cfr_repo,
            target_nodes=1,
            target_seconds=0,
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
    validate_contract(
        seeds=args.seeds,
        target_nodes=TARGET_NODES,
        target_seconds=TARGET_ACTIVE_SECONDS,
        smoke=args.smoke,
    )
    rows = [
        {"task_index": index, "algorithm_id": algorithm_id, "seed": seed}
        for index, (algorithm_id, seed) in enumerate(task_schedule(args.seeds))
    ]
    print(json.dumps(rows, indent=2))


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="Run one stable array task")
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--deep-cfr-repo", type=Path, default=DEFAULT_DEEP_CFR_REPO)
    worker.add_argument("--seeds", type=_parse_seeds, default=HELDOUT_SEEDS)
    worker.add_argument("--target-nodes", type=int, default=TARGET_NODES)
    worker.add_argument("--target-seconds", type=float, default=TARGET_ACTIVE_SECONDS)
    worker.add_argument("--smoke", action="store_true")
    worker.add_argument("--no-resume", dest="resume", action="store_false")
    worker.set_defaults(func=_cmd_worker, resume=True)

    aggregate = subparsers.add_parser("aggregate", help="Validate and analyse all workers")
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--seeds", type=_parse_seeds, default=HELDOUT_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(func=_cmd_aggregate)

    smoke = subparsers.add_parser("smoke", help="Run all four tiny workers and analysis")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--deep-cfr-repo", type=Path, default=DEFAULT_DEEP_CFR_REPO)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)

    schedule = subparsers.add_parser("schedule", help="Print the array-task mapping")
    schedule.add_argument("--seeds", type=_parse_seeds, default=HELDOUT_SEEDS)
    schedule.add_argument("--smoke", action="store_true")
    schedule.set_defaults(func=_cmd_schedule)
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
