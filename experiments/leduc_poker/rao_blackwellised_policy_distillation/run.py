"""CLI for Experiment 34's staged average-policy fitting study."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import read_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    checkpoint_schedule as experiment_29_schedule,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.worker import (
    run_worker as run_experiment_29_worker,
)

from .analyse import aggregate_results, select_configs
from .config import PRODUCTION_SEEDS, SMOKE_SEEDS, task_schedule, validate_contract
from .worker import run_development_worker, run_validation_worker


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


def _worker_dir(root: Path, stage: str, index: int, seed: int) -> Path:
    return Path(root) / f"{stage}_workers" / f"task_{index:03d}_{stage}_seed_{seed}"


def _completed(path: Path, stage: str, seed: int) -> dict | None:
    result_path = path / "worker_result.json"
    if not result_path.is_file():
        return None
    result = read_json(result_path)
    if (
        result.get("status") == "complete"
        and result.get("stage") == stage
        and int(result.get("source_seed", -1)) == int(seed)
        and result.get("repository_commit") == _repository_commit()
    ):
        return result
    return None


def _seed_for_task(index: int, seeds) -> int:
    tasks = task_schedule(seeds)
    if index < 0 or index >= len(tasks):
        raise ValueError(f"Task index {index} outside [0, {len(tasks) - 1}]")
    return int(tasks[index])


def _normalise_smoke(args) -> None:
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)


def _cmd_development_worker(args) -> None:
    _normalise_smoke(args)
    seed = _seed_for_task(args.task_index, args.seeds)
    worker_dir = _worker_dir(args.output_root, "development", args.task_index, seed)
    result = _completed(worker_dir, "development", seed) if args.resume else None
    if result is None:
        result = run_development_worker(
            seed=seed,
            source_root=args.source_root,
            worker_dir=worker_dir,
            smoke=args.smoke,
        )
    print(json.dumps(result, indent=2))


def _cmd_select(args) -> None:
    _normalise_smoke(args)
    result = select_configs(
        development_root=Path(args.output_root) / "development_workers",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "selection",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_validation_worker(args) -> None:
    _normalise_smoke(args)
    seed = _seed_for_task(args.task_index, args.seeds)
    worker_dir = _worker_dir(args.output_root, "validation", args.task_index, seed)
    result = _completed(worker_dir, "validation", seed) if args.resume else None
    if result is None:
        result = run_validation_worker(
            seed=seed,
            source_root=args.source_root,
            selection_path=Path(args.output_root) / "selection" / "selected_configs.json",
            worker_dir=worker_dir,
            smoke=args.smoke,
        )
    print(json.dumps(result, indent=2))


def _cmd_aggregate(args) -> None:
    _normalise_smoke(args)
    result = aggregate_results(
        development_root=Path(args.output_root) / "development_workers",
        validation_root=Path(args.output_root) / "validation_workers",
        selection_path=Path(args.output_root) / "selection" / "selected_configs.json",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args) -> None:
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.resume:
        shutil.rmtree(output_root)
    source_root = output_root / "experiment_29_source"
    source_worker = source_root / "workers" / "task_000_promoted_ucv_cross_entropy_seed_0"
    LOGGER.info("Creating tiny Experiment 29 source")
    run_experiment_29_worker(
        seed=0,
        schedule=experiment_29_schedule(smoke=True),
        worker_dir=source_worker,
        smoke=True,
        resume=args.resume,
    )
    run_development_worker(
        seed=0,
        source_root=source_root,
        worker_dir=_worker_dir(output_root, "development", 0, 0),
        smoke=True,
    )
    select_configs(
        development_root=output_root / "development_workers",
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "selection",
        smoke=True,
    )
    run_validation_worker(
        seed=0,
        source_root=source_root,
        selection_path=output_root / "selection" / "selected_configs.json",
        worker_dir=_worker_dir(output_root, "validation", 0, 0),
        smoke=True,
    )
    result = aggregate_results(
        development_root=output_root / "development_workers",
        validation_root=output_root / "validation_workers",
        selection_path=output_root / "selection" / "selected_configs.json",
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args) -> None:
    _normalise_smoke(args)
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("development-worker", "validation-worker"):
        worker = sub.add_parser(command)
        worker.add_argument("--task-index", type=int, required=True)
        worker.add_argument("--output-root", type=Path, required=True)
        worker.add_argument("--source-root", type=Path, required=True)
        worker.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
        worker.add_argument("--smoke", action="store_true")
        worker.add_argument("--no-resume", dest="resume", action="store_false")
        worker.set_defaults(resume=True)
    sub.choices["development-worker"].set_defaults(func=_cmd_development_worker)
    sub.choices["validation-worker"].set_defaults(func=_cmd_validation_worker)
    for command, function in (("select", _cmd_select), ("aggregate", _cmd_aggregate)):
        stage = sub.add_parser(command)
        stage.add_argument("--output-root", type=Path, required=True)
        stage.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
        stage.add_argument("--smoke", action="store_true")
        stage.set_defaults(func=function)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)
    schedule = sub.add_parser("schedule")
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
