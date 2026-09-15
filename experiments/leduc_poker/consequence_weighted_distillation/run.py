"""CLI for Experiment 33's staged proxy-selection and redistillation study."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

from experiments.leduc_poker.causal_rare_state_audit.worker import (
    run_worker as run_experiment_30_worker,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import read_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    checkpoint_schedule as experiment_29_schedule,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.worker import (
    run_worker as run_experiment_29_worker,
)

from .analyse import aggregate_proxy_workers, aggregate_results
from .config import PRODUCTION_SEEDS, SMOKE_SEEDS, task_schedule, validate_contract
from .distill_worker import run_distill_worker
from .proxy_worker import run_proxy_worker


LOGGER = logging.getLogger(__name__)


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


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


def _cmd_proxy_worker(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    seed = _seed_for_task(args.task_index, args.seeds)
    worker_dir = _worker_dir(args.output_root, "proxy", args.task_index, seed)
    result = _completed(worker_dir, "proxy", seed) if args.resume else None
    if result is None:
        result = run_proxy_worker(
            seed=seed,
            source_root=args.source_root,
            audit_root=args.audit_root,
            worker_dir=worker_dir,
            smoke=args.smoke,
        )
    print(json.dumps(result, indent=2))


def _cmd_select(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    result = aggregate_proxy_workers(
        proxy_root=Path(args.output_root) / "proxy_workers",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "selection",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_distill_worker(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    seed = _seed_for_task(args.task_index, args.seeds)
    worker_dir = _worker_dir(args.output_root, "distill", args.task_index, seed)
    result = _completed(worker_dir, "distill", seed) if args.resume else None
    if result is None:
        result = run_distill_worker(
            seed=seed,
            source_root=args.source_root,
            proxy_root=Path(args.output_root) / "proxy_workers",
            selection_path=Path(args.output_root) / "selection" / "selected_proxy.json",
            worker_dir=worker_dir,
            smoke=args.smoke,
        )
    print(json.dumps(result, indent=2))


def _cmd_aggregate(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    result = aggregate_results(
        proxy_root=Path(args.output_root) / "proxy_workers",
        distill_root=Path(args.output_root) / "distill_workers",
        selection_path=Path(args.output_root) / "selection" / "selected_proxy.json",
        seeds=args.seeds,
        output_dir=Path(args.output_root) / "analysis",
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args):
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.resume:
        shutil.rmtree(output_root)
    source_root = output_root / "experiment_29_source"
    source_worker = source_root / "workers" / "task_000_promoted_ucv_cross_entropy_seed_0"
    audit_root = output_root / "experiment_30_audit"
    audit_worker = audit_root / "workers" / "task_000_rare_state_audit_seed_0"
    LOGGER.info("Creating tiny Experiment 29 source and Experiment 30 audit")
    run_experiment_29_worker(
        seed=0,
        schedule=experiment_29_schedule(smoke=True),
        worker_dir=source_worker,
        smoke=True,
        resume=args.resume,
    )
    audit_result = audit_worker / "worker_result.json"
    if not (args.resume and audit_result.is_file()):
        run_experiment_30_worker(
            seed=0,
            source_root=source_root,
            worker_dir=audit_worker,
            smoke=True,
        )
    run_proxy_worker(
        seed=0,
        source_root=source_root,
        audit_root=audit_root,
        worker_dir=_worker_dir(output_root, "proxy", 0, 0),
        smoke=True,
    )
    aggregate_proxy_workers(
        proxy_root=output_root / "proxy_workers",
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "selection",
        smoke=True,
    )
    run_distill_worker(
        seed=0,
        source_root=source_root,
        proxy_root=output_root / "proxy_workers",
        selection_path=output_root / "selection" / "selected_proxy.json",
        worker_dir=_worker_dir(output_root, "distill", 0, 0),
        smoke=True,
    )
    result = aggregate_results(
        proxy_root=output_root / "proxy_workers",
        distill_root=output_root / "distill_workers",
        selection_path=output_root / "selection" / "selected_proxy.json",
        seeds=SMOKE_SEEDS,
        output_dir=output_root / "analysis",
        smoke=True,
    )
    print(json.dumps(result, indent=2))


def _cmd_schedule(args):
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    validate_contract(seeds=args.seeds, smoke=args.smoke)
    tasks = [
        {"task_index": index, "source_seed": seed}
        for index, seed in enumerate(task_schedule(args.seeds))
    ]
    print(json.dumps({"tasks": tasks}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("proxy-worker", "distill-worker"):
        worker = sub.add_parser(command)
        worker.add_argument("--task-index", type=int, required=True)
        worker.add_argument("--output-root", type=Path, required=True)
        worker.add_argument("--source-root", type=Path, required=True)
        worker.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
        worker.add_argument("--smoke", action="store_true")
        worker.add_argument("--no-resume", dest="resume", action="store_false")
        worker.set_defaults(resume=True)
    sub.choices["proxy-worker"].add_argument("--audit-root", type=Path, required=True)
    sub.choices["proxy-worker"].set_defaults(func=_cmd_proxy_worker)
    sub.choices["distill-worker"].set_defaults(func=_cmd_distill_worker)
    select = sub.add_parser("select")
    select.add_argument("--output-root", type=Path, required=True)
    select.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    select.add_argument("--smoke", action="store_true")
    select.set_defaults(func=_cmd_select)
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


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
