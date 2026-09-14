"""Run and aggregate Experiment 31."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

import pyspiel
from open_spiel.python import policy

from experiments.leduc_poker.live_forward_search_common import (
    aggregate_experiment,
    load_blueprint,
    read_json,
    run_resolver_worker,
)
from .config import (
    EXPERIMENT_NAME,
    EXPLORATION,
    PRODUCTION_SEEDS,
    RESOLVER_KIND,
    SEARCH_UPDATE_BUDGETS,
    SMOKE_SEARCH_UPDATE_BUDGETS,
    SMOKE_SEEDS,
    SOURCE_CHECKPOINT_ID,
    contract_manifest,
)


LOGGER = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _parse_seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("At least one seed is required")
    return result


def _normalise(args) -> None:
    if args.smoke and args.seeds == PRODUCTION_SEEDS:
        args.seeds = SMOKE_SEEDS
    expected = SMOKE_SEEDS if args.smoke else PRODUCTION_SEEDS
    if tuple(args.seeds) != tuple(expected):
        raise ValueError(f"Seeds must be {expected}, got {args.seeds}")


def _worker_dir(root: Path, index: int, seed: int) -> Path:
    return Path(root) / "workers" / f"task_{index:03d}_tabular_cfr_resolve_seed_{seed}"


def _run_worker(args):
    _normalise(args)
    seed = int(args.seeds[args.task_index])
    worker_dir = _worker_dir(args.output_root, args.task_index, seed)
    existing = worker_dir / "worker_result.json"
    if args.resume and existing.is_file():
        result = read_json(existing)
        if result.get("status") == "complete" and result.get("repository_commit") == _commit():
            return result
    game = pyspiel.load_game("leduc_poker")
    if args.smoke:
        blueprint = policy.UniformRandomPolicy(game)
        source = {"source_experiment_id": "synthetic_smoke", "path": "uniform_policy"}
    else:
        blueprint, source = load_blueprint(
            game, args.source_root, seed, SOURCE_CHECKPOINT_ID
        )
    return run_resolver_worker(
        experiment_name=EXPERIMENT_NAME,
        resolver_kind=RESOLVER_KIND,
        seed=seed,
        blueprint=blueprint,
        source_record=source,
        budgets=(SMOKE_SEARCH_UPDATE_BUDGETS if args.smoke else SEARCH_UPDATE_BUDGETS),
        resolver_seeds=(31_000_000 + seed,),
        exploration=EXPLORATION,
        worker_dir=worker_dir,
        smoke=args.smoke,
        repository_commit=_commit(),
        contract=contract_manifest(smoke=args.smoke),
    )


def _cmd_worker(args):
    print(json.dumps(_run_worker(args), indent=2))


def _cmd_aggregate(args):
    _normalise(args)
    result = aggregate_experiment(
        workers_root=Path(args.output_root) / "workers",
        output_dir=Path(args.output_root) / "analysis",
        experiment_name=EXPERIMENT_NAME,
        seeds=args.seeds,
        smoke=args.smoke,
        contract=contract_manifest(smoke=args.smoke),
    )
    print(json.dumps(result, indent=2))


def _cmd_smoke(args):
    args.smoke = True
    args.seeds = SMOKE_SEEDS
    args.task_index = 0
    if Path(args.output_root).exists() and not args.resume:
        shutil.rmtree(args.output_root)
    _run_worker(args)
    _cmd_aggregate(args)


def _cmd_schedule(args):
    _normalise(args)
    print(json.dumps({"tasks": [
        {"task_index": index, "source_seed": seed}
        for index, seed in enumerate(args.seeds)
    ], "contract": contract_manifest(smoke=args.smoke)}, indent=2))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("worker", _cmd_worker), ("aggregate", _cmd_aggregate)):
        item = sub.add_parser(name)
        item.add_argument("--output-root", type=Path, required=True)
        item.add_argument("--source-root", type=Path, default=Path("."))
        item.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
        item.add_argument("--smoke", action="store_true")
        if name == "worker":
            item.add_argument("--task-index", type=int, required=True)
            item.add_argument("--no-resume", dest="resume", action="store_false")
            item.set_defaults(resume=True)
        item.set_defaults(func=function)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output-root", type=Path, required=True)
    smoke.add_argument("--source-root", type=Path, default=Path("."))
    smoke.add_argument("--no-resume", dest="resume", action="store_false")
    smoke.set_defaults(func=_cmd_smoke, resume=True)
    schedule = sub.add_parser("schedule")
    schedule.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    schedule.add_argument("--smoke", action="store_true")
    schedule.set_defaults(func=_cmd_schedule)
    return parser


def main():
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
