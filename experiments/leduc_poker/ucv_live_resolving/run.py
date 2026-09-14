"""Run and aggregate Experiment 32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import pyspiel
from open_spiel.python import policy

from experiments.leduc_poker.live_forward_search_common import (
    add_cfr_comparison,
    aggregate_experiment,
    load_blueprint,
    read_json,
    run_resolver_worker,
)
from .config import (
    EXPERIMENT_NAME,
    EXPLORATION,
    PRODUCTION_SEEDS,
    RESOLVER_KINDS,
    SEARCH_UPDATE_BUDGETS,
    SMOKE_SEARCH_UPDATE_BUDGETS,
    SMOKE_SEEDS,
    SOURCE_CHECKPOINT_ID,
    contract_manifest,
    resolver_seeds,
)


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
    return Path(root) / "workers" / f"task_{index:03d}_ucv_resolve_seed_{seed}"


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
        resolver_kind=RESOLVER_KINDS,
        seed=seed,
        blueprint=blueprint,
        source_record=source,
        budgets=(SMOKE_SEARCH_UPDATE_BUDGETS if args.smoke else SEARCH_UPDATE_BUDGETS),
        resolver_seeds=resolver_seeds(seed, smoke=args.smoke),
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
    output = Path(args.output_root) / "analysis"
    result = aggregate_experiment(
        workers_root=Path(args.output_root) / "workers",
        output_dir=output,
        experiment_name=EXPERIMENT_NAME,
        seeds=args.seeds,
        smoke=args.smoke,
        contract=contract_manifest(smoke=args.smoke),
    )
    comparison = add_cfr_comparison(
        ucv_analysis_dir=output, experiment_31_root=args.experiment_31_root
    )
    result["artifacts"].update(comparison)
    from experiments.leduc_poker.live_forward_search_common import write_json
    write_json(output / "aggregate_manifest.json", result)
    print(json.dumps(result, indent=2))


def _cmd_smoke(args):
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.resume:
        shutil.rmtree(output_root)
    exp31_root = output_root / "experiment_31_reference"
    subprocess.run(
        [
            "python3", "-m",
            "experiments.leduc_poker.tabular_cfr_live_resolving.run",
            "smoke", "--output-root", str(exp31_root), "--no-resume",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    args.smoke = True
    args.seeds = SMOKE_SEEDS
    args.task_index = 0
    args.experiment_31_root = exp31_root
    _run_worker(args)
    _cmd_aggregate(args)


def _cmd_schedule(args):
    _normalise(args)
    print(json.dumps({"tasks": [
        {"task_index": index, "source_seed": seed,
         "resolver_seeds": list(resolver_seeds(seed, smoke=args.smoke))}
        for index, seed in enumerate(args.seeds)
    ], "contract": contract_manifest(smoke=args.smoke)}, indent=2))


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
    aggregate.add_argument("--experiment-31-root", type=Path, required=True)
    aggregate.add_argument("--seeds", type=_parse_seeds, default=PRODUCTION_SEEDS)
    aggregate.add_argument("--smoke", action="store_true")
    aggregate.set_defaults(func=_cmd_aggregate)
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
    args.func(args)


if __name__ == "__main__":
    main()
