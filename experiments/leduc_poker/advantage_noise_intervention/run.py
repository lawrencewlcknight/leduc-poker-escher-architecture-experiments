"""Experiment 52 CLI: frozen-checkpoint oracle intervention, never a training run."""
import argparse
import json
from pathlib import Path
from .config import contract, tasks
from .source import fetch_source
from .worker import run_worker, code_identity, runtime, write_json
from .analyse import aggregate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("contract")
    for name in ("worker", "pilot", "smoke", "aggregate"):
        parser = sub.add_parser(name)
        parser.add_argument("--output-root", type=Path, required=True)
        if name in ("worker", "pilot"):
            parser.add_argument("--task-index", type=int, choices=range(len(tasks())), default=0)
            parser.add_argument("--source-worker", type=Path, required=True)
    fetch = sub.add_parser("fetch-source")
    fetch.add_argument("--bucket", required=True)
    fetch.add_argument("--source-run-id", default="exp35-confirm-20260916-011231")
    fetch.add_argument("--task-index", type=int, choices=range(len(tasks())), required=True)
    fetch.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    if args.action == "contract":
        print(json.dumps(contract(), indent=2))
    elif args.action == "fetch-source":
        fetch_source(args.bucket, args.source_run_id, args.task_index, args.directory)
    elif args.action == "aggregate":
        print(json.dumps(aggregate(args.output_root), indent=2))
    elif args.action == "smoke":
        run_worker(args.output_root, smoke=True)
        aggregate(args.output_root, smoke=True)
        write_json(args.output_root/"SMOKE_SUCCESS.json", {"status": "complete", "code": code_identity(), "runtime": runtime()})
    else:
        result = run_worker(args.output_root, args.task_index, args.source_worker, pilot=args.action == "pilot")
        if args.action == "pilot":
            write_json(args.output_root/"PILOT_SUCCESS.json", {"status": "complete", "code": code_identity(),
                "runtime": runtime(), "result": result, "scope": "one production-size paired repetition; not pooled with the study"})


if __name__ == "__main__":
    main()
