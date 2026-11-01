"""Single-line Batch worker identities, isolated from noisy optional imports.

Keep top-level imports stdlib-only: stdout is a machine-readable shell value.
Diagnostics from experiment configuration imports belong on stderr instead.
"""
import argparse
from contextlib import redirect_stdout
import importlib
import os
import re
import sys


CONFIG_MODULES = {
    50: "experiments.leduc_poker.critic_averaging_mechanism.config",
    51: "experiments.leduc_poker.critic_update_budget.config",
    52: "experiments.leduc_poker.advantage_noise_intervention.config",
}


def validate_worker_uri(uri):
    """Validate the restricted GCS output prefixes used by these launchers."""
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]*/[A-Za-z0-9._/-]+", uri):
        raise ValueError("Invalid worker URI: expected a gs://bucket/path without whitespace")
    if any(part in ("", ".", "..") for part in uri[5:].split("/")):
        raise ValueError("Invalid worker URI: empty or relative path segment")
    return uri


def worker_remote_from_env(variable):
    """Unset means local-only; a set but invalid destination must fail early."""
    remote = os.environ.get(variable)
    return None if remote is None else validate_worker_uri(remote)


def worker_name(experiment, index):
    if experiment not in CONFIG_MODULES:
        raise ValueError(f"Unknown experiment: {experiment}")
    if not isinstance(index, int) or index < 0:
        raise ValueError("Task index must be a nonnegative integer")
    # OpenSpiel may print optional-module diagnostics when these modules load.
    # Preserve those diagnostics without letting them become path components.
    with redirect_stdout(sys.stderr):
        schedule = importlib.import_module(CONFIG_MODULES[experiment]).tasks()
    if index >= len(schedule):
        raise ValueError(f"Task index {index} outside schedule of {len(schedule)} workers")
    first, second = schedule[index]
    if experiment == 52:
        seed, checkpoint = first, second
        if checkpoint not in ("time_24h", "time_36h"):
            raise ValueError("Unexpected checkpoint in worker schedule")
        name = f"seed_{seed}_{checkpoint}"
    else:
        arm, seed = first, second
        if arm not in ("A", "B", "C", "D"):
            raise ValueError("Unexpected arm in worker schedule")
        name = f"task_{index:03d}_{arm}_{seed}"
    if not isinstance(seed, int) or seed < 0 or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("Invalid worker identity in schedule")
    return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=tuple(CONFIG_MODULES), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args(argv)
    try:
        validate_worker_uri(args.remote_root)
        name = worker_name(args.experiment, args.task_index)
        validate_worker_uri(f"{args.remote_root}/workers/{name}")
    except ValueError as error:
        parser.error(str(error))
    print(name)


if __name__ == "__main__":
    main()
