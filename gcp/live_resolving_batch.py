#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for Experiments 31 and 32."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
MODULES = {
    31: "experiments.leduc_poker.tabular_cfr_live_resolving.run",
    32: "experiments.leduc_poker.ucv_live_resolving.run",
}


def _q(value) -> str:
    return shlex.quote(str(value))


def _bootstrap(args) -> str:
    return f"""
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/cache
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
UV_INSTALL_DIR=/tmp/uv-bin
REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
BUCKET_ROOT={_q(args.bucket_root.rstrip('/'))}
RUN_ID={_q(args.run_id)}
EXP29_RUN_ID={_q(args.experiment_29_run_id)}
EXP31_RUN_ID={_q(args.experiment_31_run_id)}
WORK_ROOT=/workspace/live-resolving-exp{args.experiment_id}
REPOSITORY="$WORK_ROOT/repository"
OUTPUT_ROOT="$WORK_ROOT/output"
SOURCE_ROOT="$WORK_ROOT/experiment_29_source"
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
mkdir -p "$WORK_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_INSTALL_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
curl -LsSf https://astral.sh/uv/install.sh | \\
  env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
export PATH="$UV_INSTALL_DIR:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed /tmp/live-resolving-venv
source /tmp/live-resolving-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r "$REPOSITORY/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation -e "$REPOSITORY"
python -m pip check
cd "$REPOSITORY"
""".strip()


def _controller_script(args) -> str:
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
WORK_ROOT=/workspace/live-resolving-controller
REPOSITORY="$WORK_ROOT/repository"
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git ca-certificates python3
mkdir -p "$WORK_ROOT"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
cd "$REPOSITORY"
export PROJECT_ID={_q(args.project_id)}
export REGION={_q(args.region)}
export BUCKET={_q(args.bucket_root.rstrip('/'))}
export SA_EMAIL={_q(args.service_account)}
export REPO_REF={_q(args.repo_ref)}
export EXP29_RUN_ID={_q(args.experiment_29_run_id)}
export EXP31_RUN_ID={_q(args.experiment_31_run_id)}
export RUN_ID={_q(args.run_id)}
export PARALLELISM={_q(args.parallelism)}
export LIVE_RESOLVING_REMOTE_CONTROLLER=1
exec bash gcp/run_live_resolving_experiment.sh {args.experiment_id} {_q(args.controller_action)}
"""


def _script(args) -> str:
    if args.kind == "controller":
        return _controller_script(args)
    module = MODULES[args.experiment_id]
    bootstrap = _bootstrap(args)
    if args.kind == "smoke":
        action = f"""
python -m {module} smoke --output-root "$OUTPUT_ROOT" --no-resume
gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID/smoke"
""".strip()
    elif args.kind == "train":
        package = (
            "tabular_cfr_live_resolving" if args.experiment_id == 31
            else "ucv_live_resolving"
        )
        task_label = "tabular_cfr_resolve" if args.experiment_id == 31 else "ucv_resolve"
        action = f"""
TASK_INDEX="${{BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}}"
SOURCE_SEED="$(python - "$TASK_INDEX" <<'PY' | tail -n 1
import sys
from experiments.leduc_poker.{package}.config import PRODUCTION_SEEDS
print(PRODUCTION_SEEDS[int(sys.argv[1])])
PY
)"
SOURCE_TASK="task_$(printf '%03d' "$TASK_INDEX")_promoted_ucv_cross_entropy_seed_$SOURCE_SEED"
SOURCE_DIR="$SOURCE_ROOT/workers/$SOURCE_TASK/snapshots"
SOURCE_FILE="promoted_ucv_cross_entropy_seed_${{SOURCE_SEED}}_time_36h.pkl"
TASK_NAME="task_$(printf '%03d' "$TASK_INDEX")_{task_label}_seed_$SOURCE_SEED"
REMOTE_TASK="$BUCKET_ROOT/$RUN_ID/workers/$TASK_NAME"
mkdir -p "$SOURCE_DIR" "$OUTPUT_ROOT/workers"
gcloud storage cp \\
  "$BUCKET_ROOT/$EXP29_RUN_ID/workers/$SOURCE_TASK/snapshots/$SOURCE_FILE" \\
  "$SOURCE_DIR/$SOURCE_FILE"
upload_worker() {{
  if [[ -d "$OUTPUT_ROOT/workers/$TASK_NAME" ]]; then
    gcloud storage rsync --recursive "$OUTPUT_ROOT/workers/$TASK_NAME" "$REMOTE_TASK" || true
  fi
}}
trap upload_worker EXIT
python -m {module} worker --task-index "$TASK_INDEX" \\
  --output-root "$OUTPUT_ROOT" --source-root "$SOURCE_ROOT" --no-resume
""".strip()
    elif args.kind == "aggregate":
        comparison = ""
        if args.experiment_id == 32:
            comparison = """
mkdir -p "$WORK_ROOT/experiment_31_reference/analysis"
gcloud storage rsync --recursive \\
  "$BUCKET_ROOT/$EXP31_RUN_ID/analysis" \\
  "$WORK_ROOT/experiment_31_reference/analysis"
EXTRA_ARGS=(--experiment-31-root "$WORK_ROOT/experiment_31_reference")
""".strip()
        else:
            comparison = "EXTRA_ARGS=()"
        action = f"""
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
{comparison}
python -m {module} aggregate --output-root "$OUTPUT_ROOT" "${{EXTRA_ARGS[@]}}"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
""".strip()
    else:
        raise ValueError(args.kind)
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{action}\n"


def build_job(args) -> dict:
    task_count = 5 if args.kind == "train" else 1
    parallelism = min(task_count, args.parallelism)
    if args.kind == "train":
        max_duration = "10800s" if args.experiment_id == 31 else "21600s"
        cpu, memory, machine, disk = 4000, 15000, "n2-standard-4", 30
    elif args.kind == "controller":
        max_duration, cpu, memory, machine, disk = "43200s", 1000, 1500, "e2-small", 20
    else:
        max_duration, cpu, memory, machine, disk = "7200s", 4000, 15000, "n2-standard-4", 30
    return {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"script": {"text": _script(args)}}],
                "computeResource": {"cpuMilli": cpu, "memoryMib": memory},
                "maxRetryCount": 2 if args.kind == "controller" else 0,
                "maxRunDuration": max_duration,
            },
            "taskCount": task_count,
            "parallelism": parallelism,
            "taskCountPerNode": 1,
        }],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [{"policy": {
                "machineType": machine,
                "provisioningModel": "STANDARD",
                "bootDisk": {"sizeGb": disk, "type": "pd-balanced"},
            }}],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": f"live-resolving-{args.experiment_id}", "stage": args.kind},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=int, choices=(31, 32), required=True)
    parser.add_argument(
        "--kind", choices=("controller", "smoke", "train", "aggregate"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--experiment-29-run-id", required=True)
    parser.add_argument("--experiment-31-run-id", default="unused")
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--parallelism", type=int, default=5)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--controller-action",
        choices=("orchestrate", "orchestrate-resume"),
        default="orchestrate",
    )
    args = parser.parse_args()
    if args.experiment_id == 32 and args.experiment_31_run_id == "unused":
        parser.error("Experiment 32 requires --experiment-31-run-id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(build_job(args), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
