#!/usr/bin/env python3
"""Build Google Cloud Batch jobs for Experiments 36--39."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.leduc_poker.policy_post_training.config import (
    METHODS,
    PRODUCTION_SEEDS,
)


REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
MODULE = "experiments.leduc_poker.policy_post_training.run"
TASK_COUNT = len(PRODUCTION_SEEDS)


def _q(value):
    return shlex.quote(str(value))


def _bootstrap(args):
    return f"""
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES=""
export MPLCONFIGDIR=/tmp/matplotlib
export XDG_CACHE_HOME=/tmp/cache
export XDG_DATA_HOME=/tmp/data
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
UV_INSTALL_DIR=/tmp/uv-bin
RETRY_ATTEMPT="${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
WORK_ROOT="/workspace/policy-post-training-{args.method}-$RETRY_ATTEMPT"
REPOSITORY="$WORK_ROOT/repository"
OUTPUT_ROOT="$WORK_ROOT/output"
SOURCE_ROOT="$WORK_ROOT/experiment_29_source"
VENV_ROOT="/tmp/policy-post-training-{args.method}-venv-$RETRY_ATTEMPT"
REPO_URL={_q(args.repo_url)}
REPO_REF={_q(args.repo_ref)}
BUCKET_ROOT={_q(args.bucket_root.rstrip('/'))}
RUN_ID={_q(args.run_id)}
EXP29_RUN_ID={_q(args.experiment_29_run_id)}
METHOD={_q(args.method)}

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
mkdir -p "$WORK_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" \
  "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_INSTALL_DIR"
git clone --filter=blob:none "$REPO_URL" "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach "$REPO_REF"
curl -LsSf https://astral.sh/uv/install.sh | \
  env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
export PATH="$UV_INSTALL_DIR:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r "$REPOSITORY/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation -e "$REPOSITORY"
python -m pip check
cd "$REPOSITORY"
""".strip()


def _controller_script(args):
    return f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
WORK_ROOT="/workspace/post-training-controller-${{BATCH_TASK_RETRY_ATTEMPT:-0}}"
REPOSITORY="$WORK_ROOT/repository"
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git ca-certificates python3
mkdir -p "$WORK_ROOT"
git clone --filter=blob:none {_q(args.repo_url)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {_q(args.repo_ref)}
cd "$REPOSITORY"
export PROJECT_ID={_q(args.project_id)}
export REGION={_q(args.region)}
export BUCKET={_q(args.bucket_root.rstrip('/'))}
export SA_EMAIL={_q(args.service_account)}
export REPO_REF={_q(args.repo_ref)}
export EXP29_RUN_ID={_q(args.experiment_29_run_id)}
export RUN_ID={_q(args.run_id)}
export PARALLELISM={_q(args.parallelism)}
export POST_TRAINING_REMOTE_CONTROLLER=1
exec bash gcp/run_policy_post_training.sh {_q(args.method)} {_q(args.controller_action)}
"""


def _script(args):
    if args.kind == "controller":
        return _controller_script(args)
    bootstrap = _bootstrap(args)
    if args.kind == "smoke":
        action = f"""
python -m {MODULE} --method "$METHOD" smoke --output-root "$OUTPUT_ROOT" --no-resume
gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID/smoke"
""".strip()
    elif args.kind == "train":
        action = f"""
TASK_INDEX="${{BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}}"
TASK_METADATA="$(python - "$TASK_INDEX" <<'PY' | sed -n 's/^POST_TRAINING_TASK //p' | tail -n 1
import sys
from experiments.leduc_poker.policy_post_training.config import PRODUCTION_SEEDS
index = int(sys.argv[1])
seed = PRODUCTION_SEEDS[index]
print("POST_TRAINING_TASK", seed, f"task_{{index:03d}}_promoted_ucv_cross_entropy_seed_{{seed}}")
PY
)"
read -r SOURCE_SEED SOURCE_TASK EXTRA_METADATA <<< "$TASK_METADATA"
EXPECTED_SOURCE="task_$(printf '%03d' "$TASK_INDEX")_promoted_ucv_cross_entropy_seed_$SOURCE_SEED"
if [[ ! "$SOURCE_SEED" =~ ^[0-9]+$ || "$SOURCE_TASK" != "$EXPECTED_SOURCE" || -n "$EXTRA_METADATA" ]]; then
  echo "Invalid post-training task metadata: $TASK_METADATA" >&2
  exit 2
fi
TASK_NAME="task_$(printf '%03d' "$TASK_INDEX")_${{METHOD}}_seed_$SOURCE_SEED"
REMOTE_TASK="$BUCKET_ROOT/$RUN_ID/workers/$TASK_NAME"
SOURCE_DIR="$SOURCE_ROOT/workers/$SOURCE_TASK/training_states"
mkdir -p "$OUTPUT_ROOT/workers"
if gcloud storage ls "$REMOTE_TASK/SUCCESS.json" >/dev/null 2>&1; then
  mkdir -p "$OUTPUT_ROOT/workers/$TASK_NAME"
  gcloud storage rsync --recursive "$REMOTE_TASK" "$OUTPUT_ROOT/workers/$TASK_NAME"
else
  mkdir -p "$SOURCE_DIR"
  SOURCE_FILE="promoted_ucv_cross_entropy_seed_${{SOURCE_SEED}}_time_36h.pt"
  gcloud storage cp \
    "$BUCKET_ROOT/$EXP29_RUN_ID/workers/$SOURCE_TASK/training_states/$SOURCE_FILE" \
    "$SOURCE_DIR/$SOURCE_FILE"
fi
upload_worker() {{
  if [[ -d "$OUTPUT_ROOT/workers/$TASK_NAME" ]]; then
    gcloud storage rsync --recursive "$OUTPUT_ROOT/workers/$TASK_NAME" "$REMOTE_TASK" || true
  fi
}}
trap upload_worker EXIT
python -m {MODULE} --method "$METHOD" worker \
  --task-index "$TASK_INDEX" --output-root "$OUTPUT_ROOT" --source-root "$SOURCE_ROOT"
""".strip()
    elif args.kind == "aggregate":
        action = f"""
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive --exclude='(^|/)policies(/|$)' \
  "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
python -m {MODULE} --method "$METHOD" aggregate --output-root "$OUTPUT_ROOT"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
""".strip()
    else:
        raise ValueError(args.kind)
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{action}\n"


def build_job(args):
    task_count = TASK_COUNT if args.kind == "train" else 1
    parallelism = min(task_count, args.parallelism)
    if args.kind == "train":
        duration, cpu, memory, machine, disk = "43200s", 4000, 15000, "n2-standard-4", 50
    elif args.kind == "smoke":
        duration, cpu, memory, machine, disk = "7200s", 4000, 15000, "n2-standard-4", 50
    elif args.kind == "controller":
        duration, cpu, memory, machine, disk = "172800s", 1000, 1500, "e2-small", 30
    else:
        duration, cpu, memory, machine, disk = "10800s", 4000, 15000, "n2-standard-4", 50
    return {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"script": {"text": _script(args)}}],
                "computeResource": {"cpuMilli": cpu, "memoryMib": memory},
                "maxRetryCount": 2 if args.kind == "controller" else (1 if args.kind == "train" else 0),
                "maxRunDuration": duration,
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
        "labels": {
            "experiment": f"exp{METHODS[args.method]['experiment_id']}",
            "stage": args.kind,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("controller", "smoke", "train", "aggregate"), required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--repo-ref", required=True)
    parser.add_argument("--experiment-29-run-id", required=True)
    parser.add_argument("--repo-url", default=REPO_URL)
    parser.add_argument("--parallelism", type=int, default=TASK_COUNT)
    parser.add_argument("--project-id", default="")
    parser.add_argument("--region", default="")
    parser.add_argument("--controller-action", choices=("orchestrate", "orchestrate-resume"), default="orchestrate")
    args = parser.parse_args()
    if args.parallelism < 1 or args.parallelism > TASK_COUNT:
        parser.error(f"--parallelism must be between 1 and {TASK_COUNT}")
    if args.kind == "controller" and (not args.project_id or not args.region):
        parser.error("controller jobs require --project-id and --region")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(build_job(args), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
