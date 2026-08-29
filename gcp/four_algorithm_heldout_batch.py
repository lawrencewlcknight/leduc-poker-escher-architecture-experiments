#!/usr/bin/env python3
"""Build a Google Cloud Batch JSON definition for the held-out benchmark."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path


ARCH_REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
DEEP_REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-deep-cfr-experiments.git"


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

ARCH_REPO_URL={_q(args.arch_repo_url)}
DEEP_REPO_URL={_q(args.deep_repo_url)}
ARCH_REPO_REF={_q(args.arch_repo_ref)}
DEEP_REPO_REF={_q(args.deep_repo_ref)}
BUCKET_ROOT={_q(args.bucket_root.rstrip('/'))}
RUN_ID={_q(args.run_id)}
WORK_ROOT=/workspace/heldout-benchmark
ARCH_REPO="$WORK_ROOT/architecture"
DEEP_REPO="$WORK_ROOT/deep-cfr"
OUTPUT_ROOT="$WORK_ROOT/output"

if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
mkdir -p "$WORK_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
git clone --filter=blob:none "$ARCH_REPO_URL" "$ARCH_REPO"
git -C "$ARCH_REPO" checkout --detach "$ARCH_REPO_REF"
git clone --filter=blob:none "$DEEP_REPO_URL" "$DEEP_REPO"
git -C "$DEEP_REPO" checkout --detach "$DEEP_REPO_REF"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed /tmp/heldout-benchmark-venv
source /tmp/heldout-benchmark-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r "$ARCH_REPO/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation -e "$ARCH_REPO"
python -m pip check
cd "$ARCH_REPO"
""".strip()


def _script(args) -> str:
    bootstrap = _bootstrap(args)
    if args.kind == "smoke":
        action = """
python -m experiments.leduc_poker.four_algorithm_heldout_benchmark.run smoke \
  --output-root "$OUTPUT_ROOT" \
  --deep-cfr-repo "$DEEP_REPO" \
  --no-resume
gcloud storage rsync --recursive "$OUTPUT_ROOT" "$BUCKET_ROOT/$RUN_ID/smoke"
""".strip()
    elif args.kind == "train":
        action = """
TASK_INDEX="${BATCH_TASK_INDEX:?Google Batch did not set BATCH_TASK_INDEX}"
TASK_NAME="$(python - "$TASK_INDEX" <<'PY'
import sys
from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import HELDOUT_SEEDS, task_schedule
index = int(sys.argv[1])
algorithm, seed = task_schedule(HELDOUT_SEEDS)[index]
print(f"task_{index:03d}_{algorithm}_seed_{seed}")
PY
)"
REMOTE_TASK="$BUCKET_ROOT/$RUN_ID/workers/$TASK_NAME"
mkdir -p "$OUTPUT_ROOT/workers"
if gcloud storage ls "$REMOTE_TASK/SUCCESS.json" >/dev/null 2>&1; then
  echo "Restoring already-complete task: $REMOTE_TASK"
  mkdir -p "$OUTPUT_ROOT/workers/$TASK_NAME"
  gcloud storage rsync --recursive "$REMOTE_TASK" "$OUTPUT_ROOT/workers/$TASK_NAME"
fi
upload_worker() {
  if [[ -d "$OUTPUT_ROOT/workers/$TASK_NAME" ]]; then
    gcloud storage rsync --recursive \
      "$OUTPUT_ROOT/workers/$TASK_NAME" "$REMOTE_TASK" || true
  fi
}
trap upload_worker EXIT
python -m experiments.leduc_poker.four_algorithm_heldout_benchmark.run worker \
  --task-index "$TASK_INDEX" \
  --output-root "$OUTPUT_ROOT" \
  --deep-cfr-repo "$DEEP_REPO"
""".strip()
    elif args.kind == "aggregate":
        action = """
mkdir -p "$OUTPUT_ROOT/workers"
gcloud storage rsync --recursive "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT_ROOT/workers"
python -m experiments.leduc_poker.four_algorithm_heldout_benchmark.run aggregate \
  --output-root "$OUTPUT_ROOT"
gcloud storage rsync --recursive "$OUTPUT_ROOT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
""".strip()
    else:  # pragma: no cover - argparse guards this
        raise ValueError(args.kind)
    return f"#!/usr/bin/env bash\nset -Eeuo pipefail\n{bootstrap}\n{action}\n"


def build_job(args) -> dict:
    task_count = 32 if args.kind == "train" else 1
    parallelism = min(task_count, args.parallelism)
    if args.kind == "train":
        max_duration = "64800s"  # 18 hours including bootstrap and endpoint overshoot.
        cpu_milli = 8000
        memory_mib = 30000
    elif args.kind == "smoke":
        max_duration = "7200s"
        cpu_milli = 8000
        memory_mib = 30000
    else:
        max_duration = "7200s"
        cpu_milli = 4000
        memory_mib = 15000
    retries = args.max_retries
    return {
        "taskGroups": [
            {
                "taskSpec": {
                    "runnables": [{"script": {"text": _script(args)}}],
                    "computeResource": {
                        "cpuMilli": cpu_milli,
                        "memoryMib": memory_mib,
                    },
                    "maxRetryCount": retries,
                    "maxRunDuration": max_duration,
                },
                "taskCount": task_count,
                "parallelism": parallelism,
                # Do not let Batch co-locate two long training tasks on one VM.
                "taskCountPerNode": 1,
            }
        ],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [
                {
                    "policy": {
                        "machineType": (
                            "n2-standard-8" if args.kind != "aggregate" else "n2-standard-4"
                        ),
                        "provisioningModel": args.provisioning_model,
                        "bootDisk": {"sizeGb": 100, "type": "pd-balanced"},
                    }
                }
            ],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {
            "experiment": "four-alg-heldout",
            "stage": args.kind,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("smoke", "train", "aggregate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket-root", required=True)
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--arch-repo-ref", required=True)
    parser.add_argument("--deep-repo-ref", required=True)
    parser.add_argument("--arch-repo-url", default=ARCH_REPO_URL)
    parser.add_argument("--deep-repo-url", default=DEEP_REPO_URL)
    parser.add_argument("--parallelism", type=int, default=32)
    parser.add_argument("--provisioning-model", choices=("STANDARD", "SPOT"), default="STANDARD")
    parser.add_argument("--max-retries", type=int, default=0)
    args = parser.parse_args()
    if args.parallelism < 1 or args.parallelism > 32:
        parser.error("--parallelism must be between 1 and 32")
    if args.max_retries < 0 or args.max_retries > 10:
        parser.error("--max-retries must be between 0 and 10")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(build_job(args), handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
