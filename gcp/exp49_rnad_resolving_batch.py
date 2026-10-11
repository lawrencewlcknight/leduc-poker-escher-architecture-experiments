#!/usr/bin/env python3
"""One CPU-only Batch VM for the Experiment 49 pilot or frozen-policy audit."""

import argparse
import json
from pathlib import Path
import shlex

MODULE = "experiments.leduc_poker.rnad_ucv_resolving.run"
REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"


def build_job(args):
    if args.mode not in ("pilot", "full") or not 1 <= args.workers <= 4:
        raise ValueError("Invalid mode or worker count")
    if args.mode == "full" and not args.pilot_run_id:
        raise ValueError("Full run requires a completed pilot")
    q = shlex.quote
    remote = args.bucket.rstrip("/") + "/" + args.run_id
    pilot_fetch = ""
    pilot_arg = ""
    if args.mode == "full":
        pilot_fetch = (f"mkdir -p \"$WORK/pilot\"\n"
                       f"gcloud storage cp {q(args.bucket.rstrip('/') + '/' + args.pilot_run_id + '/SUCCESS.json')} "
                       f"\"$WORK/pilot/SUCCESS.json\"")
        pilot_arg = '--pilot-output "$WORK/pilot"'
    script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/exp49-mpl XDG_CACHE_HOME=/tmp/exp49-cache
export UV_CACHE_DIR=/tmp/exp49-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/exp49-python
WORK=/workspace/exp49
mkdir -p "$WORK/source" "$WORK/result" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
REMOTE={q(remote)}
upload() {{
  code=$?
  trap - EXIT
  set +e
  # Publish completion only after every other output is durably uploaded.
  gcloud storage rsync --recursive --exclude='^SUCCESS\\.json$' "$WORK/result" "$REMOTE"
  upload_code=$?
  gcloud storage cp "$WORK/batch.log" "$REMOTE/batch.log"
  log_code=$?
  if [[ "$code" == 0 && "$upload_code" != 0 ]]; then code=$upload_code; fi
  if [[ "$code" == 0 && "$log_code" != 0 ]]; then code=$log_code; fi
  if [[ "$code" == 0 ]]; then
    gcloud storage cp "$WORK/result/SUCCESS.json" "$REMOTE/SUCCESS.json"
    code=$?
  fi
  exit "$code"
}}
trap upload EXIT
exec > >(tee -a "$WORK/batch.log") 2>&1
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv
git clone --filter=blob:none {q(REPO_URL)} "$WORK/repository"
git -C "$WORK/repository" checkout --detach {q(args.repo_ref)}
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/exp49-uv UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/exp49-uv:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4 open_spiel==1.6.3 \\
  'torch==2.7.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps --no-build-isolation -e "$WORK/repository"
python -m pip check
cd "$WORK/repository"
python -m {MODULE} smoke --output-root "$WORK/smoke"
python -m {MODULE} fetch --source-root "$WORK/source" --source-uri {q(args.source_uri)}
{pilot_fetch}
python -m {MODULE} run --source-root "$WORK/source" --output-root "$WORK/result" \\
  --mode {q(args.mode)} --workers {args.workers} {pilot_arg}
cp "$WORK/smoke/smoke.json" "$WORK/result/analysis/smoke.json"
"""
    return {
        "taskGroups": [{"taskSpec": {
            "runnables": [{"script": {"text": script}}],
            "computeResource": {"cpuMilli": 16000, "memoryMib": 60000},
            "maxRunDuration": f"{args.max_hours * 3600}s", "maxRetryCount": 0,
        }, "taskCount": 1, "parallelism": 1, "taskCountPerNode": 1}],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [{"policy": {
                "machineType": "n2-standard-16", "provisioningModel": "STANDARD",
                "bootDisk": {"sizeGb": 50, "type": "pd-balanced"},
            }}],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "exp49", "stage": args.mode},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket", "run-id", "source-uri", "repo-ref", "service-account", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--pilot-run-id")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--max-hours", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.max_hours <= 48:
        parser.error("--max-hours must be between 1 and 48 (a safety ceiling, not an estimate)")
    Path(args.output).write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
