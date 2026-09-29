#!/usr/bin/env python3
"""One evaluation-only Batch VM: smoke gate, exact analysis, final upload."""

import argparse
import json
from pathlib import Path
import shlex

MODULE = "experiments.leduc_poker.grouped_ucv_temporal_head_to_head.run"
REPO_URL = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"


def build_job(args):
    q = shlex.quote
    script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/exp45-mpl XDG_CACHE_HOME=/tmp/exp45-cache
export UV_CACHE_DIR=/tmp/exp45-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/exp45-python
WORK=/workspace/exp45-evaluation
mkdir -p "$WORK/source" "$WORK/output" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
REMOTE={q(args.bucket.rstrip('/') + '/' + args.run_id)}
SOURCE={q(args.bucket.rstrip('/') + '/' + args.source_run_id)}
upload() {{
  code=$?
  trap - EXIT
  set +e
  gcloud storage rsync --recursive "$WORK/output" "$REMOTE"
  upload_code=$?
  if [[ "$code" == 0 && "$upload_code" != 0 ]]; then code=$upload_code; fi
  exit "$code"
}}
trap upload EXIT
exec > >(tee -a "$WORK/output/batch.log") 2>&1
if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv
git clone --filter=blob:none {q(REPO_URL)} "$WORK/repository"
git -C "$WORK/repository" checkout --detach {q(args.repo_ref)}
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/exp45-uv UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/exp45-uv:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
# Only evaluation dependencies; no TensorFlow or neural training environment.
python -m pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4 open_spiel==1.6.3 \\
  'torch==2.7.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps --no-build-isolation -e "$WORK/repository"
python -m pip check
cd "$WORK/repository"
python -m {MODULE} fetch --source-root "$WORK/source" --source-uri "$SOURCE"
python -m {MODULE} smoke --source-root "$WORK/source" --output-root "$WORK/output/smoke"
python -m {MODULE} run --source-root "$WORK/source" --output-root "$WORK/output"
"""
    return {
        "taskGroups": [{"taskSpec": {
            "runnables": [{"script": {"text": script}}],
            "computeResource": {"cpuMilli": 4000, "memoryMib": 15000},
            "maxRunDuration": "14400s", "maxRetryCount": 0,
        }, "taskCount": 1, "parallelism": 1, "taskCountPerNode": 1}],
        "allocationPolicy": {
            "serviceAccount": {"email": args.service_account},
            "instances": [{"policy": {
                "machineType": "n2-standard-4", "provisioningModel": "STANDARD",
                "bootDisk": {"sizeGb": 50, "type": "pd-balanced"},
            }}],
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "exp45", "stage": "evaluate"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket", "run-id", "source-run-id", "repo-ref", "service-account", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
