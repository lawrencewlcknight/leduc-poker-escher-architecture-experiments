#!/usr/bin/env python3
"""One bounded evaluation-only VM; no controller, child jobs or training."""
import argparse
import json
from pathlib import Path
import shlex

DEEP_REF = "1669e5af4cbc88c648626148fd9c395c2e5d4583"
MODULE = "experiments.leduc_poker.current_candidate_policy_league.run"


def build_job(args):
    q = shlex.quote
    script = f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/exp46-mpl XDG_CACHE_HOME=/tmp/exp46-cache
export UV_CACHE_DIR=/tmp/exp46-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/exp46-python
WORK=/workspace/exp46-league
mkdir -p "$WORK/source" "$WORK/output" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
REMOTE={q(args.bucket.rstrip('/') + '/' + args.run_id)}
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
git clone --filter=blob:none https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git "$WORK/repository"
git -C "$WORK/repository" checkout --detach {q(args.repo_ref)}
git clone --filter=blob:none https://github.com/lawrencewlcknight/leduc-poker-deep-cfr-experiments.git "$WORK/deep-cfr"
git -C "$WORK/deep-cfr" checkout --detach {DEEP_REF}
export DEEP_CFR_REPO="$WORK/deep-cfr"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/exp46-uv UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/exp46-uv:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.4 h5py==3.13.0 \\
  pandas==2.2.3 psutil==7.0.0 open_spiel==1.6.3 'torch==2.7.0+cpu' \\
  --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps --no-build-isolation -e "$WORK/repository"
python -m pip check
python -m pip freeze > "$WORK/output/environment.txt"
cd "$WORK/repository"
# Freeze metadata and check every comparator in a real seven-arm smoke first.
python -m {MODULE} prepare --source-root "$WORK/source" --smoke-only
python -m {MODULE} smoke --source-root "$WORK/source" --output-root "$WORK/output/smoke"
python -m {MODULE} prepare --source-root "$WORK/source"
python -m {MODULE} run --source-root "$WORK/source" --output-root "$WORK/output"
# Only provenance sidecars/manifests, never downloaded archives or reservoirs.
mkdir -p "$WORK/output/source_provenance"
cp "$WORK/source/manifest.json" "$WORK/source/smoke_manifest.json" "$WORK/output/source_provenance/"
cp "$WORK/source/objects/"*.origin.json "$WORK/output/source_provenance/"
'''
    return {
        "taskGroups": [{"taskSpec": {
            "runnables": [{"script": {"text": script}}],
            "computeResource": {"cpuMilli": 8000, "memoryMib": 30000},
            "maxRunDuration": "28800s", "maxRetryCount": 0,
        }, "taskCount": 1, "parallelism": 1, "taskCountPerNode": 1}],
        "allocationPolicy": {"serviceAccount": {"email": args.service_account},
            "instances": [{"policy": {"machineType": "n2-standard-8", "provisioningModel": "STANDARD",
                                      "bootDisk": {"sizeGb": 80, "type": "pd-balanced"}}}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
        "labels": {"experiment": "exp46", "stage": "evaluate"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bucket", "run-id", "repo-ref", "service-account", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build_job(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
