#!/usr/bin/env python3
"""Bounded CPU Batch jobs for six saved-checkpoint noise interventions."""
import argparse
import json
import math
from pathlib import Path
import shlex

MODULE = "experiments.leduc_poker.advantage_noise_intervention.run"
REPO = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
LAUNCHER = "gcp/run_advantage_noise_intervention.sh"


def q(value):
    return shlex.quote(str(value))


def script(a):
    text = f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/exp52-mpl XDG_CACHE_HOME=/tmp/exp52-cache
WORK=/workspace/exp52
REPOSITORY="$WORK/repository"
OUTPUT="$WORK/output"
SOURCE="$WORK/source"
BUCKET_ROOT={q(a.bucket)}
SOURCE_BUCKET={q(a.source_bucket)}
SOURCE_RUN_ID={q(a.source_run_id)}
RUN_ID={q(a.run_id)}
mkdir -p "$WORK" "$OUTPUT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if command -v sudo >/dev/null; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
git clone --filter=blob:none {q(REPO)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {q(a.repo_ref)}
cd "$REPOSITORY"
'''
    if a.kind == "controller":
        return text + f'''
export PROJECT_ID={q(a.project)} REGION={q(a.region)} BUCKET={q(a.bucket)}
export SOURCE_BUCKET SOURCE_RUN_ID RUN_ID
export SA_EMAIL={q(a.service_account)} REPO_REF={q(a.repo_ref)}
export PARALLELISM={a.parallelism} ATTEMPT={q(a.attempt)} EXP52_REMOTE_CONTROLLER=1
exec bash {LAUNCHER} orchestrate
'''
    text += '''
export UV_CACHE_DIR=/tmp/exp52-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/exp52-python
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/exp52-uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/exp52-uv-bin:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed /tmp/exp52-venv
source /tmp/exp52-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check
'''
    if a.kind == "aggregate":
        return text + f'''
gcloud storage rsync --recursive "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
'''
    if a.kind == "smoke":
        return text + f'''
python -m unittest discover -s tests -p test_advantage_noise_intervention.py
python -m {MODULE} smoke --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT" "$BUCKET_ROOT/$RUN_ID/smoke"
'''
    index = '0' if a.kind == "pilot" else '${BATCH_TASK_INDEX:?Missing task index}'
    action = "pilot" if a.kind == "pilot" else "worker"
    remote = "$BUCKET_ROOT/$RUN_ID/pilot" if a.kind == "pilot" else "$BUCKET_ROOT/$RUN_ID"
    return text + f'''
INDEX="{index}"
REMOTE="{remote}"
python -m {MODULE} fetch-source --bucket "$SOURCE_BUCKET" --source-run-id "$SOURCE_RUN_ID" \\
  --task-index "$INDEX" --directory "$SOURCE"
TASK="$(python -c 'from experiments.leduc_poker.advantage_noise_intervention.config import tasks; import sys; s,c=tasks()[int(sys.argv[1])]; print(f"seed_{{s}}_{{c}}")' "$INDEX")"
export EXP52_REMOTE_WORKER="$REMOTE/workers/$TASK"
LOCAL_WORKER="$OUTPUT/workers/$TASK"
mkdir -p "$LOCAL_WORKER"
if gcloud storage ls "$EXP52_REMOTE_WORKER/manifest.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive "$EXP52_REMOTE_WORKER" "$LOCAL_WORKER"
fi
finish() {{
  rc=$?
  trap - EXIT
  if ! python - "$LOCAL_WORKER" "$OUTPUT" "$REMOTE" <<'PYUPLOAD'
from pathlib import Path
import subprocess,sys
from experiments.leduc_poker.advantage_noise_intervention.worker import upload
upload(Path(sys.argv[1]))
marker=Path(sys.argv[2])/'PILOT_SUCCESS.json'
if marker.exists():
    subprocess.run(['gcloud','storage','cp',str(marker),sys.argv[3]+'/PILOT_SUCCESS.json'],check=True)
PYUPLOAD
  then
    [[ "$rc" != 0 ]] || rc=1
  fi
  exit "$rc"
}}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python -m {MODULE} {action} --task-index "$INDEX" --source-worker "$SOURCE" --output-root "$OUTPUT"
'''


def build_job(a):
    controller = a.kind == "controller"
    count = 6 if a.kind == "worker" else 1
    hours = {"smoke": 3, "pilot": 6, "worker": 12, "aggregate": 3,
             "controller": 12*math.ceil(6/a.parallelism)+8}[a.kind]
    return {"taskGroups": [{"taskCount": count, "parallelism": min(count,a.parallelism), "taskCountPerNode": 1,
        "taskSpec": {"runnables": [{"script": {"text": script(a)}}],
            "computeResource": {"cpuMilli": 1000 if controller else 8000, "memoryMib": 1500 if controller else 30000},
            "maxRetryCount": 0, "maxRunDuration": f"{hours*3600}s"}}],
        "allocationPolicy": {"serviceAccount": {"email": a.service_account}, "instances": [{"policy": {
            "machineType": "e2-small" if controller else "n2-standard-8", "provisioningModel": "STANDARD",
            "bootDisk": {"type": "pd-balanced", "sizeGb": 30 if controller else 150}}}]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"}, "labels": {"experiment": "exp52-noise", "stage": a.kind}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("controller","smoke","pilot","worker","aggregate"), required=True)
    for name in ("project","region","bucket","source-bucket","run-id","service-account","repo-ref"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--source-run-id", default="exp35-confirm-20260916-011231")
    p.add_argument("--parallelism", type=int, default=3)
    p.add_argument("--attempt", default="initial")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.parallelism <= 6:
        p.error("parallelism must be 1 through 6")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2)+"\n")
