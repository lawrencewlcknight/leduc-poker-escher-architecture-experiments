#!/usr/bin/env python3
"""Bounded remote jobs with explicit, identity-checked continuation."""
import argparse
import json
from pathlib import Path
import math
import shlex

MODULE="experiments.leduc_poker.critic_update_budget.run"
REPO="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
LAUNCHER="gcp/run_critic_update_budget.sh"



def q(x):
    return shlex.quote(str(x))


def script(a):
    text=f'''#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLCONFIGDIR=/tmp/exp51-mpl XDG_CACHE_HOME=/tmp/exp51-cache
WORK=/workspace/exp51
REPOSITORY="$WORK/repository"
OUTPUT="$WORK/output"
BUCKET_ROOT={q(a.bucket)}
RUN_ID={q(a.run_id)}
mkdir -p "$WORK" "$OUTPUT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
if command -v sudo >/dev/null; then SUDO=sudo; else SUDO=; fi
$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
git clone --filter=blob:none {q(REPO)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {q(a.repo_ref)}
cd "$REPOSITORY"
'''
    if a.kind=="controller":
        return text+f'''
export PROJECT_ID={q(a.project)} REGION={q(a.region)} BUCKET={q(a.bucket)}
export SA_EMAIL={q(a.service_account)} REPO_REF={q(a.repo_ref)} PARALLELISM={a.parallelism}
export RUN_ID EXP51_REMOTE_CONTROLLER=1
export ATTEMPT={q(a.attempt)}
exec bash {LAUNCHER} orchestrate
'''
    text+='''
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/tmp/uv-bin UV_NO_MODIFY_PATH=1 sh
export PATH="/tmp/uv-bin:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed /tmp/exp51-venv
source /tmp/exp51-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check
'''
    if a.kind=="train":
        text+=r'''
INDEX="${BATCH_TASK_INDEX:?Missing task index}"
TASK="$(python -m experiments.cloud_worker_identity --experiment 51 --task-index "$INDEX" --remote-root "$BUCKET_ROOT/$RUN_ID")"
LOCAL_WORKER="$OUTPUT/workers/$TASK"
export EXP51_REMOTE_WORKER="$BUCKET_ROOT/$RUN_ID/workers/$TASK"
mkdir -p "$LOCAL_WORKER"
if gcloud storage ls "$EXP51_REMOTE_WORKER/manifest.json" >/dev/null 2>&1; then
  gcloud storage rsync --recursive --exclude '.*(\.pt|\.tmp)$' "$EXP51_REMOTE_WORKER" "$LOCAL_WORKER"
  if [[ -f "$LOCAL_WORKER/latest.json" && ! -f "$LOCAL_WORKER/SUCCESS.json" ]]; then
    STATE_FILE="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["file"] in ("state_0.pt","state_1.pt"); print(d["file"])' "$LOCAL_WORKER/latest.json")"
    gcloud storage cp "$EXP51_REMOTE_WORKER/$STATE_FILE" "$LOCAL_WORKER/$STATE_FILE"
  fi
fi
upload() {
  rc=$?
  trap - EXIT
  if ! gcloud storage rsync --recursive --exclude '.*(\.pt|\.tmp)$' "$LOCAL_WORKER" "$EXP51_REMOTE_WORKER"; then
    [[ "$rc" != 0 ]] || rc=1
  fi
  exit "$rc"
}
trap upload EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
'''
        return text+f'python -m {MODULE} worker --task-index "$INDEX" --output-root "$OUTPUT"\n'
    if a.kind=="pilot":
        if not a.source_uri.startswith("gs://") or len(a.source_sha256)!=64:
            raise ValueError("Mature pilot requires source URI and SHA256")
        return text+f'''
gcloud storage cp {q(a.source_uri)} "$WORK/source.pt"
python -m {MODULE} pilot --source-state "$WORK/source.pt" --source-sha256 {q(a.source_sha256)} --output-root "$OUTPUT/pilot"
gcloud storage rsync --recursive "$OUTPUT/pilot" "$BUCKET_ROOT/$RUN_ID/pilot"
'''
    if a.kind=="aggregate":
        return text+rf'''
mkdir -p "$OUTPUT/workers"
gcloud storage rsync --recursive --exclude '.*(\.pt|\.tmp)$' "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
'''
    return text+rf'''
finish_smoke() {{
  rc=$?
  trap - EXIT
  if ! gcloud storage rsync --recursive --exclude '.*(\.pt|\.tmp)$' "$OUTPUT" "$BUCKET_ROOT/$RUN_ID/smoke"; then
    [[ "$rc" != 0 ]] || rc=1
  fi
  exit "$rc"
}}
trap finish_smoke EXIT
python -m unittest discover -s tests -p test_critic_update_budget.py
python -m unittest discover -s tests -p test_cloud_worker_identity.py
python -m unittest discover -s tests -p test_optional_pilot_launchers.py
python -m {MODULE} smoke --output-root "$OUTPUT"
'''


def build_job(a):
    controller=a.kind=="controller"
    count=12 if a.kind=="train" else 1
    timeout={"smoke":3,"pilot":6,"train":72,"aggregate":6,"controller":72*math.ceil(12/a.parallelism)+12}[a.kind]
    return {"taskGroups":[{"taskCount":count,"parallelism":min(count,a.parallelism),"taskCountPerNode":1,
        "taskSpec":{"runnables":[{"script":{"text":script(a)}}],
            "computeResource":{"cpuMilli":1000 if controller else 8000,"memoryMib":1500 if controller else 30000},
            "maxRetryCount":0,"maxRunDuration":f"{timeout*3600}s"}}],
        "allocationPolicy":{"serviceAccount":{"email":a.service_account},"instances":[{"policy":{
            "machineType":"e2-small" if controller else "n2-standard-8","provisioningModel":"STANDARD",
            "bootDisk":{"type":"pd-balanced","sizeGb":30 if controller else 150}}}]},
        "logsPolicy":{"destination":"CLOUD_LOGGING"},"labels":{"experiment":"exp51-critic-budget","stage":a.kind}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("controller","smoke","pilot","train","aggregate"),required=True)
    for name in ("project","region","bucket","run-id","repo-ref","service-account"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--parallelism",type=int,default=8)
    p.add_argument("--source-uri",default="")
    p.add_argument("--source-sha256",default="")
    p.add_argument("--attempt",default="initial")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not 3<=a.parallelism<=12:
        p.error("parallelism must be 3 through 12 (bounded controller duration)")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(build_job(a),indent=2)+"\n")
