#!/usr/bin/env python3
"""Bounded CPU Batch jobs for six saved-checkpoint noise interventions."""
import argparse
import json
import math
from pathlib import Path
import shlex

if __name__ == "__main__" and not __package__:
    from batch_bootstrap import bootstrap, python_environment
else:
    from gcp.batch_bootstrap import bootstrap, python_environment

MODULE = "experiments.leduc_poker.advantage_noise_intervention.run"
REPO = "https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
LAUNCHER = "gcp/run_advantage_noise_intervention.sh"


def q(value):
    return shlex.quote(str(value))


def script(a):
    launcher_ref = getattr(a, "launcher_ref", None) or a.repo_ref
    text = bootstrap(52, REPO, a.repo_ref, controller=a.kind == "controller", launcher_ref=launcher_ref)
    text += f'''
BUCKET_ROOT={q(a.bucket)}
SOURCE_BUCKET={q(a.source_bucket)}
SOURCE_RUN_ID={q(a.source_run_id)}
RUN_ID={q(a.run_id)}
'''
    if a.kind == "controller":
        return text + f'''
export PROJECT_ID={q(a.project)} REGION={q(a.region)} BUCKET={q(a.bucket)}
export SOURCE_BUCKET SOURCE_RUN_ID RUN_ID
export SA_EMAIL={q(a.service_account)} REPO_REF={q(a.repo_ref)}
export LAUNCHER_REF={q(launcher_ref)}
export PARALLELISM={a.parallelism} ATTEMPT={q(a.attempt)} EXP52_REMOTE_CONTROLLER=1
exec bash {LAUNCHER} orchestrate
'''
    text += python_environment()
    if a.kind == "aggregate":
        return text + f'''
gcloud storage rsync --recursive "$BUCKET_ROOT/$RUN_ID/workers" "$OUTPUT/workers"
python -m {MODULE} aggregate --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT/analysis" "$BUCKET_ROOT/$RUN_ID/analysis"
'''
    if a.kind == "smoke":
        return text + f'''
python -m unittest discover -s tests -p test_advantage_noise_intervention.py
python -m unittest discover -s tests -p test_cloud_worker_identity.py
python -m {MODULE} smoke --output-root "$OUTPUT"
gcloud storage rsync --recursive "$OUTPUT" "$BUCKET_ROOT/$RUN_ID/smoke"
'''
    index = '0' if a.kind == "pilot" else '${BATCH_TASK_INDEX:?Missing task index}'
    action = "pilot" if a.kind == "pilot" else "worker"
    remote = "$BUCKET_ROOT/$RUN_ID/pilot" if a.kind == "pilot" else "$BUCKET_ROOT/$RUN_ID"
    return text + f'''
INDEX="{index}"
REMOTE="{remote}"
TASK="$(python -m experiments.cloud_worker_identity --experiment 52 --task-index "$INDEX" --remote-root "$REMOTE")"
export EXP52_REMOTE_WORKER="$REMOTE/workers/$TASK"
LOCAL_WORKER="$OUTPUT/workers/$TASK"
mkdir -p "$LOCAL_WORKER"
python -m {MODULE} fetch-source --bucket "$SOURCE_BUCKET" --source-run-id "$SOURCE_RUN_ID" \\
  --task-index "$INDEX" --directory "$SOURCE"
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
    p.add_argument("--launcher-ref", default=None, help="Pinned infrastructure revision; defaults to repo-ref")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.parallelism <= 6:
        p.error("parallelism must be 1 through 6")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_job(args), indent=2)+"\n")
