"""Shared, task-isolated startup for the Experiment 50--52 Batch launchers.

Keep launcher revisions separate from scientific revisions: infrastructure fixes
must not invalidate saved workers' strict source/runtime fingerprints.
"""
import shlex


def bootstrap(experiment, repo, repo_ref, *, controller=False, launcher_ref=None):
    launcher_ref = launcher_ref or repo_ref
    checkout = launcher_ref if controller else repo_ref
    text = f'''#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Batch may assign several successive tasks to one VM. Never inherit its prior
# checkout, output, source checkpoint or Python environment (including retries).
WORK_BASE="${{BATCH_WORK_ROOT:-/workspace}}"
mkdir -p "$WORK_BASE"
WORK="$(mktemp -d "$WORK_BASE/exp{experiment}-task.XXXXXXXXXX")"
REPOSITORY="$WORK/repository"
OUTPUT="$WORK/output"
SOURCE="$WORK/source"
export MPLCONFIGDIR="$WORK/mpl" XDG_CACHE_HOME="$WORK/cache"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
printf 'Task workspace: %s (task %s)\\n' "$WORK" "${{BATCH_TASK_INDEX:-controller}}"
if command -v sudo >/dev/null; then SUDO=sudo; else SUDO=; fi
apt_retry() {{
  local attempt rc
  for attempt in 1 2 3 4 5; do
    if $SUDO apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3 "$@"; then
      return 0
    else
      rc=$?
    fi
    printf 'Package setup attempt %s/5 failed (exit %s): %s\\n' "$attempt" "$rc" "$*" >&2
    if [[ "$attempt" == 5 ]]; then return "$rc"; fi
    # Wait for unattended upgrades; never delete locks or kill the lock holder.
    sleep 30
  done
}}
apt_retry update
apt_retry install -y git curl ca-certificates python3 python3-venv python3-dev build-essential
git clone --filter=blob:none {shlex.quote(repo)} "$REPOSITORY"
git -C "$REPOSITORY" checkout --detach {shlex.quote(checkout)}
printf 'Checked out source: %s\\n' "$(git -C "$REPOSITORY" rev-parse HEAD)"
cd "$REPOSITORY"
'''
    if controller:
        text += 'export STUDY_REPOSITORY="$REPOSITORY"\n'
        if launcher_ref != repo_ref:
            # Read the *original* source when validating a resumed worker's
            # smoke receipt. Do not relax or rewrite any scientific manifest.
            text += f'''
git clone --shared "$REPOSITORY" "$WORK/study"
git -C "$WORK/study" checkout --detach {shlex.quote(repo_ref)}
export STUDY_REPOSITORY="$WORK/study"
'''
    return text


def python_environment():
    return '''
export UV_CACHE_DIR="$WORK/uv-cache" UV_PYTHON_INSTALL_DIR="$WORK/python"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/uv-bin" UV_NO_MODIFY_PATH=1 sh
export PATH="$WORK/uv-bin:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed "$WORK/venv"
source "$WORK/venv/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check
'''
