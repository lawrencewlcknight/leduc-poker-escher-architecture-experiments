#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
MODULE=experiments.leduc_poker.critic_update_budget.run
case "$ACTION" in
  smoke-local)
    exec "${PYTHON:-python3}" -m "$MODULE" smoke --output-root "${SMOKE_OUTPUT:-$(mktemp -d /tmp/exp51-smoke.XXXXXX)}" ;;
  pilot-local)
    : "${SOURCE_STATE:?Set SOURCE_STATE to an Experiment 35 full state}"
    : "${SOURCE_SHA256:?Set its verified SHA256}"
    exec "${PYTHON:-python3}" -m "$MODULE" pilot --source-state "$SOURCE_STATE" --source-sha256 "$SOURCE_SHA256" \
      --output-root "${PILOT_OUTPUT:-$(mktemp -d /tmp/exp51-pilot.XXXXXX)}" ;;
esac
: "${PROJECT_ID:?Set PROJECT_ID}" "${REGION:?Set REGION}" "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}" "${REPO_REF:?Set REPO_REF to the pushed commit}"
: "${RUN_ID:?Set a persistent RUN_ID, e.g. exp51-budget-YYYYMMDD}"
[[ "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ && ${#RUN_ID} -le 30 ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || { echo 'Use an exact 40-character commit SHA' >&2; exit 2; }
LAUNCHER_REF="${LAUNCHER_REF:-$REPO_REF}"
[[ "$LAUNCHER_REF" =~ ^[0-9a-f]{40}$ ]] || { echo 'LAUNCHER_REF must be an exact commit SHA' >&2; exit 2; }
[[ "$BUCKET" == gs://* ]] || { echo 'BUCKET must start with gs://' >&2; exit 2; }
BUCKET="${BUCKET%/}"
PARALLELISM="${PARALLELISM:-8}"
[[ "$PARALLELISM" =~ ^[0-9]+$ && "$PARALLELISM" -ge 3 && "$PARALLELISM" -le 12 ]] || exit 2
ATTEMPT="${ATTEMPT:-initial}"
[[ "$ATTEMPT" =~ ^[a-z0-9-]+$ && ${#ATTEMPT} -le 12 ]] || exit 2
TEMP_DIR="$(mktemp -d /tmp/exp51-jobs.XXXXXX)"
build() {
  python3 "$SCRIPT_DIR/critic_update_budget_batch.py" --kind "$1" --output "$TEMP_DIR/$1.json" \
    --project "$PROJECT_ID" --region "$REGION" --bucket "$BUCKET" --run-id "$RUN_ID" \
    --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" --launcher-ref "$LAUNCHER_REF" --parallelism "$PARALLELISM" --attempt "$ATTEMPT" \
    --source-uri "${SOURCE_URI:-}" --source-sha256 "${SOURCE_SHA256:-}"
}
submit() { gcloud batch jobs submit "$1" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/$2.json"; }
state() { gcloud batch jobs describe "$1" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'; }
ensure() {
  local status
  if ! status="$(state "$1" 2>/dev/null)"; then submit "$1" "$2"; fi
  while true; do
    status="$(state "$1")"
    case "$status" in SUCCEEDED) return 0 ;; FAILED|DELETION_IN_PROGRESS) return 1 ;; esac
    sleep 30
  done
}
case "$ACTION" in
  dry-run)
    for kind in controller smoke train aggregate; do build "$kind"; done
    echo "Specifications only: $TEMP_DIR" ;;
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:$RUN_ID" \
      --format='table(name.basename(),status.state,createTime)' ;;
  pilot-cloud)
    : "${SOURCE_URI:?Set the historical full-state GCS URI}" "${SOURCE_SHA256:?Set verified source hash}"
    build pilot; submit "$RUN_ID-pilot-$ATTEMPT" pilot ;;
  smoke-cloud)
    build smoke; submit "$RUN_ID-smoke-$ATTEMPT" smoke ;;
  run|resume)
    git cat-file -e "$REPO_REF:experiments/leduc_poker/critic_update_budget/run.py"
    if [[ "$ACTION" == resume && "$ATTEMPT" == initial ]]; then
      echo 'Resume requires a new ATTEMPT; keep RUN_ID and REPO_REF unchanged.' >&2; exit 2
    fi
    # Timing pilots are optional and may still be running. Cloud smoke remains
    # mandatory; the controller validates it directly against the pinned code.
    build controller; submit "$RUN_ID-controller-$ATTEMPT" controller ;;
  orchestrate)
    [[ "${EXP51_REMOTE_CONTROLLER:-}" == 1 ]] || exit 2
    for kind in smoke train aggregate; do build "$kind"; done
    ensure "$RUN_ID-smoke-$ATTEMPT" smoke
    gcloud storage cp "$BUCKET/$RUN_ID/smoke/SMOKE_SUCCESS.json" "$TEMP_DIR/smoke.json"
    python3 - "$TEMP_DIR/smoke.json" <<'PYGATE'
import json,os,sys
from experiments.cloud_source_identity import source_identity
s=json.load(open(sys.argv[1]))
assert s['status']=='complete', 'Cloud smoke did not complete'
assert s['source']==source_identity('tests/test_critic_update_budget.py', os.environ.get('STUDY_REPOSITORY')), 'Smoke code differs from confirmation code'
PYGATE
    ensure "$RUN_ID-train-$ATTEMPT" train
    ensure "$RUN_ID-aggregate-$ATTEMPT" aggregate ;;
  *) echo 'Usage: dry-run | smoke-local | pilot-local | pilot-cloud | smoke-cloud | run | resume | status' >&2; exit 2 ;;
esac
