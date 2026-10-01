#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
MODULE=experiments.leduc_poker.cached_grouped_ucv_36h.run
if [[ "$ACTION" == smoke-local ]]; then
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-$(mktemp -d /tmp/exp48-smoke.XXXXXX)}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/exp48-matplotlib}"
  exec "${PYTHON:-python3}" -m "$MODULE" smoke --output-root "$SMOKE_OUTPUT"
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set the Leduc output BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to a pushed Experiment 48 commit}"
RUN_ID="${RUN_ID:-exp48-cache36h-$(date -u '+%Y%m%d-%H%M%S')}"
PARALLELISM="${PARALLELISM:-5}"
[[ ${#RUN_ID} -le 38 && "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ "$PARALLELISM" =~ ^[1-5]$ ]] || { echo 'PARALLELISM must be 1-5' >&2; exit 2; }
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
TEMP_DIR="$(mktemp -d /tmp/exp48-jobs.XXXXXX)"
state() { gcloud batch jobs describe "$1" --project "$PROJECT_ID" --location "$REGION" --format='value(status.state)'; }
build() {
  python3 "$SCRIPT_DIR/cached_grouped_ucv_36h_batch.py" --kind "$1" --output "$TEMP_DIR/$1.json" \
    --project "$PROJECT_ID" --region "$REGION" --bucket "$BUCKET_ROOT" --run-id "$RUN_ID" \
    --service-account "$SA_EMAIL" --repo-ref "$REPO_REF" --parallelism "$PARALLELISM"
}
submit() { gcloud batch jobs submit "$1" --project "$PROJECT_ID" --location "$REGION" --config "$TEMP_DIR/$2.json"; }
wait_job() {
  local status
  while true; do
    status="$(state "$1")" || return 2
    echo "$1: $status"
    case "$status" in SUCCEEDED) return 0 ;; FAILED|DELETION_IN_PROGRESS) return 1 ;; esac
    sleep 30
  done
}
ensure() {
  local status
  if status="$(state "$1" 2>/dev/null)"; then
    [[ "$status" == SUCCEEDED ]] && return 0
    [[ "$status" == FAILED || "$status" == DELETION_IN_PROGRESS ]] && return 1
  else
    submit "$1" "$2"
  fi
  wait_job "$1"
}
preflight() {
  if ! git cat-file -e "$REPO_REF^{commit}" 2>/dev/null || \
     ! git cat-file -e "$REPO_REF:experiments/leduc_poker/cached_grouped_ucv_36h/run.py" 2>/dev/null; then
    echo 'REPO_REF does not contain Experiment 48. Pull the pushed code and reset REPO_REF.' >&2; return 2
  fi
  gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT_ID" --format='value(email)' >/dev/null
  # Explicitly check controller permissions rather than leaving a long idle controller.
  local roles self_role
  roles="$(gcloud projects get-iam-policy "$PROJECT_ID" --flatten='bindings[].members' \
    --filter="bindings.members:serviceAccount:$SA_EMAIL" --format='value(bindings.role)')"
  if ! [[ "$roles" == *roles/batch.jobsEditor* || "$roles" == *roles/batch.admin* || "$roles" == *roles/editor* || "$roles" == *roles/owner* ]]; then
    echo "Runner needs child-job permissions (normally roles/batch.jobsEditor). If using custom/inherited roles, verify them separately." >&2; return 2
  fi
  self_role="$(gcloud iam service-accounts get-iam-policy "$SA_EMAIL" --project "$PROJECT_ID" \
    --flatten='bindings[].members' --filter="bindings.members:serviceAccount:$SA_EMAIL" --format='value(bindings.role)')"
  if ! [[ "$roles" == *roles/iam.serviceAccountUser* || "$self_role" == *roles/iam.serviceAccountUser* || "$roles" == *roles/owner* ]]; then
    echo 'Runner needs permission to act as its child-job service account (roles/iam.serviceAccountUser).' >&2; return 2
  fi
}
case "$ACTION" in
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:$RUN_ID" \
      --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"; exit 0 ;;
  run|smoke-cloud|reaggregate) preflight ;;
  dry-run|orchestrate) ;;
  *) echo 'Usage: run | smoke-local | smoke-cloud | status | dry-run | reaggregate' >&2; exit 2 ;;
esac
for kind in controller smoke train aggregate; do build "$kind"; done
case "$ACTION" in
  dry-run) echo "Job specifications: $TEMP_DIR" ;;
  run)
    submit "$RUN_ID-controller" controller
    echo "Submitted. Smoke, training and analysis run in GCP; laptop may disconnect." ;;
  smoke-cloud) submit "$RUN_ID-smoke" smoke ;;
  reaggregate)
    [[ "$(state "$RUN_ID-train")" == SUCCEEDED ]] || { echo 'Training must have succeeded first' >&2; exit 2; }
    submit "$RUN_ID-reaggregate-$(date -u '+%H%M%S')" aggregate ;;
  orchestrate)
    [[ "${EXP48_REMOTE_CONTROLLER:-}" == 1 ]] || { echo 'Internal action' >&2; exit 2; }
    ensure "$RUN_ID-smoke" smoke
    gcloud storage ls "$BUCKET_ROOT/$RUN_ID/smoke/SMOKE_SUCCESS.json" >/dev/null
    ensure "$RUN_ID-train" train
    ensure "$RUN_ID-aggregate" aggregate ;;
esac
