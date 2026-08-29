#!/usr/bin/env bash
set -Eeuo pipefail

# One-command Google Cloud Batch orchestration for the four-algorithm held-out
# benchmark. The default `run` action submits a remote controller and returns.
# That controller enforces: cloud smoke -> 32 training tasks -> exact aggregate.

ACTION="${1:-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILDER="$SCRIPT_DIR/four_algorithm_heldout_batch.py"

if [[ "$ACTION" == "smoke-local" ]]; then
  DEEP_CFR_LOCAL_REPO="${DEEP_CFR_LOCAL_REPO:-$REPO_DIR/../../leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments}"
  SMOKE_OUTPUT="${SMOKE_OUTPUT:-/tmp/four-algorithm-heldout-smoke}"
  cd "$REPO_DIR"
  exec python3 -m experiments.leduc_poker.four_algorithm_heldout_benchmark.run smoke \
    --output-root "$SMOKE_OUTPUT" \
    --deep-cfr-repo "$DEEP_CFR_LOCAL_REPO" \
    --no-resume
fi

: "${PROJECT_ID:?Set PROJECT_ID to the Google Cloud project}"
: "${REGION:?Set REGION to the Google Cloud Batch region}"
: "${BUCKET:?Set BUCKET to a bucket name or gs:// bucket/prefix}"
: "${SA_EMAIL:?Set SA_EMAIL to the Batch service account}"
: "${ARCH_REPO_REF:?Set ARCH_REPO_REF to the pushed architecture-repository commit SHA}"
: "${DEEP_CFR_REPO_REF:?Set DEEP_CFR_REPO_REF to the frozen Deep CFR commit SHA}"

RUN_ID="${RUN_ID:-heldout-$(date -u '+%Y%m%d-%H%M%S')}"
if [[ ${#RUN_ID} -gt 40 || ! "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ ]]; then
  echo "RUN_ID must be 2-40 lowercase letters, digits or hyphens; start with a letter and end alphanumeric" >&2
  exit 2
fi
PARALLELISM="${PARALLELISM:-32}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-STANDARD}"
if [[ "$PROVISIONING_MODEL" != "STANDARD" && "$PROVISIONING_MODEL" != "SPOT" ]]; then
  echo "PROVISIONING_MODEL must be STANDARD or SPOT" >&2
  exit 2
fi
if [[ -n "${MAX_RETRIES:-}" ]]; then
  RETRIES="$MAX_RETRIES"
elif [[ "$PROVISIONING_MODEL" == "SPOT" ]]; then
  RETRIES=2
else
  RETRIES=0
fi
if [[ "$BUCKET" == gs://* ]]; then
  BUCKET_ROOT="${BUCKET%/}"
else
  BUCKET_ROOT="gs://${BUCKET%/}"
fi

SMOKE_JOB="${RUN_ID}-smoke"
TRAIN_JOB="${RUN_ID}-train"
AGGREGATE_JOB="${RUN_ID}-aggregate"
CONTROLLER_JOB="${RUN_ID}-controller"
CONTROLLER_ACTION="orchestrate"
if [[ "$ACTION" == "resume" ]]; then
  RESUME_TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  CONTROLLER_JOB="${RUN_ID}-controller-resume-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
elif [[ "$ACTION" == "orchestrate-resume" ]]; then
  RESUME_TAG="${RESUME_TAG:-$(date -u '+%H%M%S')}"
  TRAIN_JOB="${RUN_ID}-retry-${RESUME_TAG}"
  AGGREGATE_JOB="${RUN_ID}-reaggregate-${RESUME_TAG}"
  CONTROLLER_ACTION="orchestrate-resume"
fi
TEMP_DIR="$(mktemp -d /tmp/four-alg-heldout-batch.XXXXXX)"
trap 'rm -rf "$TEMP_DIR"' EXIT

build_json() {
  local kind="$1"
  local output="$2"
  local model="$PROVISIONING_MODEL"
  local retries="$RETRIES"
  if [[ "$kind" != "train" ]]; then
    model=STANDARD
    retries=0
  fi
  python3 "$BUILDER" \
    --kind "$kind" \
    --output "$output" \
    --run-id "$RUN_ID" \
    --bucket-root "$BUCKET_ROOT" \
    --service-account "$SA_EMAIL" \
    --arch-repo-ref "$ARCH_REPO_REF" \
    --deep-repo-ref "$DEEP_CFR_REPO_REF" \
    --parallelism "$PARALLELISM" \
    --provisioning-model "$model" \
    --max-retries "$retries" \
    --project-id "$PROJECT_ID" \
    --region "$REGION" \
    --controller-action "$CONTROLLER_ACTION"
}

submit_job() {
  local name="$1"
  local config="$2"
  echo "Submitting $name"
  gcloud batch jobs submit "$name" --project "$PROJECT_ID" --location "$REGION" --config "$config"
}

job_state() {
  gcloud batch jobs describe "$1" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --format='value(status.state)'
}

wait_for_job() {
  local name="$1"
  local state
  while true; do
    state="$(job_state "$name")"
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $name: $state"
    case "$state" in
      SUCCEEDED) return 0 ;;
      FAILED|DELETION_IN_PROGRESS) return 1 ;;
    esac
    sleep 30
  done
}

complete_or_retry() {
  local primary_name="$1"
  local retry_name="$2"
  local config="$3"
  local state
  if state="$(job_state "$primary_name" 2>/dev/null)"; then
    if [[ "$state" == "SUCCEEDED" ]]; then
      echo "$primary_name already succeeded"
      return 0
    fi
    if [[ "$state" != "FAILED" && "$state" != "DELETION_IN_PROGRESS" ]]; then
      if wait_for_job "$primary_name"; then
        return 0
      fi
    fi
    echo "$primary_name did not succeed; submitting $retry_name"
  else
    echo "$primary_name does not exist; submitting $retry_name"
  fi
  submit_job "$retry_name" "$config"
  wait_for_job "$retry_name"
}

ensure_job_succeeds() {
  local name="$1"
  local config="$2"
  local state
  if state="$(job_state "$name" 2>/dev/null)"; then
    if [[ "$state" == "SUCCEEDED" ]]; then
      echo "$name already succeeded"
      return 0
    fi
    if [[ "$state" == "FAILED" || "$state" == "DELETION_IN_PROGRESS" ]]; then
      echo "$name is terminal with state $state" >&2
      return 1
    fi
    wait_for_job "$name"
    return
  fi
  submit_job "$name" "$config"
  wait_for_job "$name"
}

build_json controller "$TEMP_DIR/controller.json"
build_json smoke "$TEMP_DIR/smoke.json"
build_json train "$TEMP_DIR/train.json"
build_json aggregate "$TEMP_DIR/aggregate.json"

case "$ACTION" in
  dry-run)
    cp "$TEMP_DIR/controller.json" "$REPO_DIR/heldout_controller_job.json"
    cp "$TEMP_DIR/smoke.json" "$REPO_DIR/heldout_smoke_job.json"
    cp "$TEMP_DIR/train.json" "$REPO_DIR/heldout_train_job.json"
    cp "$TEMP_DIR/aggregate.json" "$REPO_DIR/heldout_aggregate_job.json"
    echo "Wrote controller, smoke, train and aggregate definitions to $REPO_DIR"
    ;;
  status)
    gcloud batch jobs list \
      --project "$PROJECT_ID" \
      --location "$REGION" \
      --filter="name:${RUN_ID}" \
      --format='table(name.basename(),status.state,createTime)'
    echo "Artifacts: $BUCKET_ROOT/$RUN_ID/"
    ;;
  smoke-cloud)
    submit_job "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
    echo "Cloud smoke is now independent of this laptop. Check with the status action."
    ;;
  resume)
    submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"
    echo "Remote recovery controller submitted: $CONTROLLER_JOB"
    echo "The laptop may now be disconnected or switched off."
    echo "Artifacts will be written to $BUCKET_ROOT/$RUN_ID/"
    ;;
  run)
    submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"
    echo "Remote controller submitted: $CONTROLLER_JOB"
    echo "The laptop may now be disconnected or switched off."
    echo "The controller will run smoke, training and aggregation in order."
    echo "Artifacts will be written to $BUCKET_ROOT/$RUN_ID/"
    ;;
  orchestrate)
    if [[ "${HELDOUT_REMOTE_CONTROLLER:-}" != "1" ]]; then
      echo "orchestrate is an internal action for the remote controller" >&2
      exit 2
    fi
    if ! ensure_job_succeeds "$SMOKE_JOB" "$TEMP_DIR/smoke.json"; then
      echo "Cloud smoke failed; production was not submitted." >&2
      exit 1
    fi
    ensure_job_succeeds "$TRAIN_JOB" "$TEMP_DIR/train.json"
    ensure_job_succeeds "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    echo "Benchmark complete: $BUCKET_ROOT/$RUN_ID/analysis/"
    ;;
  orchestrate-resume)
    if [[ "${HELDOUT_REMOTE_CONTROLLER:-}" != "1" ]]; then
      echo "orchestrate-resume is an internal action for the remote controller" >&2
      exit 2
    fi
    if smoke_state="$(job_state "$SMOKE_JOB" 2>/dev/null)"; then
      if [[ "$smoke_state" != "SUCCEEDED" ]]; then
        wait_for_job "$SMOKE_JOB"
      fi
    else
      submit_job "$SMOKE_JOB" "$TEMP_DIR/smoke.json"
      wait_for_job "$SMOKE_JOB"
    fi
    complete_or_retry "${RUN_ID}-train" "$TRAIN_JOB" "$TEMP_DIR/train.json"
    complete_or_retry \
      "${RUN_ID}-aggregate" "$AGGREGATE_JOB" "$TEMP_DIR/aggregate.json"
    echo "Recovered benchmark complete: $BUCKET_ROOT/$RUN_ID/analysis/"
    ;;
  *)
    echo "Usage: $0 [run|resume|smoke-local|smoke-cloud|status|dry-run]" >&2
    exit 2
    ;;
esac
