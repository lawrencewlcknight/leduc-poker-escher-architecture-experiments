#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODULE=experiments.leduc_poker.grouped_ucv_temporal_head_to_head.run
SOURCE_ROOT="${SOURCE_ROOT:-$ROOT/outputs/exp45_source}"
if [[ "$ACTION" == smoke-local ]]; then
  : "${BUCKET:?Set BUCKET to the bucket holding Experiment 35}"
  [[ "$BUCKET" == gs://* ]] && SOURCE_BUCKET="${BUCKET%/}" || SOURCE_BUCKET="gs://${BUCKET%/}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/exp45-mpl}"
  python3 -m "$MODULE" fetch --smoke-only --source-root "$SOURCE_ROOT" \
    --source-uri "$SOURCE_BUCKET/${EXP35_RUN_ID:-exp35-confirm-20260916-011231}"
  python3 -m "$MODULE" smoke --source-root "$SOURCE_ROOT" \
    --output-root "${SMOKE_OUTPUT:-$ROOT/outputs/exp45_smoke_$(date -u '+%Y%m%d-%H%M%S')}"
  exit
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET}"
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
: "${RUN_ID:?Set a new RUN_ID beginning exp45-h2h-}"
if [[ ! "$RUN_ID" =~ ^exp45-h2h-[a-z0-9-]+$ || ${#RUN_ID} -gt 45 ]]; then
  echo "RUN_ID must begin exp45-h2h- and contain at most 45 lowercase letters/digits/hyphens." >&2; exit 2
fi
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs describe "$RUN_ID-evaluate" --project="$PROJECT_ID" \
    --location="$REGION" --format='value(status.state)'
  echo "Outputs: $BUCKET_ROOT/$RUN_ID/"
  exit
fi
[[ "$ACTION" == run || "$ACTION" == dry-run ]] || {
  echo "Usage: $0 [run|dry-run|status|smoke-local]" >&2; exit 2;
}
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed experiment commit}"
EXP35_RUN_ID="${EXP35_RUN_ID:-exp35-confirm-20260916-011231}"
[[ "$EXP35_RUN_ID" =~ ^[a-z0-9-]+$ ]] || { echo "Invalid EXP35_RUN_ID" >&2; exit 2; }
git cat-file -e "${REPO_REF}^{commit}" || { echo "REPO_REF is not a local commit" >&2; exit 2; }
git cat-file -e "${REPO_REF}:experiments/leduc_poker/grouped_ucv_temporal_head_to_head/run.py" || {
  echo "REPO_REF predates architecture Experiment 45" >&2; exit 2;
}
if [[ "$ACTION" == run ]]; then
  gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null
  gcloud storage ls "$BUCKET_ROOT/$EXP35_RUN_ID/analysis/snapshot_inventory.csv" >/dev/null
  if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/SUCCESS.json" >/dev/null 2>&1; then
    echo "Completed output exists; choose a new RUN_ID" >&2; exit 2
  fi
fi
JOB_JSON="$(mktemp /tmp/exp45-batch.XXXXXX)"
trap 'rm -f "$JOB_JSON"' EXIT
python3 gcp/grouped_ucv_temporal_head_to_head_batch.py \
  --bucket "$BUCKET_ROOT" --run-id "$RUN_ID" --source-run-id "$EXP35_RUN_ID" \
  --repo-ref "$REPO_REF" --service-account "$SA_EMAIL" --output "$JOB_JSON"
if [[ "$ACTION" == dry-run ]]; then
  cat "$JOB_JSON"
else
  gcloud batch jobs submit "$RUN_ID-evaluate" --project="$PROJECT_ID" \
    --location="$REGION" --config="$JOB_JSON"
  echo "Submitted one evaluation VM (no retraining). Your laptop can now disconnect."
  echo "Outputs: $BUCKET_ROOT/$RUN_ID/"
fi
