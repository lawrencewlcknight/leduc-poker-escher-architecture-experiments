#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODULE=experiments.leduc_poker.rnad_ucv_resolving.run
if [[ "$ACTION" == smoke-local ]]; then
  python3 -m "$MODULE" smoke --output-root "${SMOKE_OUTPUT:-$ROOT/outputs/exp49_smoke_$(date -u '+%Y%m%d-%H%M%S')}"
  exit
fi
[[ "$ACTION" == run || "$ACTION" == dry-run || "$ACTION" == status ]] || {
  echo "Usage: bash $0 [run|dry-run|status|smoke-local]" >&2; exit 2;
}
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set the destination BUCKET}"
: "${RUN_ID:?Set a new RUN_ID beginning exp49-rnad-}"
[[ "$RUN_ID" =~ ^exp49-rnad-[a-z0-9-]+$ && ${#RUN_ID} -le 45 ]] || {
  echo "Invalid RUN_ID (must begin exp49-rnad-, max 45 lowercase characters)" >&2; exit 2;
}
MODE="${MODE:-pilot}"
[[ "$MODE" == pilot || "$MODE" == full ]] || { echo "MODE must be pilot or full" >&2; exit 2; }
[[ "$BUCKET" == gs://* ]] && BUCKET_ROOT="${BUCKET%/}" || BUCKET_ROOT="gs://${BUCKET%/}"
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs describe "$RUN_ID-$MODE" --project="$PROJECT_ID" --location="$REGION" --format='value(status.state)'
  exit
fi
: "${SA_EMAIL:?Set the existing Batch runner service account}"
: "${REPO_REF:?Set REPO_REF to the pushed Experiment 49 commit}"
SOURCE_URI="${SOURCE_URI:-gs://clever-overview-399515-leduc-poker-dream-results/exp35-confirm-20260916-011231}"
WORKERS="${WORKERS:-4}"
[[ "$WORKERS" =~ ^[1-4]$ ]] || { echo "WORKERS must be 1-4" >&2; exit 2; }
PILOT_RUN_ID="${PILOT_RUN_ID:-}"
if [[ "$MODE" == full ]]; then
  [[ "$PILOT_RUN_ID" =~ ^exp49-rnad-[a-z0-9-]+$ ]] || { echo "Set PILOT_RUN_ID to the completed CPU pilot" >&2; exit 2; }
fi
RESOLVED_REF="$(git rev-parse --verify "${REPO_REF}^{commit}")"
git cat-file -e "$RESOLVED_REF:experiments/leduc_poker/rnad_ucv_resolving/run.py" || {
  echo "REPO_REF predates Experiment 49; push the implementation and choose that commit." >&2; exit 2;
}
if [[ "$ACTION" == run ]]; then
  gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null
  gcloud storage ls "$SOURCE_URI/analysis/snapshot_inventory.csv" >/dev/null
  if gcloud storage ls "$BUCKET_ROOT/$RUN_ID/manifest.json" >/dev/null 2>&1; then
    echo "Output prefix already used; choose a new RUN_ID." >&2; exit 2
  fi
  if [[ "$MODE" == full ]]; then
    gcloud storage ls "$BUCKET_ROOT/$PILOT_RUN_ID/SUCCESS.json" >/dev/null
  fi
fi
JOB_JSON="$(mktemp /tmp/exp49-batch.XXXXXX)"
trap 'rm -f "$JOB_JSON"' EXIT
python3 gcp/exp49_rnad_resolving_batch.py --bucket "$BUCKET_ROOT" --run-id "$RUN_ID" \
  --source-uri "$SOURCE_URI" --repo-ref "$RESOLVED_REF" --service-account "$SA_EMAIL" \
  --mode "$MODE" --workers "$WORKERS" --pilot-run-id "$PILOT_RUN_ID" \
  --max-hours "${MAX_HOURS:-24}" --output "$JOB_JSON"
if [[ "$ACTION" == dry-run ]]; then
  cat "$JOB_JSON"
else
  gcloud batch jobs submit "$RUN_ID-$MODE" --project="$PROJECT_ID" --location="$REGION" --config="$JOB_JSON"
  echo "Submitted CPU-only $MODE on one n2-standard-16. No blueprint retraining."
  echo "Outputs: $BUCKET_ROOT/$RUN_ID/"
fi
