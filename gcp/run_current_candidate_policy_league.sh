#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-run}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODULE=experiments.leduc_poker.current_candidate_policy_league.run
SOURCE_ROOT="${LEAGUE_SOURCE_ROOT:-$ROOT/outputs/exp46_source}"
if [[ "$ACTION" == smoke-local ]]; then
  export DEEP_CFR_REPO="${DEEP_CFR_REPO:-$ROOT/../../leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/exp46-mpl}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python3 -m "$MODULE" prepare --source-root "$SOURCE_ROOT" --smoke-only
  python3 -m "$MODULE" smoke --source-root "$SOURCE_ROOT" \
    --output-root "${LEAGUE_SMOKE_OUTPUT:-$ROOT/outputs/exp46_smoke_$(date -u '+%Y%m%d-%H%M%S')}"
  exit
fi
if [[ "$ACTION" == preflight ]]; then
  python3 -m "$MODULE" prepare --source-root "$SOURCE_ROOT"
  exit
fi
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${BUCKET:?Set BUCKET (destination only; comparator source buckets are pinned)}"
: "${RUN_ID:?Set a new RUN_ID beginning exp46-league-}"
[[ "$BUCKET" == gs://* ]] && DESTINATION="${BUCKET%/}" || DESTINATION="gs://${BUCKET%/}"
if [[ ! "$RUN_ID" =~ ^exp46-league-[a-z0-9-]+$ || ${#RUN_ID} -gt 45 ]]; then
  echo "RUN_ID must begin exp46-league- and be at most 45 lowercase letters/digits/hyphens" >&2; exit 2
fi
if [[ "$ACTION" == status ]]; then
  gcloud batch jobs describe "$RUN_ID-evaluate" --project="$PROJECT_ID" --location="$REGION" --format='value(status.state)'
  echo "Outputs: $DESTINATION/$RUN_ID/"
  exit
fi
[[ "$ACTION" == run || "$ACTION" == dry-run ]] || {
  echo "Usage: $0 [run|dry-run|preflight|status|smoke-local]" >&2; exit 2;
}
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${REPO_REF:?Set REPO_REF to the pushed experiment commit}"
REPO_REF="$(git rev-parse --verify "${REPO_REF}^{commit}")"
git cat-file -e "$REPO_REF:experiments/leduc_poker/current_candidate_policy_league/run.py" || {
  echo "REPO_REF predates architecture Experiment 46; commit and push the implementation first" >&2; exit 2;
}
if [[ "$ACTION" == run ]]; then
  gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" >/dev/null
  if gcloud storage ls "$DESTINATION/$RUN_ID/" >/dev/null 2>&1; then
    echo "Destination exists; use a new RUN_ID. Existing outputs will not be overwritten." >&2; exit 2
  fi
  # Metadata-only gate: reject missing seed/endpoints before starting a paid VM.
  python3 -m "$MODULE" prepare --source-root "$SOURCE_ROOT"
fi
JOB_JSON="$(mktemp /tmp/exp46-batch.XXXXXX)"
trap 'rm -f "$JOB_JSON"' EXIT
python3 gcp/current_candidate_policy_league_batch.py --bucket "$DESTINATION" --run-id "$RUN_ID" \
  --repo-ref "$REPO_REF" --service-account "$SA_EMAIL" --output "$JOB_JSON"
if [[ "$ACTION" == dry-run ]]; then
  cat "$JOB_JSON"
else
  gcloud batch jobs submit "$RUN_ID-evaluate" --project="$PROJECT_ID" --location="$REGION" --config="$JOB_JSON"
  echo "Submitted one evaluation-only VM with a built-in smoke gate. Laptop may disconnect."
  echo "Outputs: $DESTINATION/$RUN_ID/"
fi
