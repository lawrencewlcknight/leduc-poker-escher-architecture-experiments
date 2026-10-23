#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-dry-run}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
MODULE=experiments.leduc_poker.advantage_noise_intervention.run
if [[ "$ACTION" == smoke-local ]]; then
  exec "${PYTHON:-python3}" -m "$MODULE" smoke --output-root "${SMOKE_OUTPUT:-$(mktemp -d /tmp/exp52-smoke.XXXXXX)}"
fi
: "${PROJECT_ID:?Set PROJECT_ID}" "${REGION:?Set REGION}" "${BUCKET:?Set BUCKET}"
: "${SA_EMAIL:?Set SA_EMAIL}" "${REPO_REF:?Set exact pushed commit SHA}" "${RUN_ID:?Set persistent RUN_ID}"
SOURCE_BUCKET="${SOURCE_BUCKET:-$BUCKET}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-exp35-confirm-20260916-011231}"
PARALLELISM="${PARALLELISM:-3}"
ATTEMPT="${ATTEMPT:-initial}"
[[ "$RUN_ID" =~ ^[a-z][a-z0-9-]*[a-z0-9]$ && ${#RUN_ID} -le 30 ]] || exit 2
[[ "$SOURCE_RUN_ID" =~ ^[a-z][a-z0-9-]+$ ]] || exit 2
[[ "$REPO_REF" =~ ^[0-9a-f]{40}$ ]] || { echo 'Use an exact 40-character commit SHA' >&2; exit 2; }
[[ "$BUCKET" == gs://* && "$SOURCE_BUCKET" == gs://* ]] || exit 2
[[ "$PARALLELISM" =~ ^[0-9]+$ && "$PARALLELISM" -ge 1 && "$PARALLELISM" -le 6 ]] || exit 2
[[ "$ATTEMPT" =~ ^[a-z0-9-]+$ && ${#ATTEMPT} -le 12 ]] || exit 2
BUCKET="${BUCKET%/}"; SOURCE_BUCKET="${SOURCE_BUCKET%/}"
TEMP_DIR="$(mktemp -d /tmp/exp52-jobs.XXXXXX)"
build() {
  python3 "$SCRIPT_DIR/advantage_noise_intervention_batch.py" --kind "$1" --output "$TEMP_DIR/$1.json" \
    --project "$PROJECT_ID" --region "$REGION" --bucket "$BUCKET" --source-bucket "$SOURCE_BUCKET" \
    --source-run-id "$SOURCE_RUN_ID" --run-id "$RUN_ID" --service-account "$SA_EMAIL" \
    --repo-ref "$REPO_REF" --parallelism "$PARALLELISM" --attempt "$ATTEMPT"
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
    for kind in controller smoke pilot worker aggregate; do build "$kind"; done
    echo "Specifications only: $TEMP_DIR" ;;
  status)
    gcloud batch jobs list --project "$PROJECT_ID" --location "$REGION" --filter="name:$RUN_ID" \
      --format='table(name.basename(),status.state,createTime)' ;;
  smoke-cloud|pilot-cloud)
    kind="${ACTION%-cloud}"
    git cat-file -e "$REPO_REF:experiments/leduc_poker/advantage_noise_intervention/run.py"
    build "$kind"; submit "$RUN_ID-$kind-$ATTEMPT" "$kind" ;;
  run|resume)
    git cat-file -e "$REPO_REF:experiments/leduc_poker/advantage_noise_intervention/run.py"
    if [[ "$ACTION" == resume && "$ATTEMPT" == initial ]]; then
      echo 'Resume requires a new ATTEMPT; keep RUN_ID and REPO_REF unchanged.' >&2; exit 2
    fi
    gcloud storage cp "$BUCKET/$RUN_ID/pilot/PILOT_SUCCESS.json" "$TEMP_DIR/pilot.json"
    python3 - "$TEMP_DIR/pilot.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p['status']=='complete' and p['result']['pilot'] and not p['result']['smoke']
assert p['result']['fits']==6 and p['result']['fit_steps']==750
PY
    build controller; submit "$RUN_ID-controller-$ATTEMPT" controller ;;
  orchestrate)
    [[ "${EXP52_REMOTE_CONTROLLER:-}" == 1 ]] || exit 2
    for kind in smoke worker aggregate; do build "$kind"; done
    ensure "$RUN_ID-smoke-$ATTEMPT" smoke
    gcloud storage cp "$BUCKET/$RUN_ID/smoke/SMOKE_SUCCESS.json" "$TEMP_DIR/smoke.json"
    gcloud storage cp "$BUCKET/$RUN_ID/pilot/PILOT_SUCCESS.json" "$TEMP_DIR/pilot.json"
    python3 - "$TEMP_DIR/pilot.json" "$TEMP_DIR/smoke.json" <<'PY'
import json,sys
p,s=(json.load(open(path)) for path in sys.argv[1:])
assert p['status']==s['status']=='complete'
assert p['result']['pilot'] and not p['result']['smoke'] and p['result']['fits']==6
assert p['code']==s['code'], 'Pilot and study code differ'
for key in ('python','torch','numpy','open_spiel','platform'):
    assert p['runtime'][key]==s['runtime'][key], f'Runtime mismatch: {key}'
PY
    ensure "$RUN_ID-worker-$ATTEMPT" worker
    ensure "$RUN_ID-aggregate-$ATTEMPT" aggregate ;;
  *) echo 'Usage: dry-run | smoke-local | smoke-cloud | pilot-cloud | run | resume | status' >&2; exit 2 ;;
esac
