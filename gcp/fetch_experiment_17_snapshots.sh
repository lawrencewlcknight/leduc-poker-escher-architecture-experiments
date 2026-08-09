#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash gcp/fetch_experiment_17_snapshots.sh RESULTS_BUCKET DESTINATION" >&2
  exit 2
fi

RESULTS_BUCKET="${1%/}"
DESTINATION="$2"

mkdir -p \
  "$DESTINATION/deep_cfr" \
  "$DESTINATION/dream" \
  "$DESTINATION/escher" \
  "$DESTINATION/unbiased_control_variate_escher"

gcloud storage cp --recursive \
  "$RESULTS_BUCKET/leduc-deep-cfr-exp27-20260801-164400/outputs/cloud/leduc-deep-cfr-exp27-/leduc_poker_deep_cfr_final_candidate_checkpoint_head_to_head_20260801_154711/snapshots" \
  "$DESTINATION/deep_cfr"

gcloud storage cp --recursive \
  "$RESULTS_BUCKET/leduc-dream-exp43-20260802-155924/outputs/cloud/leduc-dream-exp43/20260802_150224/snapshots" \
  "$DESTINATION/dream"

gcloud storage cp --recursive \
  "$RESULTS_BUCKET/leduc-escher-exp43-20260801-171016/outputs/cloud/leduc-escher-exp43/leduc_poker_escher_final_candidate_checkpoint_head_to_head_20260801_161251/snapshots" \
  "$DESTINATION/escher"

gcloud storage cp --recursive \
  "$RESULTS_BUCKET/leduc-escher-arch-exp16-20260802-155627/outputs/cloud/leduc-escher-arch-exp16/unbiased_escher_temporal_checkpoint_head_to_head_20260802_145917/snapshots" \
  "$DESTINATION/unbiased_control_variate_escher"
