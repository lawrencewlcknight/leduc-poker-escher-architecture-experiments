#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash gcp/fetch_experiment_17_snapshots.sh DESTINATION" >&2
  exit 2
fi

DESTINATION="$1"

# The four source experiments were uploaded by three different repositories.
# They therefore live in repository-specific buckets rather than the output
# bucket selected for the Experiment 17 Batch job. Keep the audited defaults
# explicit and allow an operator to override an individual source for a copied
# project or migrated artifact set.
DEEP_CFR_RESULTS_BUCKET="${EXPERIMENT_17_DEEP_CFR_RESULTS_BUCKET:-gs://clever-overview-399515-leduc-poker-results}"
DREAM_RESULTS_BUCKET="${EXPERIMENT_17_DREAM_RESULTS_BUCKET:-gs://clever-overview-399515-leduc-poker-dream-results}"
ESCHER_RESULTS_BUCKET="${EXPERIMENT_17_ESCHER_RESULTS_BUCKET:-gs://clever-overview-399515-leduc-poker-escher-results}"
UCV_ESCHER_RESULTS_BUCKET="${EXPERIMENT_17_UCV_ESCHER_RESULTS_BUCKET:-gs://clever-overview-399515-leduc-poker-dream-results}"

for variable in \
  DEEP_CFR_RESULTS_BUCKET \
  DREAM_RESULTS_BUCKET \
  ESCHER_RESULTS_BUCKET \
  UCV_ESCHER_RESULTS_BUCKET; do
  printf -v "$variable" '%s' "${!variable%/}"
done

mkdir -p \
  "$DESTINATION/deep_cfr" \
  "$DESTINATION/dream" \
  "$DESTINATION/escher" \
  "$DESTINATION/unbiased_control_variate_escher"

gcloud storage cp --recursive \
  "$DEEP_CFR_RESULTS_BUCKET/leduc-deep-cfr-exp27-20260801-164400/outputs/cloud/leduc-deep-cfr-exp27-/leduc_poker_deep_cfr_final_candidate_checkpoint_head_to_head_20260801_154711/snapshots" \
  "$DESTINATION/deep_cfr"

gcloud storage cp --recursive \
  "$DREAM_RESULTS_BUCKET/leduc-dream-exp43-20260802-155924/outputs/cloud/leduc-dream-exp43/20260802_150224/snapshots" \
  "$DESTINATION/dream"

gcloud storage cp --recursive \
  "$ESCHER_RESULTS_BUCKET/leduc-escher-exp43-20260801-171016/outputs/cloud/leduc-escher-exp43/leduc_poker_escher_final_candidate_checkpoint_head_to_head_20260801_161251/snapshots" \
  "$DESTINATION/escher"

gcloud storage cp --recursive \
  "$UCV_ESCHER_RESULTS_BUCKET/leduc-escher-arch-exp16-20260802-155627/outputs/cloud/leduc-escher-arch-exp16/unbiased_escher_temporal_checkpoint_head_to_head_20260802_145917/snapshots" \
  "$DESTINATION/unbiased_control_variate_escher"
