#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash gcp/fetch_experiment_17_snapshots.sh DESTINATION" >&2
  exit 2
fi

DESTINATION="$1"

# The source experiments originally lived in three repository-specific buckets,
# but the Batch service account intentionally has access only to this repository's
# results bucket. Experiment 17 therefore uses an audited, versioned staging bundle
# containing the 100 original snapshot objects. This avoids granting the Batch
# identity broad access to unrelated Deep CFR and ESCHER artifacts.
SOURCE_BUNDLE_ROOT="${EXPERIMENT_17_SOURCE_BUNDLE_ROOT:-gs://clever-overview-399515-leduc-poker-dream-results/experiment-17-inputs/six-algorithm-final-policy-head-to-head-v1}"
SOURCE_BUNDLE_ROOT="${SOURCE_BUNDLE_ROOT%/}"

mkdir -p \
  "$DESTINATION/deep_cfr" \
  "$DESTINATION/dream" \
  "$DESTINATION/escher" \
  "$DESTINATION/unbiased_control_variate_escher"

for algorithm in \
  deep_cfr \
  dream \
  escher \
  unbiased_control_variate_escher; do
  gcloud storage cp --recursive \
    "$SOURCE_BUNDLE_ROOT/$algorithm/snapshots" \
    "$DESTINATION/$algorithm"
done
