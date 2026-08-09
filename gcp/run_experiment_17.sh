#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_ROOT="${EXPERIMENT_17_SNAPSHOT_ROOT:-/tmp/exp17_snapshots}"

# Use one entry point for smoke and production Batch jobs so neither can omit
# the external snapshot staging step required by the four pre-trained arms.
bash gcp/fetch_experiment_17_snapshots.sh "$SNAPSHOT_ROOT"

exec python -m experiments.leduc_poker.six_algorithm_final_policy_head_to_head.run \
  --snapshot-root "$SNAPSHOT_ROOT" \
  "$@"
