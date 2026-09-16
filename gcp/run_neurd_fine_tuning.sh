#!/usr/bin/env bash
set -Eeuo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_policy_post_training.sh" neurd_fine_tuning "${1:-run}"

