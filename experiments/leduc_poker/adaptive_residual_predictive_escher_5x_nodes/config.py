"""Configuration for Experiment 4 at Experiment 2 node budgets."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.leduc_poker.adaptive_residual_predictive_escher.config import (
    ADAPTIVE_CONFIG as EXPERIMENT_3_ADAPTIVE_CONFIG,
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    DEFAULT_SEEDS,
)


# Experiment 4 is deliberately a horizon-only extension: the architecture and
# every learning setting are identical to Experiment 3.
ADAPTIVE_CONFIG = deepcopy(EXPERIMENT_3_ADAPTIVE_CONFIG)

# Exact final ESCHER node totals from Experiment 2. The adaptive solver stops
# after the first complete outer iteration crossing its paired target, matching
# the node-budget protocol used for both VR algorithms in Experiments 1 and 2.
EXPERIMENT_2_NODE_TARGETS = {
    0: 4_700_205,
    1: 4_701_540,
    2: 4_684_695,
}

REFERENCE_CURVES = Path(__file__).with_name("experiment2_checkpoint_curves.csv")
REFERENCE_CURVES_SHA256 = (
    "0bd4ace4ea2611a34971aaf7c6ab676c05e39faa3bb3069113d641fac3b53b85"
)
REFERENCE_CURVE_ROWS = 323

EXPERIMENT_2_SOURCE = {
    "batch_job": (
        "projects/clever-overview-399515/locations/europe-west1/jobs/"
        "leduc-escher-arch-exp2-20260717-105458"
    ),
    "run_directory": "escher_vs_vr_deep_cfr_5x_nodes_20260717_095755",
    "source_file": "checkpoint_curves.csv",
    "source_sha256": REFERENCE_CURVES_SHA256,
    "algorithms": ["escher_exp28", "vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"],
    "seeds": DEFAULT_SEEDS,
}

EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 8
BATCH_TIMEOUT_SECONDS = 18 * 60 * 60

