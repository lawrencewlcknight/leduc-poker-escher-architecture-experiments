"""Configuration for the 15-million-node Experiment 7 comparison."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.config import (
    ALGORITHMS as EXPERIMENT_2_ALGORITHMS,
    UPSTREAM,
    VR_PAPER_CONFIG,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    ALGORITHM_ID as CANDIDATE_ALGORITHM_ID,
    ALGORITHM_LABEL as CANDIDATE_ALGORITHM_LABEL,
    UNBIASED_CONFIG,
)


EXPERIMENT_ID = 7
DEFAULT_SEEDS = [0, 1, 2]
TARGET_NODES = 15_000_000

VR_DEEP_DCFR_PLUS = "vr_deep_dcfr_plus"
VR_DEEP_PDCFR_PLUS = "vr_deep_pdcfr_plus"
DEFAULT_ALGORITHM_IDS = (
    VR_DEEP_DCFR_PLUS,
    VR_DEEP_PDCFR_PLUS,
    CANDIDATE_ALGORITHM_ID,
)

ALGORITHMS = {
    VR_DEEP_DCFR_PLUS: deepcopy(EXPERIMENT_2_ALGORITHMS[VR_DEEP_DCFR_PLUS]),
    VR_DEEP_PDCFR_PLUS: deepcopy(EXPERIMENT_2_ALGORITHMS[VR_DEEP_PDCFR_PLUS]),
    CANDIDATE_ALGORITHM_ID: {
        "algorithm_id": CANDIDATE_ALGORITHM_ID,
        "algorithm_label": CANDIDATE_ALGORITHM_LABEL,
        "class_name": "UnbiasedControlVariateEscher",
    },
}

# Experiment 2 and Experiment 6 reached about 4.7M nodes in 28--31 outer
# iterations. A 120-iteration safety cap comfortably covers the projected
# 89--99 iterations needed to cross 15M without changing an update rule.
MAX_NUM_ITERATIONS = 120
VR_CONFIG = deepcopy(VR_PAPER_CONFIG)
VR_CONFIG["max_num_iterations"] = MAX_NUM_ITERATIONS
CANDIDATE_CONFIG = deepcopy(UNBIASED_CONFIG)
CANDIDATE_CONFIG["max_num_iterations"] = MAX_NUM_ITERATIONS

# Estimates use measured Experiment 2 and 6 throughput and include conservative
# operational headroom for longer replay buffers, evaluation and VM variance.
MEASURED_SEQUENTIAL_RUNTIME_HOURS = 65
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 78
RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES = 96 * 60
SEQUENTIAL_BATCH_TIMEOUT_SECONDS = RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES * 60
MEASURED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS = 34
EXPECTED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS = 42
PARALLEL_BATCH_TIMEOUT_SECONDS = 48 * 60 * 60

MEASURED_RUNTIME_PER_SEED_HOURS = {
    VR_DEEP_DCFR_PLUS: 4.63,
    VR_DEEP_PDCFR_PLUS: 5.66,
    CANDIDATE_ALGORITHM_ID: 11.22,
}
