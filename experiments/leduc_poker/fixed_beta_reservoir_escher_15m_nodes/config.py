"""Configuration and immutable Experiment 7 references for Experiment 14."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.config import (
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    CANDIDATE_CONFIG as EXPERIMENT_13_CONFIG,
    DEFAULT_SEEDS,
)
from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    MAX_NUM_ITERATIONS,
    TARGET_NODES,
)


EXPERIMENT_ID = 14
REFERENCE_VR_DEEP_DCFR_PLUS = "vr_deep_dcfr_plus"
REFERENCE_VR_DEEP_PDCFR_PLUS = "vr_deep_pdcfr_plus"
REFERENCE_UNBIASED_ESCHER = "unbiased_control_variate_escher"
REFERENCE_ALGORITHM_IDS = (
    REFERENCE_VR_DEEP_DCFR_PLUS,
    REFERENCE_VR_DEEP_PDCFR_PLUS,
    REFERENCE_UNBIASED_ESCHER,
)
REFERENCE_ALGORITHM_LABELS = {
    REFERENCE_VR_DEEP_DCFR_PLUS: "VR-DeepDCFR+ (Experiment 7)",
    REFERENCE_VR_DEEP_PDCFR_PLUS: "VR-DeepPDCFR+ (Experiment 7)",
    REFERENCE_UNBIASED_ESCHER: (
        "Unbiased Control-Variate ESCHER (Experiment 7)"
    ),
}

CANDIDATE_CONFIG = deepcopy(EXPERIMENT_13_CONFIG)
CANDIDATE_CONFIG["max_num_iterations"] = MAX_NUM_ITERATIONS

REFERENCE_CURVES = Path(__file__).with_name("experiment7_checkpoint_curves.csv")
REFERENCE_SUMMARIES = Path(__file__).with_name("experiment7_seed_summary.csv")
REFERENCE_CURVES_SHA256 = (
    "d0869cc7926525ddc7afd31b9a87c5d30929a10b556f69e79a6d943ebb6b9e38"
)
REFERENCE_SUMMARIES_SHA256 = (
    "028d6f364613cee6211858d2792785957c8fd558e5435a8df976737236610853"
)
REFERENCE_CURVE_ROWS = 862
REFERENCE_SUMMARY_ROWS = 9

EXPERIMENT_7_SOURCE = {
    "batch_job": (
        "projects/clever-overview-399515/locations/europe-west1/jobs/"
        "leduc-escher-arch-exp7-20260720-002431"
    ),
    "run_directory": "unbiased_escher_vs_vr_deep_cfr_15m_nodes_20260719_232708",
    "curves_source_file": "checkpoint_curves.csv",
    "summary_source_file": "seed_summary.csv",
    "algorithm_ids": REFERENCE_ALGORITHM_IDS,
    "seeds": DEFAULT_SEEDS,
    "target_nodes_touched": TARGET_NODES,
}

# Experiment 7's Experiment 6 candidate required 32.62 hours for three
# sequential 15M-node seeds. This architecture performs the same optimisation
# work, so allow 36 hours and retain a 48-hour Batch timeout.
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 36
RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES = 48 * 60
BATCH_TIMEOUT_SECONDS = RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES * 60
