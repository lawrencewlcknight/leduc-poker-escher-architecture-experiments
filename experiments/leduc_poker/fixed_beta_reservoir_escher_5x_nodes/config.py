"""Configuration and immutable Experiment 6 references for Experiment 13."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    UNBIASED_CONFIG,
)


EXPERIMENT_ID = 13
ALGORITHM_ID = "fixed_beta_reservoir_escher"
ALGORITHM_LABEL = "Fixed-Beta Reservoir ESCHER"
REFERENCE_ALGORITHM_ID = "unbiased_control_variate_escher"
REFERENCE_ALGORITHM_LABEL = "Unbiased Control-Variate ESCHER (Experiment 6)"

CANDIDATE_CONFIG = deepcopy(UNBIASED_CONFIG)
CANDIDATE_CONFIG.update(
    {
        "q_ensemble_size": 3,
        "fixed_control_variate_beta": 1.0,
    }
)

REFERENCE_CURVES = Path(__file__).with_name("experiment6_checkpoint_curves.csv")
REFERENCE_SUMMARIES = Path(__file__).with_name("experiment6_seed_summary.csv")
REFERENCE_CURVES_SHA256 = (
    "7f0ecfca091130565275fc27c775cdcd4e96b62eb122759209d9d4f17b0e65b5"
)
REFERENCE_SUMMARIES_SHA256 = (
    "10a43adeb4f415f34e45f2498cd25d85977bb53e0da13300ed7618071635daf9"
)
REFERENCE_CURVE_ROWS = 90
REFERENCE_SUMMARY_ROWS = 3

EXPERIMENT_6_SOURCE = {
    "batch_job": (
        "projects/clever-overview-399515/locations/europe-west1/jobs/"
        "leduc-escher-arch-exp6-20260718-230108"
    ),
    "run_directory": "unbiased_control_variate_escher_5x_nodes_20260718_220419",
    "curves_source_file": "candidate_checkpoint_curves.csv",
    "summary_source_file": "candidate_seed_summary.csv",
    "algorithm_id": REFERENCE_ALGORITHM_ID,
    "seeds": DEFAULT_SEEDS,
}

# Experiment 6 required 10.59 hours for three sequential seeds. This candidate
# performs the same optimisation work; reservoir replacement and fixed beta add
# negligible compute. Allow 12 hours and retain a 24-hour Batch timeout.
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 12
RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES = 24 * 60
BATCH_TIMEOUT_SECONDS = RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES * 60
