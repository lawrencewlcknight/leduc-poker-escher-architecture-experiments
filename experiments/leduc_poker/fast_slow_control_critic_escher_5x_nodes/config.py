"""Configuration and immutable Experiment 6 references for Experiment 9."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    UNBIASED_CONFIG,
)


EXPERIMENT_ID = 9
ALGORITHM_ID = "fast_slow_control_critic_escher"
ALGORITHM_LABEL = "Fast/Slow Control-Critic ESCHER"
REFERENCE_ALGORITHM_ID = "unbiased_control_variate_escher"
REFERENCE_ALGORITHM_LABEL = "Unbiased Control-Variate ESCHER (Experiment 6)"

FAST_SLOW_CONFIG = deepcopy(UNBIASED_CONFIG)
FAST_SLOW_CONFIG.update(
    {
        # Preserve Experiment 6's three disjoint trajectory folds. Each fold
        # now owns a recent fast critic and a lifetime-reservoir slow critic.
        "q_ensemble_size": 3,
        # Capacity covers roughly 1.5 typical Leduc outer iterations, avoiding
        # truncation. Replay is cleared before every iteration, so the semantic
        # window is exactly one outer iteration.
        "fast_q_buffer_size": 250_000,
        "fast_q_train_steps": 5_000,
        # The controller replay is recent/circular and is fitted to returns from
        # predictions made by folds that did not train on that trajectory.
        "rho_buffer_size": 250_000,
        "rho_batch_size": 2_048,
        "rho_train_steps": 2_000,
        "rho_learning_rate": 1e-3,
    }
)

REFERENCE_CURVES = Path(__file__).with_name("experiment6_checkpoint_curves.csv")
REFERENCE_SUMMARIES = Path(__file__).with_name("experiment6_seed_summary.csv")
# Filled from the immutable cloud output copied into this experiment package.
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

# Experiment 6 took 10.59 hours for three sequential seeds. The new critic does
# 1.5x as many Q optimisation steps, two-timescale inference and controller
# fitting; allow about 24 hours operationally and retain a 48-hour timeout.
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 24
RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES = 48 * 60
BATCH_TIMEOUT_SECONDS = RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES * 60
