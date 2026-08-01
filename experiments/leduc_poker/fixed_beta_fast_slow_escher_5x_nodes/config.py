"""Configuration and immutable references for Experiment 15."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.config import (
    FAST_SLOW_CONFIG,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
)


EXPERIMENT_ID = 15
ALGORITHM_ID = "fixed_beta_fast_slow_control_critic_escher"
ALGORITHM_LABEL = "Fixed-Beta Fast/Slow Control-Critic ESCHER"

EXPERIMENT_6_ALGORITHM_ID = "unbiased_control_variate_escher"
EXPERIMENT_6_ALGORITHM_LABEL = "Unbiased Control-Variate ESCHER (Experiment 6)"
EXPERIMENT_9_ALGORITHM_ID = "fast_slow_control_critic_escher"
EXPERIMENT_9_ALGORITHM_LABEL = "Fast/Slow Control-Critic ESCHER (Experiment 9)"
EXPERIMENT_13_ALGORITHM_ID = "fixed_beta_reservoir_escher"
EXPERIMENT_13_ALGORITHM_LABEL = "Fixed-Beta Reservoir ESCHER (Experiment 13)"

CANDIDATE_CONFIG = deepcopy(FAST_SLOW_CONFIG)
CANDIDATE_CONFIG.update(
    {
        "q_ensemble_size": 3,
        "fixed_control_variate_beta": 1.0,
    }
)

EXPERIMENT_9_CURVES = Path(__file__).with_name(
    "experiment9_combined_checkpoint_curves.csv"
)
EXPERIMENT_9_SUMMARIES = Path(__file__).with_name(
    "experiment9_combined_seed_summary.csv"
)
EXPERIMENT_13_CURVES = Path(__file__).with_name(
    "experiment13_checkpoint_curves.csv"
)
EXPERIMENT_13_SUMMARIES = Path(__file__).with_name(
    "experiment13_seed_summary.csv"
)

EXPERIMENT_9_CURVES_SHA256 = (
    "b811edc29f6f50d92bba6763eba4a76df6864b6143985d877be3ddb293617994"
)
EXPERIMENT_9_SUMMARIES_SHA256 = (
    "583e9949c3c02ac781cdbc76c15951a8db26a30ca4d5dae929b52ea5083e47f4"
)
EXPERIMENT_13_CURVES_SHA256 = (
    "586298bdc0453c6103ec7f3993f76a666dc3544c2b25a65f42d3124627c4a8fd"
)
EXPERIMENT_13_SUMMARIES_SHA256 = (
    "d55a91eb855b506f78de45cb1817a4a063c89e8d53e210428a4e1d8c9af63f04"
)
EXPERIMENT_9_CURVE_ROWS = 182
EXPERIMENT_9_SUMMARY_ROWS = 6
EXPERIMENT_13_CURVE_ROWS = 92
EXPERIMENT_13_SUMMARY_ROWS = 3

REFERENCE_SOURCES = {
    "experiment_9": {
        "batch_job": (
            "projects/clever-overview-399515/locations/europe-west1/jobs/"
            "leduc-escher-arch-exp9-20260720-002452"
        ),
        "run_directory": "fast_slow_control_critic_escher_5x_nodes_20260719_232742",
        "algorithm_ids": (
            EXPERIMENT_6_ALGORITHM_ID,
            EXPERIMENT_9_ALGORITHM_ID,
        ),
    },
    "experiment_13": {
        "batch_job": (
            "projects/clever-overview-399515/locations/europe-west1/jobs/"
            "leduc-escher-arch-exp13-20260725-032411"
        ),
        "run_directory": "fixed_beta_reservoir_escher_5x_nodes_20260725_022722",
        "algorithm_ids": (EXPERIMENT_13_ALGORITHM_ID,),
    },
}

# Experiment 9 took 15.36 hours for three sequential seeds on n2-standard-8.
# Fixing beta and isolating replay RNGs add negligible work. Budget 17 hours
# expected and provide a 36-hour Batch timeout for setup and runtime variance.
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 17
RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES = 36 * 60
BATCH_TIMEOUT_SECONDS = RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES * 60
