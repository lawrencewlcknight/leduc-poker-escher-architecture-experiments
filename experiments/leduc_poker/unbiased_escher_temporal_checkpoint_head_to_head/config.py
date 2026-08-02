"""Configuration for Experiment 16 temporal checkpoint head-to-head."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    CANDIDATE_CONFIG as EXPERIMENT_7_CANDIDATE_CONFIG,
)


EXPERIMENT_ID = 16
EXPERIMENT_NAME = "unbiased_escher_temporal_checkpoint_head_to_head"
ALGORITHM_ID = "unbiased_control_variate_escher"
ALGORITHM_LABEL = "Unbiased Control-Variate ESCHER (Experiment 7)"

# Match the five independent seeds and inferential power of Deep CFR
# Experiment 27. Five seeds make 1 / 32 the smallest attainable one-sided
# exact sign-flip p-value.
DEFAULT_SEEDS = (1234, 2025, 31415, 27182, 16180)

# Logical checkpoint identifiers remain common across seeds. The corresponding
# policies are captured after the first complete outer iteration crossing each
# node threshold, and actual nodes touched are authoritative in every plot.
CHECKPOINT_SCHEDULE = (1, 2, 3, 4, 5)
CHECKPOINT_NODE_THRESHOLDS = (
    3_000_000,
    6_000_000,
    9_000_000,
    12_000_000,
    15_000_000,
)
TARGET_NODES = CHECKPOINT_NODE_THRESHOLDS[-1]

CANDIDATE_CONFIG = deepcopy(EXPERIMENT_7_CANDIDATE_CONFIG)

EQUIVALENCE_EPSILON = 1e-3
MEASURED_HOURS_PER_SEED = 10.874778532330835
EXPECTED_SEQUENTIAL_HOURS = 65
RECOMMENDED_BATCH_TIMEOUT_MINUTES = 96 * 60
BATCH_TIMEOUT_SECONDS = RECOMMENDED_BATCH_TIMEOUT_MINUTES * 60


def validate_config(
    config: Mapping[str, object],
    checkpoint_schedule=CHECKPOINT_SCHEDULE,
    checkpoint_node_thresholds=CHECKPOINT_NODE_THRESHOLDS,
) -> None:
    """Validate the uninterrupted node-threshold snapshot contract."""
    schedule = tuple(int(value) for value in checkpoint_schedule)
    thresholds = tuple(int(value) for value in checkpoint_node_thresholds)
    if len(schedule) != 5 or len(thresholds) != 5:
        raise ValueError("Exactly five checkpoint stages and thresholds are required")
    if any(left >= right for left, right in zip(schedule, schedule[1:])):
        raise ValueError("Checkpoint stages must be strictly increasing")
    if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError("Checkpoint node thresholds must be strictly increasing")
    if thresholds[-1] <= 0:
        raise ValueError("The final node threshold must be positive")
    if int(config["evaluation_frequency"]) != 1:
        raise ValueError(
            "Experiment 16 requires the Experiment 7 policy fit after every "
            "complete outer iteration"
        )
    if not bool(config["preserve_evaluation_rng"]):
        raise ValueError("Checkpoint evaluation must preserve the training RNG")
    if str(config["game_name"]) != "leduc_poker":
        raise ValueError("Exact Experiment 16 evaluation supports Leduc poker only")


__all__ = [
    "ALGORITHM_ID",
    "ALGORITHM_LABEL",
    "BATCH_TIMEOUT_SECONDS",
    "CANDIDATE_CONFIG",
    "CHECKPOINT_NODE_THRESHOLDS",
    "CHECKPOINT_SCHEDULE",
    "DEFAULT_SEEDS",
    "EQUIVALENCE_EPSILON",
    "EXPECTED_SEQUENTIAL_HOURS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "MEASURED_HOURS_PER_SEED",
    "RECOMMENDED_BATCH_TIMEOUT_MINUTES",
    "TARGET_NODES",
    "validate_config",
]
