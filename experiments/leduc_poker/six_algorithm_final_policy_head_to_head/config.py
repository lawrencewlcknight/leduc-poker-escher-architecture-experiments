"""Configuration contract for Experiment 17."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    ALGORITHMS as EXPERIMENT_7_ALGORITHMS,
    TARGET_NODES,
    UPSTREAM,
    VR_CONFIG as EXPERIMENT_7_VR_CONFIG,
)


EXPERIMENT_ID = 17
EXPERIMENT_NAME = "six_algorithm_final_policy_head_to_head"
GAME_NAME = "leduc_poker"
DEFAULT_SEEDS = (1234, 2025, 31415, 27182, 16180)

DEEP_CFR = "deep_cfr"
DREAM = "dream"
ESCHER = "escher"
VR_DEEP_DCFR_PLUS = "vr_deep_dcfr_plus"
VR_DEEP_PDCFR_PLUS = "vr_deep_pdcfr_plus"
UCV_ESCHER = "unbiased_control_variate_escher"

ALGORITHM_ORDER = (
    DEEP_CFR,
    DREAM,
    ESCHER,
    VR_DEEP_DCFR_PLUS,
    VR_DEEP_PDCFR_PLUS,
    UCV_ESCHER,
)

ALGORITHMS = {
    DEEP_CFR: {
        "algorithm_label": "Deep CFR",
        "source_experiment": "Deep CFR Experiment 27",
        "snapshot_subdirectory": DEEP_CFR,
        "requires_training": False,
    },
    DREAM: {
        "algorithm_label": "DREAM",
        "source_experiment": "DREAM Experiment 43",
        "snapshot_subdirectory": DREAM,
        "requires_training": False,
    },
    ESCHER: {
        "algorithm_label": "ESCHER",
        "source_experiment": "ESCHER Experiment 43",
        "snapshot_subdirectory": ESCHER,
        "requires_training": False,
    },
    VR_DEEP_DCFR_PLUS: {
        **deepcopy(EXPERIMENT_7_ALGORITHMS[VR_DEEP_DCFR_PLUS]),
        "source_experiment": "ESCHER Architecture Experiment 7 authors' parameterisation",
        "snapshot_subdirectory": VR_DEEP_DCFR_PLUS,
        "requires_training": True,
    },
    VR_DEEP_PDCFR_PLUS: {
        **deepcopy(EXPERIMENT_7_ALGORITHMS[VR_DEEP_PDCFR_PLUS]),
        "source_experiment": "ESCHER Architecture Experiment 7 authors' parameterisation",
        "snapshot_subdirectory": VR_DEEP_PDCFR_PLUS,
        "requires_training": True,
    },
    UCV_ESCHER: {
        "algorithm_label": "UCV-ESCHER",
        "source_experiment": "ESCHER Architecture Experiment 16",
        "snapshot_subdirectory": UCV_ESCHER,
        "requires_training": False,
    },
}

EXISTING_SNAPSHOT_ALGORITHMS = (DEEP_CFR, DREAM, ESCHER, UCV_ESCHER)
VR_ALGORITHMS = (VR_DEEP_DCFR_PLUS, VR_DEEP_PDCFR_PLUS)
VR_CONFIG = deepcopy(EXPERIMENT_7_VR_CONFIG)

# Measured from the completed Experiment 7 seed_summary.csv on n2-standard-8.
MEASURED_HOURS_PER_SEED = {
    VR_DEEP_DCFR_PLUS: 6.029077573118518,
    VR_DEEP_PDCFR_PLUS: 7.029803657529073,
}
MEASURED_SEQUENTIAL_TRAINING_HOURS = sum(MEASURED_HOURS_PER_SEED.values()) * len(
    DEFAULT_SEEDS
)
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 80
RECOMMENDED_BATCH_TIMEOUT_SECONDS = 345_600

# Exact evaluation has no sampled-match error. This epsilon is only a practical
# equivalence threshold for classifying very small exact EVs.
EQUIVALENCE_EPSILON = 1e-3


def validate_contract(
    seeds, target_nodes: int, vr_config, *, require_five_seeds: bool = True
) -> None:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be non-empty and distinct")
    if require_five_seeds and len(seeds) != 5:
        raise ValueError("Experiment 17 requires exactly five distinct training seeds")
    if int(target_nodes) <= 0:
        raise ValueError("target_nodes must be positive")
    if str(vr_config["game_name"]) != GAME_NAME:
        raise ValueError("Experiment 17 supports Leduc poker only")
    if not bool(vr_config["preserve_evaluation_rng"]):
        raise ValueError("VR exact evaluations must preserve the training RNG")


__all__ = [
    "ALGORITHMS",
    "ALGORITHM_ORDER",
    "DEFAULT_SEEDS",
    "EQUIVALENCE_EPSILON",
    "EXISTING_SNAPSHOT_ALGORITHMS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "MEASURED_HOURS_PER_SEED",
    "MEASURED_SEQUENTIAL_TRAINING_HOURS",
    "RECOMMENDED_BATCH_TIMEOUT_SECONDS",
    "TARGET_NODES",
    "UPSTREAM",
    "VR_ALGORITHMS",
    "VR_CONFIG",
    "validate_contract",
]
