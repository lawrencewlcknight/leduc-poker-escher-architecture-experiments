"""Frozen contract for the four-algorithm held-out benchmark."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    CANDIDATE_CONFIG as EXPERIMENT_7_UCV_CONFIG,
    VR_CONFIG as EXPERIMENT_7_VR_CONFIG,
)


EXPERIMENT_NAME = "four_algorithm_heldout_benchmark"
GAME_NAME = "leduc_poker"

DEEP_CFR = "deep_cfr"
VR_DEEP_DCFR_PLUS = "vr_deep_dcfr_plus"
VR_DEEP_PDCFR_PLUS = "vr_deep_pdcfr_plus"
UCV_ESCHER = "unbiased_control_variate_escher"

ALGORITHM_ORDER = (
    DEEP_CFR,
    VR_DEEP_DCFR_PLUS,
    VR_DEEP_PDCFR_PLUS,
    UCV_ESCHER,
)
ALGORITHMS = {
    DEEP_CFR: {"algorithm_label": "Deep CFR", "repository": "deep_cfr"},
    VR_DEEP_DCFR_PLUS: {
        "algorithm_label": "VR-DeepDCFR+",
        "repository": "escher_architecture",
    },
    VR_DEEP_PDCFR_PLUS: {
        "algorithm_label": "VR-DeepPDCFR+",
        "repository": "escher_architecture",
    },
    UCV_ESCHER: {
        "algorithm_label": "UCV-ESCHER",
        "repository": "escher_architecture",
    },
}

# These labels were checked against the archived experiment repositories when
# this contract was created. Smoke tests must use development seed 0 and must
# never consume a held-out label.
HELDOUT_SEEDS = (
    104729,
    130363,
    155921,
    181081,
    205759,
    230969,
    256019,
    281117,
)
SMOKE_SEEDS = (0,)

NODE_ENDPOINT = "node_15m"
TIME_ENDPOINT = "time_11h"
ENDPOINT_ORDER = (NODE_ENDPOINT, TIME_ENDPOINT)
TARGET_NODES = 15_000_000
TARGET_ACTIVE_SECONDS = 11 * 60 * 60

# These are safety caps, not resource budgets. A production worker stops as
# soon as both endpoint policies have been captured.
MAX_VR_ITERATIONS = 500
MAX_DEEP_CFR_ITERATIONS = 6_000

VR_CONFIG = deepcopy(EXPERIMENT_7_VR_CONFIG)
VR_CONFIG["max_num_iterations"] = MAX_VR_ITERATIONS
UCV_CONFIG = deepcopy(EXPERIMENT_7_UCV_CONFIG)
UCV_CONFIG["max_num_iterations"] = MAX_VR_ITERATIONS


def task_schedule(seeds=HELDOUT_SEEDS):
    """Return the stable Batch task-index mapping."""
    return [
        (algorithm_id, int(seed))
        for algorithm_id in ALGORITHM_ORDER
        for seed in seeds
    ]


def validate_contract(*, seeds, target_nodes: int, target_seconds: float, smoke: bool) -> None:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be non-empty and distinct")
    if int(target_nodes) <= 0:
        raise ValueError("The node endpoint must be positive")
    if float(target_seconds) < 0:
        raise ValueError("The time endpoint cannot be negative")
    if smoke:
        if any(seed in HELDOUT_SEEDS for seed in seeds):
            raise ValueError("Smoke tests must not consume held-out seeds")
    elif seeds != HELDOUT_SEEDS:
        raise ValueError(
            "Production runs require the frozen eight held-out seeds in their declared order"
        )


__all__ = [
    "ALGORITHMS",
    "ALGORITHM_ORDER",
    "DEEP_CFR",
    "ENDPOINT_ORDER",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "HELDOUT_SEEDS",
    "MAX_DEEP_CFR_ITERATIONS",
    "MAX_VR_ITERATIONS",
    "NODE_ENDPOINT",
    "SMOKE_SEEDS",
    "TARGET_ACTIVE_SECONDS",
    "TARGET_NODES",
    "TIME_ENDPOINT",
    "UCV_CONFIG",
    "UCV_ESCHER",
    "VR_CONFIG",
    "VR_DEEP_DCFR_PLUS",
    "VR_DEEP_PDCFR_PLUS",
    "task_schedule",
    "validate_contract",
]
