"""Configuration for Experiment 18's UCV-ESCHER parallel-equivalence test."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    CANDIDATE_CONFIG,
)


EXPERIMENT_ID = 18
DEFAULT_SEEDS = [0, 1, 2]
TARGET_NODES = 15_000_000

SEQUENTIAL_VARIANT_ID = "ucv_escher_sequential"
PARALLEL_VARIANT_ID = "ucv_escher_ray_parallel"
PARALLEL_NUM_WORKERS = 3
PARALLEL_RAY_OBJECT_STORE_MEMORY = 512 * 1024 * 1024

# Declared before observing Experiment 18. These are deliberately tighter than
# the 0.05/0.02 margins used by the older ESCHER, DREAM and Deep CFR checks,
# because Experiment 7's final UCV-ESCHER exploitability is approximately 0.07.
FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN = 0.02
FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN = 0.01

VARIANTS = (
    {
        "variant_id": SEQUENTIAL_VARIANT_ID,
        "variant_label": "UCV-ESCHER sequential",
        "execution_backend": "sequential",
        "parallel_num_workers": 1,
        "variant_description": (
            "Exact Experiment 7 UCV-ESCHER with sequential trajectory collection "
            "and sequential control-learner optimisation."
        ),
    },
    {
        "variant_id": PARALLEL_VARIANT_ID,
        "variant_label": "UCV-ESCHER Ray parallel (3 workers)",
        "execution_backend": "ray_parallel",
        "parallel_num_workers": PARALLEL_NUM_WORKERS,
        "variant_description": (
            "The Experiment 7 learner with synchronous Ray-parallel trajectory "
            "collection and concurrent independent Q-fold/calibration updates."
        ),
    },
)

DEFAULT_CONFIG = deepcopy(CANDIDATE_CONFIG)

# Experiment 7 measured 11.22 hours per sequential UCV seed. Traversal and the
# four independent control learners are parallelised, but regret/policy fitting
# remains central. The estimate is intentionally conservative until measured.
MEASURED_SEQUENTIAL_HOURS_PER_SEED = 11.22
EXPECTED_PARALLEL_HOURS_PER_SEED = 8.0
EXPECTED_FULL_EXPERIMENT_HOURS = 64
RECOMMENDED_BATCH_TIMEOUT_MINUTES = 84 * 60
BATCH_TIMEOUT_SECONDS = RECOMMENDED_BATCH_TIMEOUT_MINUTES * 60
