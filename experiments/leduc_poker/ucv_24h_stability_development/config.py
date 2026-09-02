"""Frozen contract for Experiment 23's UCV stability development study."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Mapping, Sequence

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    HELDOUT_SEEDS,
    UCV_CONFIG as EXPERIMENT_19_UCV_CONFIG,
)
from experiments.leduc_poker.ucv_three_arm_15m_simplification.config import (
    PRODUCTION_SEEDS as EXPERIMENT_22_SEEDS,
)


EXPERIMENT_ID = 23
EXPERIMENT_NAME = "ucv_24h_stability_development"
GAME_NAME = "leduc_poker"

FULL_ADAPTIVE = "full_adaptive_ucv"
FAST_CORE = "fast_core_beta1_two_critics"
NONPREDICTIVE_CORE = "nonpredictive_fast_core"
STABLE_NONPREDICTIVE_CORE = "stable_nonpredictive_core"

VARIANT_ORDER = (
    FULL_ADAPTIVE,
    FAST_CORE,
    NONPREDICTIVE_CORE,
    STABLE_NONPREDICTIVE_CORE,
)
VARIANTS = {
    FULL_ADAPTIVE: {
        "variant_label": "Original UCV",
        "overrides": {},
        "purpose": "Reference architecture: adaptive beta, three critics, gated predictor.",
    },
    FAST_CORE: {
        "variant_label": "Fast core",
        "overrides": {
            "fixed_control_variate_beta": 1.0,
            "q_ensemble_size": 2,
        },
        "purpose": "Joint speed simplification while retaining calibrated sampling and predictor.",
    },
    NONPREDICTIVE_CORE: {
        "variant_label": "Non-predictive fast core",
        "overrides": {
            "fixed_control_variate_beta": 1.0,
            "q_ensemble_size": 2,
            "use_instantaneous_predictor": False,
            "force_prediction_gate_zero": True,
        },
        "purpose": "Remove the low-activation instantaneous predictor from the fast core.",
    },
    STABLE_NONPREDICTIVE_CORE: {
        "variant_label": "Stable non-predictive core",
        "overrides": {
            "fixed_control_variate_beta": 1.0,
            "q_ensemble_size": 2,
            "use_instantaneous_predictor": False,
            "force_prediction_gate_zero": True,
            "anneal_start_nodes": 15_000_000,
            "anneal_end_nodes": 45_000_000,
            "anneal_final_learning_rate": 1e-4,
            "regret_policy_gradient_clip_norm": 5.0,
        },
        "purpose": "Test whether late annealing and clipping suppress long-horizon volatility.",
    },
}


def _derived_seed(index: int) -> int:
    namespace = f"ucv-24h-stability-development-{index}"
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 900_000 + 100_000


# Fresh development labels fixed before any Experiment 23 result is observed.
PRODUCTION_SEEDS = tuple(_derived_seed(index) for index in range(4))
SMOKE_SEEDS = (0,)

TARGET_ACTIVE_HOURS = 24
CHECKPOINT_INTERVAL_HOURS = 2
CHECKPOINT_TARGET_HOURS = tuple(range(2, TARGET_ACTIVE_HOURS + 1, 2))
TARGET_NODES = 15_000_000
MAX_ITERATIONS = 500
SMOKE_TIME_SECONDS = (0.0, 0.001, 0.002)
SMOKE_TARGET_NODES = 100

BASE_CONFIG = deepcopy(EXPERIMENT_19_UCV_CONFIG)
BASE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        "fixed_control_variate_beta": None,
        "force_prediction_gate_zero": False,
        "use_instantaneous_predictor": True,
        "use_residual_calibration": True,
        "regret_policy_gradient_clip_norm": None,
        "anneal_start_nodes": None,
        "anneal_end_nodes": None,
        "anneal_final_learning_rate": None,
    }
)


def variant_config(variant_id: str) -> dict:
    if variant_id not in VARIANTS:
        raise ValueError(f"Unknown Experiment 23 variant: {variant_id}")
    config = deepcopy(BASE_CONFIG)
    config.update(VARIANTS[variant_id]["overrides"])
    return config


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    if smoke:
        time_rows = tuple(
            {
                "checkpoint_id": f"smoke_time_{index:02d}",
                "checkpoint_type": "active_time",
                "target_active_seconds": float(seconds),
                "target_active_hours": float(seconds) / 3600.0,
                "target_nodes": None,
            }
            for index, seconds in enumerate(SMOKE_TIME_SECONDS, start=1)
        )
        node_target = SMOKE_TARGET_NODES
    else:
        time_rows = tuple(
            {
                "checkpoint_id": f"time_{hours:02d}h",
                "checkpoint_type": "active_time",
                "target_active_seconds": float(hours * 3600),
                "target_active_hours": float(hours),
                "target_nodes": None,
            }
            for hours in CHECKPOINT_TARGET_HOURS
        )
        node_target = TARGET_NODES
    return time_rows + (
        {
            "checkpoint_id": "node_15m" if not smoke else "smoke_node_target",
            "checkpoint_type": "nodes",
            "target_active_seconds": None,
            "target_active_hours": None,
            "target_nodes": int(node_target),
        },
    )


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (variant_id, int(seed))
        for variant_id in VARIANT_ORDER
        for seed in seeds
    )


def validate_contract(
    *, seeds: Sequence[int], schedule: Sequence[Mapping], smoke: bool
) -> None:
    observed_seeds = tuple(int(seed) for seed in seeds)
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected_seeds}, "
            f"got {observed_seeds}"
        )
    if not smoke:
        forbidden = set(HELDOUT_SEEDS).union(EXPERIMENT_22_SEEDS)
        if set(observed_seeds).intersection(forbidden):
            raise ValueError("Experiment 23 seeds must be fresh development labels")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 23 contract")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "variant_order": list(VARIANT_ORDER),
        "variants": VARIANTS,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_active_hours": TARGET_ACTIVE_HOURS,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "target_nodes": TARGET_NODES,
        "selection_status": "development_only",
        "seed_derivation": (
            "sha256('ucv-24h-stability-development-{index}'), first 32 bits "
            "modulo 900000 plus 100000"
        ),
    }


__all__ = [
    "BASE_CONFIG",
    "CHECKPOINT_TARGET_HOURS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "FAST_CORE",
    "FULL_ADAPTIVE",
    "GAME_NAME",
    "MAX_ITERATIONS",
    "NONPREDICTIVE_CORE",
    "PRODUCTION_SEEDS",
    "SMOKE_SEEDS",
    "STABLE_NONPREDICTIVE_CORE",
    "TARGET_ACTIVE_HOURS",
    "TARGET_NODES",
    "VARIANTS",
    "VARIANT_ORDER",
    "checkpoint_schedule",
    "contract_manifest",
    "task_schedule",
    "validate_contract",
    "variant_config",
]
