"""Frozen contract for Experiment 22's three-arm 15M-node study."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Mapping, Sequence

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    HELDOUT_SEEDS,
    UCV_CONFIG as EXPERIMENT_19_UCV_CONFIG,
)
from experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.config import (
    FIXED_BETA_ONE,
    FULL_EXPERIMENT_6,
    TWO_CROSS_FITTED_CRITICS,
    VARIANTS as EXPERIMENT_8_VARIANTS,
)


EXPERIMENT_ID = 22
EXPERIMENT_NAME = "ucv_three_arm_15m_simplification"
GAME_NAME = "leduc_poker"
TARGET_NODES = 15_000_000
MAX_ITERATIONS = 200

VARIANT_ORDER = (
    FULL_EXPERIMENT_6,
    FIXED_BETA_ONE,
    TWO_CROSS_FITTED_CRITICS,
)
VARIANTS = {variant_id: deepcopy(EXPERIMENT_8_VARIANTS[variant_id]) for variant_id in VARIANT_ORDER}


def _derived_seed(index: int) -> int:
    namespace = f"ucv-three-arm-15m-simplification-development-{index}"
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 900_000 + 100_000


# Generated before any Experiment 22 run by SHA-256 namespacing above.  These
# are fresh development labels, not the Experiment 19/21 held-out labels.
PRODUCTION_SEEDS = tuple(_derived_seed(index) for index in range(6))
SMOKE_SEEDS = (0,)

BASE_CONFIG = deepcopy(EXPERIMENT_19_UCV_CONFIG)
BASE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        "fixed_control_variate_beta": None,
        "force_prediction_gate_zero": False,
        "use_instantaneous_predictor": True,
        "use_residual_calibration": True,
    }
)

BETA_HISTOGRAM_EDGES = tuple(index / 20.0 for index in range(41))
NONINFERIORITY_MARGIN = 0.01


def variant_config(variant_id: str) -> dict:
    if variant_id not in VARIANTS:
        raise ValueError(f"Unknown Experiment 22 variant: {variant_id}")
    config = deepcopy(BASE_CONFIG)
    config.update(VARIANTS[variant_id]["overrides"])
    return config


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (variant_id, int(seed))
        for variant_id in VARIANT_ORDER
        for seed in seeds
    )


def validate_contract(
    *, seeds: Sequence[int], target_nodes: int, smoke: bool
) -> None:
    observed = tuple(int(seed) for seed in seeds)
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed != expected:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected}, got {observed}"
        )
    if int(target_nodes) != (100 if smoke else TARGET_NODES):
        raise ValueError("Node target differs from the frozen Experiment 22 contract")
    if not smoke and set(observed).intersection(HELDOUT_SEEDS):
        raise ValueError("Experiment 22 development seeds overlap held-out labels")
    if len(task_schedule(observed)) != len(VARIANT_ORDER) * len(observed):
        raise ValueError("Experiment 22 task schedule is incomplete")


def contract_manifest() -> Mapping:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "variant_order": list(VARIANT_ORDER),
        "variants": VARIANTS,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_nodes": TARGET_NODES,
        "max_iterations": MAX_ITERATIONS,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "seed_derivation": (
            "sha256('ucv-three-arm-15m-simplification-development-{index}'), "
            "first 32 bits modulo 900000 plus 100000"
        ),
    }


__all__ = [
    "BASE_CONFIG",
    "BETA_HISTOGRAM_EDGES",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "FIXED_BETA_ONE",
    "FULL_EXPERIMENT_6",
    "GAME_NAME",
    "MAX_ITERATIONS",
    "NONINFERIORITY_MARGIN",
    "PRODUCTION_SEEDS",
    "SMOKE_SEEDS",
    "TARGET_NODES",
    "TWO_CROSS_FITTED_CRITICS",
    "VARIANTS",
    "VARIANT_ORDER",
    "contract_manifest",
    "task_schedule",
    "validate_contract",
    "variant_config",
]
