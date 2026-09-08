"""Frozen contract for Experiment 27's offline policy-distillation study."""

from __future__ import annotations

from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    PRODUCTION_SEEDS as EXPERIMENT_25_SEEDS,
)


EXPERIMENT_ID = 27
EXPERIMENT_NAME = "average_policy_redistillation"
SOURCE_EXPERIMENT_ID = 25
SOURCE_VARIANT_ID = AVERAGED_TARGET_ONLY
SOURCE_CHECKPOINTS = ("time_24h", "time_36h")
PRODUCTION_SEEDS = EXPERIMENT_25_SEEDS
SMOKE_SEEDS = (0,)

FIT_REPLICATES = 3
SMOKE_FIT_REPLICATES = 1
TRAIN_STEPS = 5_000
SMOKE_TRAIN_STEPS = 3
VALIDATION_SAMPLES = 50_000
SMOKE_VALIDATION_SAMPLES = 16

RESET_MSE = "reset_mse"
RESET_CROSS_ENTROPY = "reset_cross_entropy"
WARM_MSE = "warm_mse"
WARM_CROSS_ENTROPY = "warm_cross_entropy"
ORACLE_CROSS_ENTROPY = "oracle_cross_entropy"

FACTORIAL_ARMS = (RESET_MSE, RESET_CROSS_ENTROPY, WARM_MSE, WARM_CROSS_ENTROPY)
ARM_ORDER = FACTORIAL_ARMS + (ORACLE_CROSS_ENTROPY,)
ARMS = {
    RESET_MSE: {
        "label": "Weighted MSE, reset",
        "loss": "mse",
        "warm_start": False,
        "oracle_targets": False,
    },
    RESET_CROSS_ENTROPY: {
        "label": "Weighted cross-entropy, reset",
        "loss": "cross_entropy",
        "warm_start": False,
        "oracle_targets": False,
    },
    WARM_MSE: {
        "label": "Weighted MSE, warm start",
        "loss": "mse",
        "warm_start": True,
        "oracle_targets": False,
    },
    WARM_CROSS_ENTROPY: {
        "label": "Weighted cross-entropy, warm start",
        "loss": "cross_entropy",
        "warm_start": True,
        "oracle_targets": False,
    },
    ORACLE_CROSS_ENTROPY: {
        "label": "Exact-table cross-entropy (diagnostic)",
        "loss": "cross_entropy",
        "warm_start": False,
        "oracle_targets": True,
    },
}


def task_schedule(seeds=PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def fit_replicates(*, smoke: bool) -> tuple[int, ...]:
    count = SMOKE_FIT_REPLICATES if smoke else FIT_REPLICATES
    return tuple(range(count))


def validate_contract(*, seeds, smoke: bool) -> None:
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    observed = tuple(int(seed) for seed in seeds)
    if observed != expected:
        raise ValueError(f"Expected seeds {expected}, got {observed}")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_variant_id": SOURCE_VARIANT_ID,
        "source_checkpoints": list(SOURCE_CHECKPOINTS),
        "production_seeds": list(PRODUCTION_SEEDS),
        "fit_replicates": FIT_REPLICATES,
        "train_steps": TRAIN_STEPS,
        "factorial_arms": list(FACTORIAL_ARMS),
        "diagnostic_arm": ORACLE_CROSS_ENTROPY,
        "primary_outcome": "time_36h neural-minus-exact exploitability gap",
        "evidence_status": "paired offline development evidence",
    }


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "FACTORIAL_ARMS",
    "FIT_REPLICATES",
    "ORACLE_CROSS_ENTROPY",
    "PRODUCTION_SEEDS",
    "RESET_CROSS_ENTROPY",
    "RESET_MSE",
    "SMOKE_SEEDS",
    "SOURCE_CHECKPOINTS",
    "SOURCE_VARIANT_ID",
    "TRAIN_STEPS",
    "VALIDATION_SAMPLES",
    "WARM_CROSS_ENTROPY",
    "WARM_MSE",
    "contract_manifest",
    "fit_replicates",
    "task_schedule",
    "validate_contract",
]
