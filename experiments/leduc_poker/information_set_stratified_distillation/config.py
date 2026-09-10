"""Frozen contract for Experiment 28's information-set sampling study."""

from __future__ import annotations

from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    PRODUCTION_SEEDS as EXPERIMENT_25_SEEDS,
)


EXPERIMENT_ID = 28
EXPERIMENT_NAME = "information_set_stratified_distillation"
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

EMPIRICAL_INFORMATION_SET = "empirical_information_set"
SQRT_INFORMATION_SET = "sqrt_information_set"
UNIFORM_INFORMATION_SET = "uniform_information_set"
ARM_ORDER = (
    EMPIRICAL_INFORMATION_SET,
    SQRT_INFORMATION_SET,
    UNIFORM_INFORMATION_SET,
)
ARMS = {
    EMPIRICAL_INFORMATION_SET: {
        "label": "Empirical-mass sampling",
        "information_set_exponent": 1.0,
    },
    SQRT_INFORMATION_SET: {
        "label": "Square-root information-set sampling",
        "information_set_exponent": 0.5,
    },
    UNIFORM_INFORMATION_SET: {
        "label": "Uniform information-set sampling",
        "information_set_exponent": 0.0,
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
        "loss": "iteration-weighted soft-target cross-entropy",
        "sampling_with_replacement": True,
        "arms": {
            arm: {
                **ARMS[arm],
                "importance_correction": "p_information_set / q_information_set",
            }
            for arm in ARM_ORDER
        },
        "primary_outcome": "time_36h neural-minus-exact exploitability gap",
        "inferential_unit": "Experiment 25 source trajectory",
        "evidence_status": "paired offline development evidence",
    }


__all__ = [
    "ARMS",
    "ARM_ORDER",
    "EMPIRICAL_INFORMATION_SET",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "FIT_REPLICATES",
    "PRODUCTION_SEEDS",
    "SMOKE_SEEDS",
    "SOURCE_CHECKPOINTS",
    "SOURCE_VARIANT_ID",
    "SQRT_INFORMATION_SET",
    "TRAIN_STEPS",
    "UNIFORM_INFORMATION_SET",
    "VALIDATION_SAMPLES",
    "contract_manifest",
    "fit_replicates",
    "task_schedule",
    "validate_contract",
]
