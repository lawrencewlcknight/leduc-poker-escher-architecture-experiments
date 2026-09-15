"""Frozen contract for Experiment 34's average-policy fitting study."""

from __future__ import annotations

from itertools import product
from typing import Sequence

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_ID as SOURCE_CANDIDATE_ID,
    PRODUCTION_SEEDS as EXPERIMENT_29_SEEDS,
)


EXPERIMENT_ID = 34
EXPERIMENT_NAME = "rao_blackwellised_policy_distillation"
GAME_NAME = "leduc_poker"
SOURCE_EXPERIMENT_ID = 29
DEVELOPMENT_CHECKPOINT = "time_24h"
VALIDATION_CHECKPOINT = "time_36h"
PRODUCTION_SEEDS = tuple(int(seed) for seed in EXPERIMENT_29_SEEDS)
SMOKE_SEEDS = (0,)

ROW_TARGETS = "individual_replay_rows"
GROUPED_TARGETS = "grouped_information_set_targets"
TARGET_ORDER = (ROW_TARGETS, GROUPED_TARGETS)
TARGET_LABELS = {
    ROW_TARGETS: "Individual replay rows",
    GROUPED_TARGETS: "Grouped information-set targets",
}

CURRENT_SHARED = "current_shared_3x64"
WIDE_SHARED = "parameter_matched_shared_3x136"
ROUTED_EXPERTS = "player_round_experts_4x3x64"
ARCHITECTURE_ORDER = (CURRENT_SHARED, WIDE_SHARED, ROUTED_EXPERTS)
ARCHITECTURE_LABELS = {
    CURRENT_SHARED: "Current shared 3x64",
    WIDE_SHARED: "Parameter-matched shared 3x136",
    ROUTED_EXPERTS: "Player-by-round experts",
}
ARCHITECTURE_LAYERS = {
    CURRENT_SHARED: (64, 64, 64),
    WIDE_SHARED: (136, 136, 136),
    ROUTED_EXPERTS: (64, 64, 64),
}


def arm_id(architecture_id: str, target_id: str) -> str:
    return f"{architecture_id}__{target_id}"


ARM_ORDER = tuple(
    arm_id(architecture_id, target_id)
    for architecture_id, target_id in product(ARCHITECTURE_ORDER, TARGET_ORDER)
)
ARM_METADATA = {
    arm_id(architecture_id, target_id): {
        "architecture_id": architecture_id,
        "target_id": target_id,
        "arm_label": (
            f"{ARCHITECTURE_LABELS[architecture_id]} + {TARGET_LABELS[target_id]}"
        ),
    }
    for architecture_id, target_id in product(ARCHITECTURE_ORDER, TARGET_ORDER)
}
BASELINE_ARM = arm_id(CURRENT_SHARED, ROW_TARGETS)
GROUPED_ARM_BY_ARCHITECTURE = {
    architecture_id: arm_id(architecture_id, GROUPED_TARGETS)
    for architecture_id in ARCHITECTURE_ORDER
}

LEARNING_RATES = (3e-4, 1e-3, 3e-3)
TRAINING_BUDGETS = (5_000, 20_000)
CONTRACT_BASELINE_LEARNING_RATE = 1e-3
CONTRACT_BASELINE_TRAINING_BUDGET = 5_000
FIT_REPLICATES = 3
VALIDATION_SAMPLES = 50_000

SMOKE_LEARNING_RATES = (1e-3,)
SMOKE_TRAINING_BUDGETS = (2, 4)
SMOKE_FIT_REPLICATES = 1
SMOKE_VALIDATION_SAMPLES = 16

PLAYER_ROUND_GROUPS = 4
PROMOTION_RELATIVE_GAP_REDUCTION = 0.15
PROMOTION_REQUIRED_IMPROVED_SEEDS = 4
PROMOTION_MAX_SINGLE_SEED_DETERIORATION = 0.003
ARCHIVED_REFERENCE_EXPLOITABILITY = 0.054274


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def learning_rates(*, smoke: bool) -> tuple[float, ...]:
    return SMOKE_LEARNING_RATES if smoke else LEARNING_RATES


def training_budgets(*, smoke: bool) -> tuple[int, ...]:
    return SMOKE_TRAINING_BUDGETS if smoke else TRAINING_BUDGETS


def fit_replicates(*, smoke: bool) -> tuple[int, ...]:
    count = SMOKE_FIT_REPLICATES if smoke else FIT_REPLICATES
    return tuple(range(count))


def validation_samples(*, smoke: bool) -> int:
    return SMOKE_VALIDATION_SAMPLES if smoke else VALIDATION_SAMPLES


def validate_contract(*, seeds: Sequence[int], smoke: bool) -> None:
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    observed = tuple(int(seed) for seed in seeds)
    if observed != expected:
        raise ValueError(f"Expected seeds {expected}, got {observed}")
    if BASELINE_ARM not in ARM_ORDER or len(ARM_ORDER) != 6:
        raise ValueError("Experiment 34 requires the frozen 2x3 factorial")
    if TRAINING_BUDGETS[0] != CONTRACT_BASELINE_TRAINING_BUDGET:
        raise ValueError("The current-contract baseline must be evaluated at 5,000 steps")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "development_checkpoint": DEVELOPMENT_CHECKPOINT,
        "validation_checkpoint": VALIDATION_CHECKPOINT,
        "production_seeds": list(PRODUCTION_SEEDS),
        "architectures": list(ARCHITECTURE_ORDER),
        "target_representations": list(TARGET_ORDER),
        "arms": list(ARM_ORDER),
        "learning_rates": list(LEARNING_RATES),
        "training_budgets": list(TRAINING_BUDGETS),
        "fit_replicates": FIT_REPLICATES,
        "primary_outcome": "time_36h exact exploitability",
        "secondary_outcome": "time_36h neural-minus-exact exploitability gap",
        "selection_rule": (
            "per arm, minimise the Experiment 29 five-source-seed mean exact "
            "exploitability at time_24h; then worst-source-seed exploitability, "
            "then smaller training budget, then learning rate"
        ),
        "baseline_rule": (
            "the current shared row-wise baseline remains fixed at learning rate "
            "0.001 and 5,000 updates"
        ),
        "exact_teacher_role": "Leduc-only diagnostic; never a deployable candidate",
        "inferential_unit": "Experiment 29 source trajectory",
        "evidence_status": "paired offline architecture-development evidence",
        "training_effect": "average-policy fitting only; UCV trajectories are frozen",
        "promotion_rule": {
            "minimum_relative_gap_reduction": PROMOTION_RELATIVE_GAP_REDUCTION,
            "required_improved_source_seeds": PROMOTION_REQUIRED_IMPROVED_SEEDS,
            "maximum_single_seed_deterioration": PROMOTION_MAX_SINGLE_SEED_DETERIORATION,
            "must_beat_archived_experiment_29_mean": ARCHIVED_REFERENCE_EXPLOITABILITY,
        },
    }


__all__ = [
    "ARCHITECTURE_LABELS",
    "ARCHITECTURE_LAYERS",
    "ARCHITECTURE_ORDER",
    "ARM_METADATA",
    "ARM_ORDER",
    "BASELINE_ARM",
    "CURRENT_SHARED",
    "DEVELOPMENT_CHECKPOINT",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "GROUPED_ARM_BY_ARCHITECTURE",
    "GROUPED_TARGETS",
    "PLAYER_ROUND_GROUPS",
    "PRODUCTION_SEEDS",
    "ROUTED_EXPERTS",
    "ROW_TARGETS",
    "SMOKE_SEEDS",
    "TARGET_LABELS",
    "VALIDATION_CHECKPOINT",
    "WIDE_SHARED",
    "arm_id",
    "contract_manifest",
    "fit_replicates",
    "learning_rates",
    "task_schedule",
    "training_budgets",
    "validate_contract",
    "validation_samples",
]
