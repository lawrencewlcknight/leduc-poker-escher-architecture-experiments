"""Frozen contract for Experiment 29's promoted-candidate comparison."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.leduc_poker.deep_cfr_ucv_36h_plateau.config import (
    PRODUCTION_SEEDS as EXPERIMENT_21_SEEDS,
)
from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    checkpoint_schedule as experiment_25_checkpoint_schedule,
    training_state_checkpoint_ids as experiment_25_training_state_checkpoint_ids,
    variant_config as experiment_25_variant_config,
)
from unbiased_escher.policy_distillation import SOFT_TARGET_CROSS_ENTROPY


EXPERIMENT_ID = 29
EXPERIMENT_NAME = "promoted_ucv_cross_entropy_36h"
GAME_NAME = "leduc_poker"

CANDIDATE_ID = "promoted_ucv_cross_entropy"
CANDIDATE_LABEL = "Revised UCV (averaged critic + CE policy)"
REFERENCE_ALGORITHM_IDS = (
    "deep_cfr",
    "unbiased_control_variate_escher",
    "selected_nonpredictive_ucv",
)
REFERENCE_EXPERIMENT_21_RUN_ID = "exp21-36h-20260830-141641"
REFERENCE_EXPERIMENT_24_RUN_ID = "exp24-selected-20260905-222025"
REFERENCE_EXPERIMENT_21_METRICS_SHA256 = (
    "69793945ace65fdb430902ce9f6762216e5ed5e7a0a0bfc5286b49312a81ca57"
)
REFERENCE_EXPERIMENT_24_METRICS_SHA256 = (
    "4900889ac6c1afd25aaeab9196cc8bdb03af4c5470a5a851e68c8e755f02cd77"
)

PRODUCTION_SEEDS = tuple(int(seed) for seed in EXPERIMENT_21_SEEDS)
SMOKE_SEEDS = (0,)
TARGET_ACTIVE_HOURS = 36
CHECKPOINT_INTERVAL_HOURS = 2
CHECKPOINT_TARGET_HOURS = tuple(range(2, TARGET_ACTIVE_HOURS + 1, 2))
TARGET_NODES = 15_000_000
MAX_ITERATIONS = 800

CANDIDATE_CONFIG = deepcopy(experiment_25_variant_config(AVERAGED_TARGET_ONLY))
CANDIDATE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        "average_policy_loss": SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
    }
)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    return tuple(
        dict(row) for row in experiment_25_checkpoint_schedule(smoke=smoke)
    )


def training_state_checkpoint_ids(*, smoke: bool = False) -> tuple[str, ...]:
    return tuple(experiment_25_training_state_checkpoint_ids(smoke=smoke))


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple((CANDIDATE_ID, int(seed)) for seed in seeds)


def validate_contract(
    *, seeds: Sequence[int], schedule: Sequence[Mapping], smoke: bool
) -> None:
    observed = tuple(int(seed) for seed in seeds)
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed != expected:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected}, "
            f"got {observed}"
        )
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 29 contract")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID,
        "candidate_label": CANDIDATE_LABEL,
        "candidate_config": CANDIDATE_CONFIG,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_active_hours": TARGET_ACTIVE_HOURS,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "training_state_hours": [24, 36],
        "target_nodes": TARGET_NODES,
        "reference_experiment_21_run_id": REFERENCE_EXPERIMENT_21_RUN_ID,
        "reference_experiment_24_run_id": REFERENCE_EXPERIMENT_24_RUN_ID,
        "primary_outcomes": [
            "24--36-hour mean exploitability",
            "36-hour exploitability",
        ],
        "evidence_status": "post_selection_paired_development_evidence",
        "selection_sources": ["Experiment 23", "Experiment 25", "Experiment 27"],
    }


__all__ = [
    "CANDIDATE_CONFIG",
    "CANDIDATE_ID",
    "CANDIDATE_LABEL",
    "CHECKPOINT_TARGET_HOURS",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "MAX_ITERATIONS",
    "PRODUCTION_SEEDS",
    "REFERENCE_ALGORITHM_IDS",
    "REFERENCE_EXPERIMENT_21_METRICS_SHA256",
    "REFERENCE_EXPERIMENT_21_RUN_ID",
    "REFERENCE_EXPERIMENT_24_METRICS_SHA256",
    "REFERENCE_EXPERIMENT_24_RUN_ID",
    "SMOKE_SEEDS",
    "TARGET_ACTIVE_HOURS",
    "TARGET_NODES",
    "checkpoint_schedule",
    "contract_manifest",
    "task_schedule",
    "training_state_checkpoint_ids",
    "validate_contract",
]
