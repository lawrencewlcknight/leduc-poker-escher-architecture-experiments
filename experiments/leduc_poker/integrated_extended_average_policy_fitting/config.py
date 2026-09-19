"""Frozen contract for Experiment 43's integrated extended policy fit."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_CONFIG as EXPERIMENT_29_CONFIG,
    PRODUCTION_SEEDS,
    checkpoint_schedule as experiment_29_checkpoint_schedule,
    training_state_checkpoint_ids as experiment_29_training_state_checkpoint_ids,
)
from experiments.leduc_poker.grouped_wide_policy_confirmation.config import (
    HISTORICAL_ALGORITHM_LABELS,
    HISTORICAL_ALGORITHM_ORDER,
    REFERENCE_EXPERIMENT_29_MANIFEST_SHA256,
    REFERENCE_EXPERIMENT_29_METRICS_SHA256,
    REFERENCE_EXPERIMENT_29_RUN_ID,
)
from unbiased_escher.policy_distillation import SOFT_TARGET_CROSS_ENTROPY


EXPERIMENT_ID = 43
EXPERIMENT_NAME = "integrated_extended_average_policy_fitting"
GAME_NAME = "leduc_poker"

CANDIDATE_ID = "extended_fit_ucv"
CANDIDATE_LABEL = "UCV-ESCHER (integrated 1,700-update refinement)"
ORDINARY_POLICY_ID = "ordinary_policy"
ORDINARY_POLICY_LABEL = "Contemporaneous Experiment 29 policy fit"
EMPIRICAL_POLICY_ID = "empirical_reservoir_policy"
EMPIRICAL_POLICY_LABEL = "Empirical reservoir policy"
EXACT_POLICY_ID = "exact_tabular_average"
EXACT_POLICY_LABEL = "Exact tabular average"

SMOKE_SEEDS = (0,)
TARGET_ACTIVE_HOURS = 36
TARGET_NODES = 15_000_000
REFINEMENT_UPDATES = 1_700
REFINEMENT_LEARNING_RATE = 3e-4
REFINEMENT_GRADIENT_CLIP_NORM = 10.0
FINAL_AUDIT_UPDATES = (400, 800, 1_200, 1_600, 1_700)
SMOKE_REFINEMENT_UPDATES = 2
SMOKE_FINAL_AUDIT_UPDATES = (1, 2)

CANDIDATE_CONFIG = deepcopy(EXPERIMENT_29_CONFIG)
CANDIDATE_CONFIG.update(
    {
        "average_policy_loss": SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
        "integrated_refinement_updates": REFINEMENT_UPDATES,
        "integrated_refinement_learning_rate": REFINEMENT_LEARNING_RATE,
        "integrated_refinement_gradient_clip_norm": REFINEMENT_GRADIENT_CLIP_NORM,
    }
)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    return tuple(dict(row) for row in experiment_29_checkpoint_schedule(smoke=smoke))


def training_state_checkpoint_ids(*, smoke: bool = False) -> tuple[str, ...]:
    return tuple(experiment_29_training_state_checkpoint_ids(smoke=smoke))


def final_active_checkpoint_id(*, smoke: bool = False) -> str:
    active = [
        row for row in checkpoint_schedule(smoke=smoke)
        if row["checkpoint_type"] == "active_time"
    ]
    if not active:
        raise ValueError("Experiment 43 schedule has no active-time checkpoint")
    return str(active[-1]["checkpoint_id"])


def refinement_schedule(*, smoke: bool = False, final_checkpoint: bool = False) -> tuple[int, ...]:
    maximum = SMOKE_REFINEMENT_UPDATES if smoke else REFINEMENT_UPDATES
    audit = SMOKE_FINAL_AUDIT_UPDATES if smoke else FINAL_AUDIT_UPDATES
    return (0,) + (tuple(audit) if final_checkpoint else (maximum,))


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[tuple[str, int], ...]:
    return tuple((CANDIDATE_ID, int(seed)) for seed in seeds)


def validate_contract(*, seeds: Sequence[int], schedule: Sequence[Mapping], smoke: bool) -> None:
    observed = tuple(int(seed) for seed in seeds)
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed != expected:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected}, got {observed}"
        )
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 43 contract")
    required = {
        "average_policy_loss": SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
        "ave_policy_network_train_steps": 5_000,
        "fixed_control_variate_beta": 1.0,
        "q_ensemble_size": 2,
        "critic_target_average_window": 4,
    }
    for key, value in required.items():
        if CANDIDATE_CONFIG.get(key) != value:
            raise ValueError(f"Frozen Experiment 43 field {key} changed")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID,
        "candidate_label": CANDIDATE_LABEL,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_active_hours": TARGET_ACTIVE_HOURS,
        "target_nodes": TARGET_NODES,
        "candidate_config": CANDIDATE_CONFIG,
        "ordinary_fit": {
            "loss": SOFT_TARGET_CROSS_ENTROPY,
            "minibatch_updates": 5_000,
            "role": "shared-trajectory internal control",
        },
        "refinement": {
            "updates": REFINEMENT_UPDATES,
            "learning_rate": REFINEMENT_LEARNING_RATE,
            "gradient_clip_norm": REFINEMENT_GRADIENT_CLIP_NORM,
            "target": "iteration-weighted mean action distribution per observed information state",
            "warm_start": "ordinary fitted network",
            "fresh_adam_at_each_checkpoint": True,
            "excluded_from_active_training_time": True,
            "training_requires_game_tree": False,
        },
        "final_audit_updates": list(FINAL_AUDIT_UPDATES),
        "historical_comparator_source": {
            "experiment_id": 29,
            "run_id": REFERENCE_EXPERIMENT_29_RUN_ID,
            "algorithm_ids": list(HISTORICAL_ALGORITHM_ORDER),
            "comparison_status": "same-seed frozen historical comparators",
        },
        "primary_outcomes": [
            "36-hour refined-policy exploitability",
            "24--36-hour refined-policy mean exploitability",
            "paired refined-minus-ordinary exploitability",
            "paired refined-minus-Experiment-29 and refined-minus-Deep-CFR exploitability",
        ],
        "evidence_status": "post-selection paired development evidence",
        "selection_source": "Experiment 41 selected the fixed 1,700-update horizon",
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "checkpoint_schedule", "contract_manifest", "final_active_checkpoint_id",
    "refinement_schedule", "task_schedule", "training_state_checkpoint_ids",
    "validate_contract",
]
