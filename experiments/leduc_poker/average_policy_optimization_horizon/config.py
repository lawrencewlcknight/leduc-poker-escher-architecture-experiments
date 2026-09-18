"""Frozen Experiment 41 contract; intentionally standard-library only."""

from __future__ import annotations


EXPERIMENT_ID = 41
EXPERIMENT_NAME = "average_policy_optimization_horizon"
GAME_NAME = "leduc_poker"
SOURCE_EXPERIMENT_ID = 29
SOURCE_CHECKPOINT = "time_36h"
SOURCE_CANDIDATE_ID = "promoted_ucv_cross_entropy"
PRODUCTION_SEEDS = (104729, 130363, 155921, 181081, 205759)
SMOKE_SEEDS = (0,)

LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 10.0
REQUIRED_CHECKPOINT_UPDATES = (400, 800, 1200, 1600, 2000)
DIAGNOSTIC_UPDATES = tuple(range(0, 2001, 100))


def runtime_config(*, smoke: bool = False) -> dict:
    if smoke:
        diagnostic_updates = (0, 1, 2, 4)
        required_updates = (2, 4)
    else:
        diagnostic_updates = DIAGNOSTIC_UPDATES
        required_updates = REQUIRED_CHECKPOINT_UPDATES
    return {
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "diagnostic_updates": diagnostic_updates,
        "required_checkpoint_updates": required_updates,
        "maximum_updates": max(diagnostic_updates),
    }


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "source_seeds": list(PRODUCTION_SEEDS),
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "required_checkpoint_updates": list(REQUIRED_CHECKPOINT_UPDATES),
        "diagnostic_updates": list(DIAGNOSTIC_UPDATES),
        "initialisation": "warm_start_from_frozen_experiment_29_policy",
        "optimizer": "fresh_adam_continued_without_restart",
        "objective": (
            "uniform_full_batch_soft_target_cross_entropy_over_unique_replay_"
            "information_states"
        ),
        "target_construction": (
            "iteration_weighted_mean_action_distribution_within_information_state"
        ),
        "training_requires_game_tree": False,
        "exact_exploitability_role": "Leduc_only_post_hoc_diagnostic",
        "checkpoint_selection": "none_fixed_horizon_convergence_map",
        "evidence_status": "exploratory_optimization_horizon_evidence",
    }


__all__ = [
    "DIAGNOSTIC_UPDATES",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "PRODUCTION_SEEDS",
    "REQUIRED_CHECKPOINT_UPDATES",
    "SMOKE_SEEDS",
    "SOURCE_CHECKPOINT",
    "contract_manifest",
    "runtime_config",
]
