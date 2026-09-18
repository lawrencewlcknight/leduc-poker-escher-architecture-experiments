"""Frozen Experiment 42 contract; intentionally standard-library only."""

from __future__ import annotations


EXPERIMENT_ID = 42
EXPERIMENT_NAME = "average_policy_rowwise_continuation"
GAME_NAME = "leduc_poker"
SOURCE_EXPERIMENT_ID = 29
SOURCE_CHECKPOINT = "time_36h"
REFERENCE_EXPERIMENT_ID = 41
PRODUCTION_SEEDS = (104729, 130363, 155921, 181081, 205759)
SMOKE_SEEDS = (0,)

EQUAL_EXAMPLE_ARM = "rowwise_equal_examples"
EQUAL_UPDATE_ARM = "rowwise_equal_updates"
ARM_IDS = (EQUAL_EXAMPLE_ARM, EQUAL_UPDATE_ARM)
GROUPED_REFERENCE_ARM = "grouped_full_batch_reference"

BATCH_SIZE = 2048
LEARNING_RATE = 3e-4
GRADIENT_CLIP_NORM = 10.0
FINAL_EQUIVALENT_UPDATE = 1700
DIAGNOSTIC_EQUIVALENT_UPDATES = (0, 400, 800, 1200, 1600, 1700)
PRACTICAL_EQUIVALENCE_MARGIN = 0.0025

# Frozen from Experiment 41's validated worker manifest.  These counts define
# equal neural-network example exposure without repeating np.unique here.
UNIQUE_INFORMATION_STATES = {
    104729: 928,
    130363: 927,
    155921: 933,
    181081: 930,
    205759: 928,
}
SOURCE_SHA256 = {
    104729: "fe0612f67fc988ea940a9052397f342ca0b4e1930e67424e9835aa12a85f5d9b",
    130363: "7920794fd0f147d4063b8fe6898ea44787729e454cce4e68e2c9be0e036a068b",
    155921: "736a5689643b7f637183b945e3cc4b288e86dbfa2600f1d1f47df9583e55c2b4",
    181081: "032de510a11fb397153df6fdbec832b821de7167fb69419da9e015e9052eb6f6",
    205759: "216c7607befeee9ced7b628eb43a1c7ea74a69babdbf2bfef99fe505d0f4a357",
}


def runtime_config(*, seed: int, smoke: bool = False) -> dict:
    if smoke:
        unique_count = 12
        diagnostic = (0, 1, 2, 4)
        batch_size = 8
    else:
        if int(seed) not in UNIQUE_INFORMATION_STATES:
            raise ValueError(f"Unknown production seed {seed}")
        unique_count = UNIQUE_INFORMATION_STATES[int(seed)]
        diagnostic = DIAGNOSTIC_EQUIVALENT_UPDATES
        batch_size = BATCH_SIZE
    return {
        "unique_information_states": unique_count,
        "diagnostic_equivalent_updates": diagnostic,
        "required_checkpoint_updates": diagnostic[1:],
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "equal_example_budget": unique_count * max(diagnostic),
        "equal_update_budget": max(diagnostic),
    }


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "reference_experiment_id": REFERENCE_EXPERIMENT_ID,
        "source_seeds": list(PRODUCTION_SEEDS),
        "arm_ids": list(ARM_IDS),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "final_equivalent_update": FINAL_EQUIVALENT_UPDATE,
        "diagnostic_equivalent_updates": list(DIAGNOSTIC_EQUIVALENT_UPDATES),
        "unique_information_states_by_seed": dict(UNIQUE_INFORMATION_STATES),
        "primary_arm": EQUAL_EXAMPLE_ARM,
        "primary_comparison": GROUPED_REFERENCE_ARM,
        "practical_equivalence_margin": PRACTICAL_EQUIVALENCE_MARGIN,
        "initialisation": "warm_start_from_frozen_experiment_29_policy",
        "optimizer": "fresh_adam_continued_without_restart_per_arm",
        "objective": "ordinary_row_sampled_iteration_weighted_soft_target_cross_entropy",
        "sampling": "uniform_replay_rows_with_replacement",
        "grouping_in_training_path": False,
        "training_requires_game_tree": False,
        "exact_exploitability_role": "Leduc_only_post_hoc_diagnostic",
        "checkpoint_selection": "none_fixed_horizons",
        "evidence_status": "paired_exploratory_scalability_screen",
    }


__all__ = [
    "ARM_IDS",
    "BATCH_SIZE",
    "DIAGNOSTIC_EQUIVALENT_UPDATES",
    "EQUAL_EXAMPLE_ARM",
    "EQUAL_UPDATE_ARM",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "GROUPED_REFERENCE_ARM",
    "PRACTICAL_EQUIVALENCE_MARGIN",
    "PRODUCTION_SEEDS",
    "SMOKE_SEEDS",
    "SOURCE_CHECKPOINT",
    "SOURCE_SHA256",
    "UNIQUE_INFORMATION_STATES",
    "contract_manifest",
    "runtime_config",
]
