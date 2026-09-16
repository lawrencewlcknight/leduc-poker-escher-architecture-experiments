"""Frozen contract for Experiment 35's fresh five-seed confirmation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Mapping, Sequence

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    HELDOUT_SEEDS,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_CONFIG as EXPERIMENT_29_CONFIG,
)
from experiments.leduc_poker.ucv_24h_stability_development.config import (
    PRODUCTION_SEEDS as EXPERIMENT_23_SEEDS,
)
from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    checkpoint_schedule as experiment_25_checkpoint_schedule,
    training_state_checkpoint_ids as experiment_25_training_state_checkpoint_ids,
    PRODUCTION_SEEDS as EXPERIMENT_25_SEEDS,
)
from experiments.leduc_poker.ucv_three_arm_15m_simplification.config import (
    PRODUCTION_SEEDS as EXPERIMENT_22_SEEDS,
)
from unbiased_escher.policy_distillation import (
    GROUPED_SOFT_TARGET_CROSS_ENTROPY,
)


EXPERIMENT_ID = 35
EXPERIMENT_NAME = "grouped_wide_policy_confirmation"
GAME_NAME = "leduc_poker"

CANDIDATE_ID = "grouped_wide_ucv"
CANDIDATE_LABEL = "UCV-ESCHER (grouped wide average policy)"
LEGACY_CONTROL_ID = "legacy_row_policy"
LEGACY_CONTROL_LABEL = "Contemporaneous row-wise 3x64 policy fit"
EMPIRICAL_POLICY_ID = "empirical_reservoir_policy"
EMPIRICAL_POLICY_LABEL = "Empirical reservoir policy"
EXACT_POLICY_ID = "exact_tabular_average"
EXACT_POLICY_LABEL = "Exact tabular average"

HISTORICAL_ALGORITHM_ORDER = (
    "deep_cfr",
    "unbiased_control_variate_escher",
    "selected_nonpredictive_ucv",
    "promoted_ucv_cross_entropy",
)
HISTORICAL_ALGORITHM_LABELS = {
    "deep_cfr": "Deep CFR",
    "unbiased_control_variate_escher": "Original UCV-ESCHER",
    "selected_nonpredictive_ucv": "Simplified UCV",
    "promoted_ucv_cross_entropy": "Revised UCV (averaged critic + CE policy)",
}
REFERENCE_EXPERIMENT_29_RUN_ID = "exp29-ce-20260912-171556"
REFERENCE_EXPERIMENT_29_METRICS_SHA256 = (
    "1bbb7ac3f3a867bb80de7b983759d365d61d48d4e43aab976a80072bc228aa8e"
)
REFERENCE_EXPERIMENT_29_MANIFEST_SHA256 = (
    "144d8967e3951d005aeb8da2b477191c0fa445fd3fbdc6ccea419eb0bae3be6c"
)


def _derived_seed(index: int) -> int:
    namespace = f"ucv-grouped-wide-fresh-confirmation-{index}"
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 900_000 + 100_000


# Generated and frozen before any Experiment 35 observation.  The validation
# below explicitly rejects all seed sets used in Experiments 19--29 and the
# earlier small-seed exploratory studies.
PRODUCTION_SEEDS = tuple(_derived_seed(index) for index in range(5))
SMOKE_SEEDS = (0,)
PREVIOUSLY_USED_SEEDS = frozenset(
    HELDOUT_SEEDS
    + EXPERIMENT_22_SEEDS
    + EXPERIMENT_23_SEEDS
    + EXPERIMENT_25_SEEDS
    + (0, 1, 2, 1234, 2025, 31415, 27182, 16180)
)

TARGET_ACTIVE_HOURS = 36
CHECKPOINT_INTERVAL_HOURS = 2
TARGET_NODES = 15_000_000
MAX_ITERATIONS = 800

POLICY_NETWORK_LAYERS = (136, 136, 136)
POLICY_LEARNING_RATE = 3e-3
POLICY_TRAINING_STEPS = 20_000
LEGACY_NETWORK_LAYERS = (64, 64, 64)
LEGACY_LEARNING_RATE = 1e-3
LEGACY_TRAINING_STEPS = 5_000

CANDIDATE_CONFIG = deepcopy(EXPERIMENT_29_CONFIG)
CANDIDATE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        "average_policy_loss": GROUPED_SOFT_TARGET_CROSS_ENTROPY,
        "average_policy_reset_each_fit": True,
        "average_policy_network_layers": POLICY_NETWORK_LAYERS,
        "average_policy_learning_rate": POLICY_LEARNING_RATE,
        "average_policy_train_steps": POLICY_TRAINING_STEPS,
        "paired_legacy_network_layers": LEGACY_NETWORK_LAYERS,
        "paired_legacy_learning_rate": LEGACY_LEARNING_RATE,
        "paired_legacy_train_steps": LEGACY_TRAINING_STEPS,
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
    if not smoke and set(observed).intersection(PREVIOUSLY_USED_SEEDS):
        raise ValueError("Experiment 35 requires previously unused seed labels")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 35 contract")
    required = {
        "average_policy_network_layers": POLICY_NETWORK_LAYERS,
        "average_policy_learning_rate": POLICY_LEARNING_RATE,
        "average_policy_train_steps": POLICY_TRAINING_STEPS,
        "average_policy_loss": GROUPED_SOFT_TARGET_CROSS_ENTROPY,
    }
    for key, expected_value in required.items():
        if CANDIDATE_CONFIG.get(key) != expected_value:
            raise ValueError(f"Frozen candidate field {key} changed")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID,
        "candidate_label": CANDIDATE_LABEL,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_active_hours": TARGET_ACTIVE_HOURS,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "training_state_hours": [24, 36],
        "target_nodes": TARGET_NODES,
        "candidate_config": CANDIDATE_CONFIG,
        "paired_control": {
            "id": LEGACY_CONTROL_ID,
            "network_layers": list(LEGACY_NETWORK_LAYERS),
            "learning_rate": LEGACY_LEARNING_RATE,
            "training_steps": LEGACY_TRAINING_STEPS,
            "source": "same frozen reservoir at each checkpoint",
            "excluded_from_active_training_time": True,
        },
        "primary_outcomes": [
            "36-hour candidate exploitability",
            "36-hour candidate neural-minus-exact distillation gap",
            "paired candidate-minus-legacy exploitability",
        ],
        "secondary_outcomes": [
            "24--36-hour mean exploitability",
            "late-window adjacent-checkpoint RMSSD",
            "empirical-reservoir approximation gap",
        ],
        "evidence_status": "fresh-seed confirmatory evidence",
        "selection_source": "Experiment 34",
        "historical_comparator_source": {
            "experiment_id": 29,
            "run_id": REFERENCE_EXPERIMENT_29_RUN_ID,
            "algorithm_ids": list(HISTORICAL_ALGORITHM_ORDER),
            "comparison_status": "independent-seed contextual comparison",
        },
        "training_effect": (
            "the paired legacy control is refitted from the same reservoir and "
            "does not alter or consume the training RNG stream"
        ),
    }


__all__ = [
    "CANDIDATE_CONFIG",
    "CANDIDATE_ID",
    "CANDIDATE_LABEL",
    "CHECKPOINT_INTERVAL_HOURS",
    "EMPIRICAL_POLICY_ID",
    "EMPIRICAL_POLICY_LABEL",
    "EXACT_POLICY_ID",
    "EXACT_POLICY_LABEL",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "HISTORICAL_ALGORITHM_LABELS",
    "HISTORICAL_ALGORITHM_ORDER",
    "LEGACY_CONTROL_ID",
    "LEGACY_CONTROL_LABEL",
    "POLICY_LEARNING_RATE",
    "POLICY_NETWORK_LAYERS",
    "POLICY_TRAINING_STEPS",
    "PREVIOUSLY_USED_SEEDS",
    "PRODUCTION_SEEDS",
    "REFERENCE_EXPERIMENT_29_MANIFEST_SHA256",
    "REFERENCE_EXPERIMENT_29_METRICS_SHA256",
    "REFERENCE_EXPERIMENT_29_RUN_ID",
    "SMOKE_SEEDS",
    "TARGET_ACTIVE_HOURS",
    "TARGET_NODES",
    "checkpoint_schedule",
    "contract_manifest",
    "task_schedule",
    "training_state_checkpoint_ids",
    "validate_contract",
]
