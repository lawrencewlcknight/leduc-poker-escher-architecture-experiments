"""Frozen contract for Experiment 24's post-selection 36-hour follow-up."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.leduc_poker.deep_cfr_ucv_36h_plateau.config import (
    PRODUCTION_SEEDS as EXPERIMENT_21_SEEDS,
    checkpoint_schedule as experiment_21_checkpoint_schedule,
)
from experiments.leduc_poker.ucv_24h_stability_development.config import (
    NONPREDICTIVE_CORE,
    variant_config as experiment_23_variant_config,
)


EXPERIMENT_ID = 24
EXPERIMENT_NAME = "selected_ucv_36h_confirmation"
GAME_NAME = "leduc_poker"

CANDIDATE_ID = "selected_nonpredictive_ucv"
CANDIDATE_LABEL = "Selected non-predictive UCV"
REFERENCE_ALGORITHM_IDS = ("deep_cfr", "unbiased_control_variate_escher")
REFERENCE_EXPERIMENT_ID = 21
REFERENCE_RUN_ID = "exp21-36h-20260830-141641"
REFERENCE_METRICS_SHA256 = (
    "69793945ace65fdb430902ce9f6762216e5ed5e7a0a0bfc5286b49312a81ca57"
)

# The Experiment 21 labels provide direct paired trajectories against its
# immutable Deep CFR and Original UCV archives. This is explicitly a
# post-selection follow-up, not a newly held-out benchmark.
PRODUCTION_SEEDS = tuple(int(seed) for seed in EXPERIMENT_21_SEEDS)
SMOKE_SEEDS = (0,)

TARGET_ACTIVE_HOURS = 36
CHECKPOINT_INTERVAL_HOURS = 2
CHECKPOINT_TARGET_HOURS = tuple(range(2, TARGET_ACTIVE_HOURS + 1, 2))
TARGET_NODES = 15_000_000
MAX_ITERATIONS = 800
SMOKE_TARGET_NODES = 100

CANDIDATE_CONFIG = deepcopy(experiment_23_variant_config(NONPREDICTIVE_CORE))
CANDIDATE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        # Make the rejected Experiment 23 stability package explicitly absent.
        "regret_policy_gradient_clip_norm": None,
        "anneal_start_nodes": None,
        "anneal_end_nodes": None,
        "anneal_final_learning_rate": None,
    }
)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    """Return time checkpoints followed by the independently captured node endpoint."""
    time_rows = tuple(
        {
            **dict(row),
            "checkpoint_type": "active_time",
            "target_nodes": None,
        }
        for row in experiment_21_checkpoint_schedule(smoke=smoke)
    )
    return time_rows + (
        {
            "checkpoint_id": "smoke_node_target" if smoke else "node_15m",
            "checkpoint_type": "nodes",
            "target_active_seconds": None,
            "target_active_hours": None,
            "target_nodes": SMOKE_TARGET_NODES if smoke else TARGET_NODES,
        },
    )


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple((CANDIDATE_ID, int(seed)) for seed in seeds)


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
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 24 contract")


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
        "target_nodes": TARGET_NODES,
        "reference_experiment_id": REFERENCE_EXPERIMENT_ID,
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_metrics_sha256": REFERENCE_METRICS_SHA256,
        "evidence_status": "post_selection_follow_up",
        "selection_source": "Experiment 23 development evidence",
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
    "REFERENCE_METRICS_SHA256",
    "REFERENCE_RUN_ID",
    "SMOKE_SEEDS",
    "TARGET_ACTIVE_HOURS",
    "TARGET_NODES",
    "checkpoint_schedule",
    "contract_manifest",
    "task_schedule",
    "validate_contract",
]
