"""Frozen contract for Experiment 30's causal rare-state audit."""

from __future__ import annotations

from typing import Sequence

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_ID as SOURCE_CANDIDATE_ID,
    PRODUCTION_SEEDS as EXPERIMENT_29_SEEDS,
)


EXPERIMENT_ID = 30
EXPERIMENT_NAME = "causal_rare_state_audit"
GAME_NAME = "leduc_poker"

SOURCE_EXPERIMENT_ID = 29
SOURCE_CHECKPOINTS = ("time_24h", "time_36h")
PRODUCTION_SEEDS = tuple(int(seed) for seed in EXPERIMENT_29_SEEDS)
SMOKE_SEEDS = (0,)

RANKING_ORDER = (
    "rarest_first",
    "largest_policy_error",
    "largest_single_repair_gain",
    "rare_error_impact",
    "random",
)
RANKING_LABELS = {
    "rarest_first": "Rarest first",
    "largest_policy_error": "Largest policy KL",
    "largest_single_repair_gain": "Largest single-state gain",
    "rare_error_impact": "Rare + erroneous + consequential",
    "random": "Random information sets",
}

REPAIR_FRACTIONS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
RANDOM_RANKING_REPLICATES = 20
SMOKE_RANDOM_RANKING_REPLICATES = 2


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def random_ranking_replicates(*, smoke: bool) -> int:
    return (
        SMOKE_RANDOM_RANKING_REPLICATES
        if smoke
        else RANDOM_RANKING_REPLICATES
    )


def validate_contract(*, seeds: Sequence[int], smoke: bool) -> None:
    observed = tuple(int(seed) for seed in seeds)
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed != expected:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected}, "
            f"got {observed}"
        )
    if REPAIR_FRACTIONS[0] != 0.0 or REPAIR_FRACTIONS[-1] != 1.0:
        raise ValueError("Repair fractions must include zero and complete repair")
    if tuple(sorted(REPAIR_FRACTIONS)) != REPAIR_FRACTIONS:
        raise ValueError("Repair fractions must be sorted")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "source_checkpoints": list(SOURCE_CHECKPOINTS),
        "production_seeds": list(PRODUCTION_SEEDS),
        "repair_fractions": list(REPAIR_FRACTIONS),
        "ranking_order": list(RANKING_ORDER),
        "random_ranking_replicates": RANDOM_RANKING_REPLICATES,
        "primary_quantity": (
            "exact exploitability reduction after replacing one neural "
            "information-set policy with its exact weighted-average policy"
        ),
        "evidence_status": "post_training_causal_diagnostic",
        "training_effect": "none",
    }


__all__ = [
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "PRODUCTION_SEEDS",
    "RANDOM_RANKING_REPLICATES",
    "RANKING_LABELS",
    "RANKING_ORDER",
    "REPAIR_FRACTIONS",
    "SMOKE_RANDOM_RANKING_REPLICATES",
    "SMOKE_SEEDS",
    "SOURCE_CANDIDATE_ID",
    "SOURCE_CHECKPOINTS",
    "SOURCE_EXPERIMENT_ID",
    "contract_manifest",
    "random_ranking_replicates",
    "task_schedule",
    "validate_contract",
]
