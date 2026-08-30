"""Frozen contract for Experiment 21's 36-hour convergence study."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Sequence

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    DEEP_CFR,
    HELDOUT_SEEDS,
    UCV_CONFIG as EXPERIMENT_19_UCV_CONFIG,
    UCV_ESCHER,
)


EXPERIMENT_ID = 21
EXPERIMENT_NAME = "deep_cfr_ucv_36h_plateau"
GAME_NAME = "leduc_poker"

ALGORITHM_ORDER = (DEEP_CFR, UCV_ESCHER)
ALGORITHMS = {
    DEEP_CFR: {"algorithm_label": "Deep CFR", "repository": "deep_cfr"},
    UCV_ESCHER: {
        "algorithm_label": "UCV-ESCHER",
        "repository": "escher_architecture",
    },
}

# Selection was frozen before Experiment 21: use the first five labels in
# Experiment 19's immutable order, never a result-selected subset.
PRODUCTION_SEEDS = tuple(int(seed) for seed in HELDOUT_SEEDS[:5])
SMOKE_SEEDS = (0,)

TARGET_ACTIVE_HOURS = 36
TARGET_ACTIVE_SECONDS = TARGET_ACTIVE_HOURS * 60 * 60
CHECKPOINT_INTERVAL_HOURS = 2
CHECKPOINT_TARGET_HOURS = tuple(
    range(CHECKPOINT_INTERVAL_HOURS, TARGET_ACTIVE_HOURS + 1, CHECKPOINT_INTERVAL_HOURS)
)

MAX_DEEP_CFR_ITERATIONS = 18_000
MAX_UCV_ITERATIONS = 600

UCV_CONFIG = deepcopy(EXPERIMENT_19_UCV_CONFIG)
UCV_CONFIG["max_num_iterations"] = MAX_UCV_ITERATIONS

# Smoke thresholds are seconds, not hours. Near-zero thresholds guarantee that
# all three archive/evaluation paths are exercised even on a slow laptop.
SMOKE_CHECKPOINT_SECONDS = (0.0, 0.001, 0.002)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    """Return stable IDs and active-time thresholds in chronological order."""
    if smoke:
        return tuple(
            {
                "checkpoint_id": f"smoke_time_{index:02d}",
                "target_active_seconds": float(seconds),
                "target_active_hours": float(seconds) / 3600.0,
            }
            for index, seconds in enumerate(SMOKE_CHECKPOINT_SECONDS, start=1)
        )
    return tuple(
        {
            "checkpoint_id": f"time_{hours:02d}h",
            "target_active_seconds": float(hours * 3600),
            "target_active_hours": float(hours),
        }
        for hours in CHECKPOINT_TARGET_HOURS
    )


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[tuple[str, int], ...]:
    """Map one stable Batch task index to one algorithm/seed trajectory."""
    return tuple(
        (algorithm_id, int(seed))
        for algorithm_id in ALGORITHM_ORDER
        for seed in seeds
    )


def validate_contract(
    *, seeds: Sequence[int], schedule: Sequence[Mapping], smoke: bool
) -> None:
    seeds = tuple(int(seed) for seed in seeds)
    expected_seeds = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if seeds != expected_seeds:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected_seeds}, got {seeds}"
        )
    expected_schedule = checkpoint_schedule(smoke=smoke)
    observed = tuple(
        {
            "checkpoint_id": str(row["checkpoint_id"]),
            "target_active_seconds": float(row["target_active_seconds"]),
            "target_active_hours": float(row["target_active_hours"]),
        }
        for row in schedule
    )
    if observed != expected_schedule:
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 21 contract")


__all__ = [
    "ALGORITHMS",
    "ALGORITHM_ORDER",
    "CHECKPOINT_INTERVAL_HOURS",
    "CHECKPOINT_TARGET_HOURS",
    "DEEP_CFR",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "MAX_DEEP_CFR_ITERATIONS",
    "MAX_UCV_ITERATIONS",
    "PRODUCTION_SEEDS",
    "SMOKE_SEEDS",
    "TARGET_ACTIVE_HOURS",
    "TARGET_ACTIVE_SECONDS",
    "UCV_CONFIG",
    "UCV_ESCHER",
    "checkpoint_schedule",
    "task_schedule",
    "validate_contract",
]
