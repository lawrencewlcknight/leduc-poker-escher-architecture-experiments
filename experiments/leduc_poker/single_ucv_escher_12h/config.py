"""Frozen contract for Experiment 44's Single UCV-ESCHER development study."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Mapping, Sequence

from experiments.leduc_poker.grouped_wide_policy_confirmation.config import (
    PREVIOUSLY_USED_SEEDS,
    PRODUCTION_SEEDS as EXPERIMENT_35_SEEDS,
)
from experiments.leduc_poker.integrated_extended_average_policy_fitting.config import (
    CANDIDATE_CONFIG as EXPERIMENT_43_CONFIG,
    REFINEMENT_GRADIENT_CLIP_NORM,
    REFINEMENT_LEARNING_RATE,
    REFINEMENT_UPDATES,
)


EXPERIMENT_ID = 44
EXPERIMENT_NAME = "single_ucv_escher_12h"
GAME_NAME = "leduc_poker"

CANDIDATE_ID = "single_ucv_escher"
CANDIDATE_LABEL = "Single UCV-ESCHER (full historical mixture)"
ORDINARY_NEURAL_ID = "ordinary_neural_average"
ORDINARY_NEURAL_LABEL = "Contemporaneous neural average (5,000 updates)"
INCUMBENT_NEURAL_ID = "extended_neural_average"
INCUMBENT_NEURAL_LABEL = "Incumbent extended neural average (1,700 extra updates)"
EMPIRICAL_ID = "empirical_reservoir_policy"
EMPIRICAL_LABEL = "Grouped empirical policy reservoir"
EXACT_ID = "exact_tabular_average"
EXACT_LABEL = "Exact reach- and iteration-weighted average"

BOUNDED_CAPACITIES = (16, 32, 64, 128)
TARGET_ACTIVE_HOURS = 12
CHECKPOINT_INTERVAL_HOURS = 2
CHECKPOINT_TARGET_HOURS = tuple(range(2, TARGET_ACTIVE_HOURS + 1, 2))
TRAINING_STATE_HOURS = (6, 12)
MAX_ITERATIONS = 800
SMOKE_SEEDS = (0,)
SMOKE_TIME_SECONDS = (0.0, 0.001)
SMOKE_REFINEMENT_UPDATES = 2
FULL_MIXTURE_TOLERANCE = 1e-10


def _derived_seed(index: int) -> int:
    namespace = f"single-ucv-escher-development-{index}"
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 900_000 + 100_000


# Fresh development labels frozen before Experiment 44 is observed.
PRODUCTION_SEEDS = tuple(_derived_seed(index) for index in range(3))

CANDIDATE_CONFIG = deepcopy(EXPERIMENT_43_CONFIG)
CANDIDATE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        "historical_policy_representation": "per_iteration_regret_networks",
        "historical_policy_weighting": "iteration_power_gamma_with_own_reach",
        "historical_bounded_capacities": BOUNDED_CAPACITIES,
    }
)


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    if smoke:
        return tuple(
            {
                "checkpoint_id": f"smoke_time_{index:02d}",
                "checkpoint_type": "active_time",
                "target_active_seconds": float(seconds),
                "target_active_hours": float(seconds) / 3600.0,
                "target_nodes": None,
            }
            for index, seconds in enumerate(SMOKE_TIME_SECONDS, start=1)
        )
    return tuple(
        {
            "checkpoint_id": f"time_{hours:02d}h",
            "checkpoint_type": "active_time",
            "target_active_seconds": float(hours * 3600),
            "target_active_hours": float(hours),
            "target_nodes": None,
        }
        for hours in CHECKPOINT_TARGET_HOURS
    )


def training_state_checkpoint_ids(*, smoke: bool = False) -> tuple[str, ...]:
    if smoke:
        return tuple(row["checkpoint_id"] for row in checkpoint_schedule(smoke=True))
    return tuple(f"time_{hours:02d}h" for hours in TRAINING_STATE_HOURS)


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[tuple[str, int], ...]:
    return tuple((CANDIDATE_ID, int(seed)) for seed in seeds)


def validate_contract(*, seeds: Sequence[int], schedule: Sequence[Mapping], smoke: bool) -> None:
    observed = tuple(int(seed) for seed in seeds)
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed != expected:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected}, got {observed}"
        )
    if not smoke:
        forbidden = set(PREVIOUSLY_USED_SEEDS).union(EXPERIMENT_35_SEEDS)
        if set(observed).intersection(forbidden):
            raise ValueError("Experiment 44 requires fresh development seed labels")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 44 contract")
    required = {
        "regret_network_type": "mlp",
        "fixed_control_variate_beta": 1.0,
        "q_ensemble_size": 2,
        "critic_target_average_window": 4,
        "use_instantaneous_predictor": False,
        "force_prediction_gate_zero": True,
    }
    for key, expected_value in required.items():
        if CANDIDATE_CONFIG.get(key) != expected_value:
            raise ValueError(f"Frozen Experiment 44 field {key} changed")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID,
        "candidate_label": CANDIDATE_LABEL,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_active_hours": TARGET_ACTIVE_HOURS,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "training_state_hours": list(TRAINING_STATE_HOURS),
        "bounded_snapshot_capacities": list(BOUNDED_CAPACITIES),
        "candidate_config": CANDIDATE_CONFIG,
        "full_historical_mixture": {
            "snapshot_time": "immediately before each outer-iteration update",
            "iteration_weight": "iteration ** gamma",
            "behavioural_conversion": "player-own-reach weighted",
            "average_policy_network_required": False,
        },
        "bounded_mixture": {
            "method": "K independent streaming weighted-reservoir slots",
            "selection_weight": "iteration ** gamma",
            "deployment": "sample one retained regret network per player per hand",
        },
        "paired_incumbent": {
            "source": "same trajectory and policy reservoir",
            "ordinary_updates": 5_000,
            "additional_grouped_updates": REFINEMENT_UPDATES,
            "additional_learning_rate": REFINEMENT_LEARNING_RATE,
            "additional_gradient_clip_norm": REFINEMENT_GRADIENT_CLIP_NORM,
        },
        "primary_outcomes": [
            "full-historical-mixture exploitability",
            "paired full-mixture minus incumbent-neural exploitability",
            "minimum bounded capacity retaining the full-mixture gain",
        ],
        "evidence_status": "paired post-selection development evidence",
    }


__all__ = [name for name in globals() if name.isupper()] + [
    "checkpoint_schedule",
    "contract_manifest",
    "task_schedule",
    "training_state_checkpoint_ids",
    "validate_contract",
]
