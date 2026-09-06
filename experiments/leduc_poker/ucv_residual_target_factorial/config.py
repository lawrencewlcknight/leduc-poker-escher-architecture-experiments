"""Frozen contract for Experiment 25's 2x2 UCV stability factorial."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Mapping, Sequence

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import HELDOUT_SEEDS
from experiments.leduc_poker.ucv_three_arm_15m_simplification.config import (
    PRODUCTION_SEEDS as EXPERIMENT_22_SEEDS,
)
from experiments.leduc_poker.ucv_24h_stability_development.config import (
    NONPREDICTIVE_CORE,
    PRODUCTION_SEEDS as EXPERIMENT_23_SEEDS,
    variant_config as experiment_23_variant_config,
)


EXPERIMENT_ID = 25
EXPERIMENT_NAME = "ucv_residual_target_factorial"
GAME_NAME = "leduc_poker"

CONTROL = "control"
RESIDUAL_ONLY = "residual_regret"
AVERAGED_TARGET_ONLY = "averaged_critic_target"
COMBINED = "residual_regret_averaged_target"

VARIANT_ORDER = (CONTROL, RESIDUAL_ONLY, AVERAGED_TARGET_ONLY, COMBINED)
VARIANTS = {
    CONTROL: {
        "variant_label": "Control",
        "residual_regret": False,
        "averaged_critic_target": False,
    },
    RESIDUAL_ONLY: {
        "variant_label": "Residual-LN regret",
        "residual_regret": True,
        "averaged_critic_target": False,
    },
    AVERAGED_TARGET_ONLY: {
        "variant_label": "Averaged critic target",
        "residual_regret": False,
        "averaged_critic_target": True,
    },
    COMBINED: {
        "variant_label": "Residual-LN + averaged target",
        "residual_regret": True,
        "averaged_critic_target": True,
    },
}

REGRET_RESIDUAL_WIDTH = 64
REGRET_RESIDUAL_BLOCKS = 4
CRITIC_TARGET_AVERAGE_WINDOW = 4


def _derived_seed(index: int) -> int:
    namespace = f"ucv-residual-target-factorial-{index}"
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 900_000 + 100_000


# Fresh paired development labels, fixed by the namespace above before results.
PRODUCTION_SEEDS = tuple(_derived_seed(index) for index in range(3))
SMOKE_SEEDS = (0,)

TARGET_ACTIVE_HOURS = 36
CHECKPOINT_INTERVAL_HOURS = 2
CHECKPOINT_TARGET_HOURS = tuple(range(2, TARGET_ACTIVE_HOURS + 1, 2))
TRAINING_STATE_HOURS = (24, 36)
LATE_WINDOW_START_HOURS = 24
TARGET_NODES = 15_000_000
MAX_ITERATIONS = 800
SMOKE_TIME_SECONDS = (0.0, 0.001, 0.002)
SMOKE_TRAINING_STATE_CHECKPOINT_IDS = ("smoke_time_02", "smoke_time_03")
SMOKE_TARGET_NODES = 100

BASE_CONFIG = deepcopy(experiment_23_variant_config(NONPREDICTIVE_CORE))
BASE_CONFIG.update(
    {
        "max_num_iterations": MAX_ITERATIONS,
        "regret_network_type": "mlp",
        "regret_residual_width": REGRET_RESIDUAL_WIDTH,
        "regret_residual_blocks": REGRET_RESIDUAL_BLOCKS,
        "critic_target_average_window": 1,
        "regret_policy_gradient_clip_norm": None,
        "anneal_start_nodes": None,
        "anneal_end_nodes": None,
        "anneal_final_learning_rate": None,
    }
)


def variant_config(variant_id: str) -> dict:
    if variant_id not in VARIANTS:
        raise ValueError(f"Unknown Experiment 25 variant: {variant_id}")
    config = deepcopy(BASE_CONFIG)
    treatment = VARIANTS[variant_id]
    if treatment["residual_regret"]:
        config["regret_network_type"] = "residual_layer_norm"
    if treatment["averaged_critic_target"]:
        config["critic_target_average_window"] = CRITIC_TARGET_AVERAGE_WINDOW
    return config


def checkpoint_schedule(*, smoke: bool = False) -> tuple[dict, ...]:
    if smoke:
        time_rows = tuple(
            {
                "checkpoint_id": f"smoke_time_{index:02d}",
                "checkpoint_type": "active_time",
                "target_active_seconds": float(seconds),
                "target_active_hours": float(seconds) / 3600.0,
                "target_nodes": None,
            }
            for index, seconds in enumerate(SMOKE_TIME_SECONDS, start=1)
        )
        node_target = SMOKE_TARGET_NODES
    else:
        time_rows = tuple(
            {
                "checkpoint_id": f"time_{hours:02d}h",
                "checkpoint_type": "active_time",
                "target_active_seconds": float(hours * 3600),
                "target_active_hours": float(hours),
                "target_nodes": None,
            }
            for hours in CHECKPOINT_TARGET_HOURS
        )
        node_target = TARGET_NODES
    return time_rows + (
        {
            "checkpoint_id": "smoke_node_target" if smoke else "node_15m",
            "checkpoint_type": "nodes",
            "target_active_seconds": None,
            "target_active_hours": None,
            "target_nodes": int(node_target),
        },
    )


def training_state_checkpoint_ids(*, smoke: bool = False) -> tuple[str, ...]:
    if smoke:
        return SMOKE_TRAINING_STATE_CHECKPOINT_IDS
    return tuple(f"time_{hours:02d}h" for hours in TRAINING_STATE_HOURS)


def task_schedule(
    seeds: Sequence[int] = PRODUCTION_SEEDS,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (variant_id, int(seed))
        for variant_id in VARIANT_ORDER
        for seed in seeds
    )


def validate_contract(
    *, seeds: Sequence[int], schedule: Sequence[Mapping], smoke: bool
) -> None:
    observed = tuple(int(seed) for seed in seeds)
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    if observed != expected:
        raise ValueError(
            f"{'Smoke' if smoke else 'Production'} seeds must be {expected}, got {observed}"
        )
    if not smoke:
        forbidden = set(HELDOUT_SEEDS).union(EXPERIMENT_22_SEEDS, EXPERIMENT_23_SEEDS)
        if set(observed).intersection(forbidden):
            raise ValueError("Experiment 25 seeds must be fresh development labels")
    if tuple(dict(row) for row in schedule) != checkpoint_schedule(smoke=smoke):
        raise ValueError("Checkpoint schedule differs from the frozen Experiment 25 contract")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "design": "2x2 paired factorial",
        "variant_order": list(VARIANT_ORDER),
        "variants": VARIANTS,
        "production_seeds": list(PRODUCTION_SEEDS),
        "target_active_hours": TARGET_ACTIVE_HOURS,
        "checkpoint_interval_hours": CHECKPOINT_INTERVAL_HOURS,
        "training_state_hours": list(TRAINING_STATE_HOURS),
        "target_nodes": TARGET_NODES,
        "regret_residual_width": REGRET_RESIDUAL_WIDTH,
        "regret_residual_blocks": REGRET_RESIDUAL_BLOCKS,
        "critic_target_average_window": CRITIC_TARGET_AVERAGE_WINDOW,
        "baseline": "Experiment 23 selected non-predictive fast core",
        "evidence_status": "paired development evidence",
    }


__all__ = [
    "AVERAGED_TARGET_ONLY",
    "BASE_CONFIG",
    "CHECKPOINT_TARGET_HOURS",
    "COMBINED",
    "CONTROL",
    "CRITIC_TARGET_AVERAGE_WINDOW",
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "GAME_NAME",
    "LATE_WINDOW_START_HOURS",
    "MAX_ITERATIONS",
    "PRODUCTION_SEEDS",
    "REGRET_RESIDUAL_BLOCKS",
    "REGRET_RESIDUAL_WIDTH",
    "RESIDUAL_ONLY",
    "SMOKE_SEEDS",
    "TARGET_ACTIVE_HOURS",
    "TARGET_NODES",
    "TRAINING_STATE_HOURS",
    "VARIANTS",
    "VARIANT_ORDER",
    "checkpoint_schedule",
    "contract_manifest",
    "task_schedule",
    "training_state_checkpoint_ids",
    "validate_contract",
    "variant_config",
]
