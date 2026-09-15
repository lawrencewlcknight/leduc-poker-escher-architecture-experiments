"""Frozen contract for Experiment 33's staged distillation study."""

from __future__ import annotations

from typing import Sequence

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_ID as SOURCE_CANDIDATE_ID,
    PRODUCTION_SEEDS as EXPERIMENT_29_SEEDS,
)


EXPERIMENT_ID = 33
EXPERIMENT_NAME = "consequence_weighted_distillation"
GAME_NAME = "leduc_poker"
SOURCE_EXPERIMENT_ID = 29
AUDIT_EXPERIMENT_ID = 30
SOURCE_CHECKPOINTS = ("time_24h", "time_36h")
SELECTION_CHECKPOINT = "time_24h"
VALIDATION_CHECKPOINT = "time_36h"
PRODUCTION_SEEDS = tuple(int(seed) for seed in EXPERIMENT_29_SEEDS)
SMOKE_SEEDS = (0,)

FIT_REPLICATES = 3
SMOKE_FIT_REPLICATES = 1
TRAIN_STEPS = 5_000
FINE_TUNE_STEPS = 2_500
SMOKE_TRAIN_STEPS = 3
SMOKE_FINE_TUNE_STEPS = 2
VALIDATION_SAMPLES = 50_000
SMOKE_VALIDATION_SAMPLES = 16
WEIGHT_EXPONENT = 0.5
WEIGHT_MIN = 0.25
WEIGHT_MAX = 4.0
TARGETED_BATCH_FRACTION = 0.75
PROXY_REPAIR_FRACTIONS = (0.01, 0.025, 0.05, 0.10, 0.20)

PROXY_ORDER = (
    "counterfactual_reach",
    "action_value_span",
    "expected_absolute_advantage",
    "value_span_x_sqrt_reach",
    "expected_advantage_x_sqrt_reach",
    "policy_error",
    "error_x_value_span",
    "error_x_expected_advantage",
    "error_x_value_span_x_sqrt_reach",
)
PROXY_LABELS = {
    "counterfactual_reach": "Counterfactual reach",
    "policy_error": "Policy KL",
    "action_value_span": "Action-value span",
    "expected_absolute_advantage": "Expected absolute advantage",
    "value_span_x_sqrt_reach": "Value span x sqrt(reach)",
    "expected_advantage_x_sqrt_reach": "Expected advantage x sqrt(reach)",
    "error_x_value_span": "Policy KL x value span",
    "error_x_expected_advantage": "Policy KL x expected advantage",
    "error_x_value_span_x_sqrt_reach": "Policy KL x value span x sqrt(reach)",
}
SELECTABLE_PROXY_ORDER = PROXY_ORDER[:5]

BASELINE = "baseline_cross_entropy"
PRIORITISED = "prioritised_importance_corrected"
CONSEQUENCE_WEIGHTED = "consequence_weighted"
ERROR_CONSEQUENCE_WEIGHTED = "error_consequence_weighted"
TWO_STAGE = "two_stage_targeted_finetune"
ORACLE_WEIGHTED = "oracle_repair_gain_weighted"
ARM_ORDER = (
    BASELINE,
    PRIORITISED,
    CONSEQUENCE_WEIGHTED,
    ERROR_CONSEQUENCE_WEIGHTED,
    TWO_STAGE,
    ORACLE_WEIGHTED,
)
ARM_LABELS = {
    BASELINE: "Standard soft-target CE",
    PRIORITISED: "Proxy-prioritised CE (importance corrected)",
    CONSEQUENCE_WEIGHTED: "Consequence-weighted CE",
    ERROR_CONSEQUENCE_WEIGHTED: "Policy-error x consequence CE",
    TWO_STAGE: "CE then targeted fine-tuning",
    ORACLE_WEIGHTED: "Exact repair-gain weighting (diagnostic)",
}


def task_schedule(seeds: Sequence[int] = PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def fit_replicates(*, smoke: bool) -> tuple[int, ...]:
    count = SMOKE_FIT_REPLICATES if smoke else FIT_REPLICATES
    return tuple(range(count))


def validate_contract(*, seeds: Sequence[int], smoke: bool) -> None:
    expected = SMOKE_SEEDS if smoke else PRODUCTION_SEEDS
    observed = tuple(int(seed) for seed in seeds)
    if observed != expected:
        raise ValueError(f"Expected seeds {expected}, got {observed}")
    if not 0.0 < TARGETED_BATCH_FRACTION < 1.0:
        raise ValueError("Targeted batch fraction must preserve baseline rehearsal")
    if not 0.0 < WEIGHT_MIN <= 1.0 <= WEIGHT_MAX:
        raise ValueError("Weight clipping must contain one")


def contract_manifest() -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "audit_experiment_id": AUDIT_EXPERIMENT_ID,
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "source_checkpoints": list(SOURCE_CHECKPOINTS),
        "selection_checkpoint": SELECTION_CHECKPOINT,
        "validation_checkpoint": VALIDATION_CHECKPOINT,
        "production_seeds": list(PRODUCTION_SEEDS),
        "proxy_candidates": list(PROXY_ORDER),
        "selectable_consequence_proxies": list(SELECTABLE_PROXY_ORDER),
        "arms": list(ARM_ORDER),
        "fit_replicates": FIT_REPLICATES,
        "train_steps": TRAIN_STEPS,
        "fine_tune_steps": FINE_TUNE_STEPS,
        "weight_exponent": WEIGHT_EXPONENT,
        "weight_clip": [WEIGHT_MIN, WEIGHT_MAX],
        "targeted_batch_fraction": TARGETED_BATCH_FRACTION,
        "selection_rule": (
            "largest mean fraction of positive single-repair gain captured "
            "by the top 10 percent at time_24h; mean Spearman then proxy id break ties"
        ),
        "primary_outcome": "time_36h neural-minus-exact exploitability gap",
        "inferential_unit": "Experiment 29 source trajectory",
        "evidence_status": "paired offline architecture-development evidence",
        "training_effect": "average-policy fitting only; UCV regret learning is frozen",
    }
