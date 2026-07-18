"""Configuration for the Leduc adaptive-ESCHER forensic experiment."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.adaptive_residual_predictive_escher.config import (
    ADAPTIVE_CONFIG as EXPERIMENT_3_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_1_NODE_TARGETS,
)


FORENSIC_CONFIG = deepcopy(EXPERIMENT_3_CONFIG)

# One-factor-at-a-time mechanism ablations. This deliberately is not a full
# factorial sweep: every arm except the control changes exactly one mechanism.
VARIANTS = {
    "scheduled_predictive_persistent": {
        "label": "Scheduled lambda + predictive + persistent Q",
        "description": "Exact Experiment 3 architecture (forensic control).",
        "lambda_mode": "scheduled",
        "use_predictive_accumulator": True,
        "q_mode": "persistent",
    },
    "lambda_one": {
        "label": "Lambda = 1 (unbiased residual correction)",
        "description": "Control with lambda fixed to one.",
        "lambda_mode": "fixed_one",
        "use_predictive_accumulator": True,
        "q_mode": "persistent",
    },
    "lambda_zero": {
        "label": "Lambda = 0 (relative-Q only)",
        "description": "Control with residual correction removed.",
        "lambda_mode": "fixed_zero",
        "use_predictive_accumulator": True,
        "q_mode": "persistent",
    },
    "residual_only_lambda": {
        "label": "Residual-adaptive lambda (no schedule floor)",
        "description": "Past residual calibration without the global floor.",
        "lambda_mode": "residual_only",
        "use_predictive_accumulator": True,
        "q_mode": "persistent",
    },
    "nonpredictive_accumulator": {
        "label": "Non-predictive accumulator",
        "description": "Cumulative clipped/discounted advantage without the predictor.",
        "lambda_mode": "scheduled",
        "use_predictive_accumulator": False,
        "q_mode": "persistent",
    },
    "reinitialized_q": {
        "label": "Reinitialised Q",
        "description": (
            "Experiment 3 estimator with the upstream reinitialised Q learner."
        ),
        "lambda_mode": "scheduled",
        "use_predictive_accumulator": True,
        "q_mode": "reinitialized",
    },
}

CONTROL_VARIANT = "scheduled_predictive_persistent"
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 12
BATCH_TIMEOUT_SECONDS = 24 * 60 * 60
