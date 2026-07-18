"""Configuration and immutable Experiment 2 references for Experiment 6."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.adaptive_residual_predictive_escher.config import (
    ADAPTIVE_CONFIG as EXPERIMENT_3_CONFIG,
    DEFAULT_SEEDS,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.config import (
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_2_SOURCE,
    REFERENCE_CURVE_ROWS,
    REFERENCE_CURVES,
    REFERENCE_CURVES_SHA256,
)


ALGORITHM_ID = "unbiased_control_variate_escher"
ALGORITHM_LABEL = "Unbiased Control-Variate ESCHER"

UNBIASED_CONFIG = deepcopy(EXPERIMENT_3_CONFIG)
for obsolete_key in (
    "lambda_start",
    "lambda_schedule_half_life",
    "lambda_schedule_power",
    "lambda_residual_ema_decay",
    "lambda_residual_scale",
    "lambda_initial_residual",
    "sampling_mode",
):
    UNBIASED_CONFIG.pop(obsolete_key)

UNBIASED_CONFIG.update(
    {
        # Three disjoint folds give two held-out predictions per trajectory.
        "q_ensemble_size": 3,
        "beta_min": 0.0,
        "beta_max": 2.0,
        "beta_ridge": 1e-4,
        # xi=(1-floor)*variance_adaptive + floor*uniform, so every legal
        # traverser action receives probability at least floor/|A(I)|.
        "sampling_uniform_floor_mass": 0.2,
        "calibration_buffer_size": 1_000_000,
        "calibration_batch_size": 2_048,
        "calibration_train_steps": 2_000,
        "calibration_learning_rate": 1e-3,
        "calibration_minimum_variance": 1e-5,
        "prediction_gate_ema_decay": 0.9,
        "prediction_gate_initial": 0.0,
    }
)

EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 14
BATCH_TIMEOUT_SECONDS = 36 * 60 * 60
