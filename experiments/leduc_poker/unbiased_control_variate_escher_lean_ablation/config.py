"""Configuration for the paired lean Experiment 6 ablation."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    UNBIASED_CONFIG,
)


EXPERIMENT_ID = 8
FULL_EXPERIMENT_6 = "full_experiment_6"
FIXED_BETA_ONE = "fixed_beta_one"
PREDICTION_GATE_ZERO = "prediction_gate_zero"
FIXED_BETA_ONE_NO_PREDICTOR = "fixed_beta_one_no_predictor"
TWO_CROSS_FITTED_CRITICS = "two_cross_fitted_critics"
SINGLE_FROZEN_TARGET_CRITIC = "single_frozen_target_critic"
UNIFORM_FULL_SUPPORT_SAMPLING = "uniform_full_support_sampling"
LEAN_CANDIDATE = "lean_candidate"

# The first seven arms are the requested one-mechanism ablations. The eighth is
# the actual proposed simplification: without it, interactions between the
# individually successful removals would remain unknown.
VARIANTS = {
    FULL_EXPERIMENT_6: {
        "variant_label": "Full Experiment 6",
        "overrides": {},
        "mechanism": "Three critics, adaptive beta/sampling and gated predictor.",
    },
    FIXED_BETA_ONE: {
        "variant_label": "Fixed beta = 1",
        "overrides": {"fixed_control_variate_beta": 1.0},
        "mechanism": "Remove adaptive beta; retain calibration for sampling.",
    },
    PREDICTION_GATE_ZERO: {
        "variant_label": "Prediction gate = 0",
        "overrides": {"force_prediction_gate_zero": True},
        "mechanism": "Train and diagnose the predictor but never use it.",
    },
    FIXED_BETA_ONE_NO_PREDICTOR: {
        "variant_label": "Beta = 1 + no predictor",
        "overrides": {
            "fixed_control_variate_beta": 1.0,
            "use_instantaneous_predictor": False,
        },
        "mechanism": "Fixed residual correction with a DCFR+ accumulator.",
    },
    TWO_CROSS_FITTED_CRITICS: {
        "variant_label": "Two cross-fitted critics",
        "overrides": {"q_ensemble_size": 2},
        "mechanism": "One strictly held-out critic prediction per trajectory.",
    },
    SINGLE_FROZEN_TARGET_CRITIC: {
        "variant_label": "Single frozen-target critic",
        "overrides": {"q_ensemble_size": 1},
        "mechanism": "Persistent target critic without cross-fitting or ensemble.",
    },
    UNIFORM_FULL_SUPPORT_SAMPLING: {
        "variant_label": "Uniform full-support sampling",
        "overrides": {"sampling_uniform_floor_mass": 1.0},
        "mechanism": "Remove residual-adaptive action sampling only.",
    },
    LEAN_CANDIDATE: {
        "variant_label": "Lean unbiased DCFR+ candidate",
        "overrides": {
            "fixed_control_variate_beta": 1.0,
            "use_instantaneous_predictor": False,
            "q_ensemble_size": 2,
            "sampling_uniform_floor_mass": 1.0,
            "use_residual_calibration": False,
        },
        "mechanism": (
            "Always-unbiased residual correction, two cross-fitted critics, "
            "non-predictive DCFR+, uniform sampling and no calibration network."
        ),
    },
}
DEFAULT_VARIANT_IDS = tuple(VARIANTS)

BASE_CONFIG = deepcopy(UNBIASED_CONFIG)
BASE_CONFIG.update(
    {
        "fixed_control_variate_beta": None,
        "force_prediction_gate_zero": False,
        "use_instantaneous_predictor": True,
        "use_residual_calibration": True,
    }
)

# Experiment 6 measured about 3.53 hours per seed. Eight arms by three seeds
# would therefore have a 84.7-hour upper bound if every simplification saved no
# time. The actual sequential estimate is lower because several arms remove
# networks; the timeouts retain substantial operational headroom.
MEASURED_FULL_EXPERIMENT_6_HOURS_PER_SEED = 3.53
EXPECTED_SEQUENTIAL_RUNTIME_HOURS = 72
RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES = 96 * 60
SEQUENTIAL_BATCH_TIMEOUT_SECONDS = RECOMMENDED_SINGLE_BATCH_TIMEOUT_MINUTES * 60
EXPECTED_PARALLEL_BY_VARIANT_RUNTIME_HOURS = 14
PARALLEL_BY_VARIANT_BATCH_TIMEOUT_SECONDS = 24 * 60 * 60
PER_WORKER_BATCH_TIMEOUT_SECONDS = 12 * 60 * 60
