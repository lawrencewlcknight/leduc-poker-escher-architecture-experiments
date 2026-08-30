"""Frozen contract for exact tabular validation of the UCV estimator."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    CANDIDATE_CONFIG,
    MAX_NUM_ITERATIONS,
)


EXPERIMENT_NAME = "ucv_exact_tabular_validation"
DEFAULT_SEEDS = (0, 1, 2)
TARGET_NODES = 15_000_000
CHECKPOINT_TARGETS = (
    ("early", 1_500_000),
    ("middle", 7_500_000),
    ("late", 15_000_000),
)

FULL_ADAPTIVE = "full_adaptive_ucv"
FIXED_BETA_ONE = "fixed_beta_one"
PREDICTION_GATE_ZERO = "prediction_gate_zero"
CALIBRATION_DISABLED = "residual_calibration_disabled"
BASELINE_FREE = "baseline_free"

VARIANTS = {
    FULL_ADAPTIVE: {
        "label": "Full adaptive UCV",
        "policy_mode": "current",
        "beta_mode": "adaptive",
        "sampling_mode": "adaptive",
        "calibration_mode": "frozen_predictor",
        "control_mode": "q_control",
    },
    FIXED_BETA_ONE: {
        "label": "Fixed beta = 1",
        "policy_mode": "current",
        "beta_mode": "fixed_one",
        "sampling_mode": "adaptive",
        "calibration_mode": "frozen_predictor",
        "control_mode": "q_control",
    },
    PREDICTION_GATE_ZERO: {
        "label": "Prediction gate = 0",
        "policy_mode": "cumulative_only",
        "beta_mode": "adaptive",
        "sampling_mode": "adaptive",
        "calibration_mode": "frozen_predictor",
        "control_mode": "q_control",
    },
    CALIBRATION_DISABLED: {
        "label": "Residual calibration disabled",
        "policy_mode": "current",
        "beta_mode": "fixed_one",
        "sampling_mode": "uniform",
        "calibration_mode": "disabled",
        "control_mode": "q_control",
    },
    BASELINE_FREE: {
        "label": "Baseline-free Horvitz--Thompson",
        "policy_mode": "current",
        "beta_mode": "zero",
        "sampling_mode": "adaptive",
        "calibration_mode": "frozen_predictor",
        "control_mode": "no_control",
    },
}
VARIANT_ORDER = tuple(VARIANTS)

BASE_CONFIG = deepcopy(CANDIDATE_CONFIG)
BASE_CONFIG.update(
    {
        "max_num_iterations": MAX_NUM_ITERATIONS,
        "evaluate_initial_policy": False,
        "early_evaluation_node_thresholds": (),
        "preserve_evaluation_rng": True,
    }
)

CONDITIONAL_BIAS_TOLERANCE = 1e-9
MEASURED_HOURS_PER_SEED = 11.22
EXPECTED_SEQUENTIAL_HOURS = 36
BATCH_TIMEOUT_SECONDS = 48 * 60 * 60


def checkpoint_contract(target_nodes: int = TARGET_NODES):
    """Scale the production fractions for explicit development smoke runs."""

    target_nodes = int(target_nodes)
    if target_nodes <= 0:
        raise ValueError("target_nodes must be positive")
    if target_nodes == TARGET_NODES:
        return CHECKPOINT_TARGETS
    early = max(1, int(round(0.10 * target_nodes)))
    middle = max(early + 1, int(round(0.50 * target_nodes)))
    late = max(middle + 1, target_nodes)
    return (("early", early), ("middle", middle), ("late", late))


__all__ = [
    "BASELINE_FREE",
    "BASE_CONFIG",
    "BATCH_TIMEOUT_SECONDS",
    "CALIBRATION_DISABLED",
    "CHECKPOINT_TARGETS",
    "CONDITIONAL_BIAS_TOLERANCE",
    "DEFAULT_SEEDS",
    "EXPECTED_SEQUENTIAL_HOURS",
    "EXPERIMENT_NAME",
    "FIXED_BETA_ONE",
    "FULL_ADAPTIVE",
    "MEASURED_HOURS_PER_SEED",
    "PREDICTION_GATE_ZERO",
    "TARGET_NODES",
    "VARIANTS",
    "VARIANT_ORDER",
    "checkpoint_contract",
]
