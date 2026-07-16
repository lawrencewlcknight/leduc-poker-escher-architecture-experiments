"""Experiment 28 configuration retained as the architecture baseline.

This module is intentionally self-contained so the new architecture repository
does not depend on any of the historical experiment packages.  The values are
the exact defaults used by Experiment 28.
"""

from __future__ import annotations

from escher_poker.constants import (
    EXPLOITABILITY_THRESHOLD,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config

DEFAULT_SEEDS = [1234, 2025, 31415, 27182, 16180]

CANDIDATE_VARIANT = {
    "variant_id": "deep_plain_standardized_regret_action_head",
    "variant_label": "Deep plain + standardized regret action head",
    "variant_description": (
        "Candidate ESCHER architecture: 256x256x128 plain policy, regret, "
        "and value trunks; standard linear policy output; one 64-unit "
        "per-action regret head; standardized legal regret targets."
    ),
}

BASE_CONFIG = {
    "experiment_name": "leduc_poker_escher_candidate_architecture_multiseed",
    "game_name": "leduc_poker",
    "num_iterations": 80,
    "num_traversals": 500,
    "num_val_fn_traversals": 500,
    "check_exploitability_every": 10,
    "policy_network_layers": (256, 256, 128),
    "regret_network_layers": (256, 256, 128),
    "value_network_layers": (256, 256, 128),
    "learning_rate": 1e-3,
    "batch_size_regret": 256,
    "batch_size_value": 256,
    "batch_size_average_policy": 10_000,
    "memory_capacity": int(5e4),
    "policy_network_train_steps": 1000,
    "regret_network_train_steps": 200,
    "value_network_train_steps": 200,
    "compute_exploitability": True,
    "reinitialize_regret_networks": True,
    "reinitialize_value_network": True,
    "save_policy_weights": False,
    "save_final_checkpoints": False,
    "train_device": "cpu",
    "infer_device": "cpu",
    "verbose": False,
    "exploitability_threshold": EXPLOITABILITY_THRESHOLD,
    "average_policy_value_target": LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    "importance_sampling": False,
    "zero_regret_fallback": "uniform",
    "all_actions": True,
    "expl": 1.0,
    "val_expl": 0.01,
    "policy_network_activation": "leakyrelu",
    "regret_network_activation": "leakyrelu",
    "value_network_activation": "leakyrelu",
    "policy_network_layer_norm": False,
    "regret_network_layer_norm": False,
    "value_network_layer_norm": False,
    "policy_network_residual_mode": "none",
    "regret_network_residual_mode": "none",
    "value_network_residual_mode": "none",
    "policy_network_head_depth": 0,
    "policy_network_head_units": None,
    "regret_network_head_depth": 1,
    "regret_network_head_units": 64,
    "regret_network_output_mode": "direct",
    "regret_target_baseline": "author_state_value",
    "regret_target_processing": "standardize",
    "regret_target_clip_value": 1.0,
    "regret_target_standardize_epsilon": 1e-6,
    "regret_replay_mode": "reservoir",
    "regret_replay_rare_history_quota": 64,
    "regret_replay_weight_floor": 1e-6,
    "use_balanced_probs": False,
    "balanced_sampling_mix": 0.0,
    "track_sampling_coverage": False,
}

DEFAULT_CONFIG = make_variant_config(BASE_CONFIG, CANDIDATE_VARIANT)
