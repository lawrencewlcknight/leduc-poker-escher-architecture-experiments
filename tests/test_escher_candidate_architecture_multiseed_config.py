"""Configuration checks for Experiment 28 candidate architecture validation."""

from __future__ import annotations

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    CANDIDATE_VARIANT,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
)


def test_candidate_architecture_matches_proposed_final_model():
    assert DEFAULT_SEEDS == [1234, 2025, 31415, 27182, 16180]
    assert DEFAULT_CONFIG["num_iterations"] == 80
    assert DEFAULT_CONFIG["num_traversals"] == 500
    assert DEFAULT_CONFIG["num_val_fn_traversals"] == 500
    assert DEFAULT_CONFIG["importance_sampling"] is False
    assert DEFAULT_CONFIG["zero_regret_fallback"] == "uniform"
    assert DEFAULT_CONFIG["all_actions"] is True
    assert DEFAULT_CONFIG["policy_network_layers"] == (256, 256, 128)
    assert DEFAULT_CONFIG["regret_network_layers"] == (256, 256, 128)
    assert DEFAULT_CONFIG["value_network_layers"] == (256, 256, 128)
    assert DEFAULT_CONFIG["policy_network_layer_norm"] is False
    assert DEFAULT_CONFIG["regret_network_layer_norm"] is False
    assert DEFAULT_CONFIG["value_network_layer_norm"] is False
    assert DEFAULT_CONFIG["policy_network_residual_mode"] == "none"
    assert DEFAULT_CONFIG["regret_network_residual_mode"] == "none"
    assert DEFAULT_CONFIG["value_network_residual_mode"] == "none"
    assert DEFAULT_CONFIG["policy_network_head_depth"] == 0
    assert DEFAULT_CONFIG["policy_network_head_units"] is None
    assert DEFAULT_CONFIG["regret_network_head_depth"] == 1
    assert DEFAULT_CONFIG["regret_network_head_units"] == 64
    assert DEFAULT_CONFIG["regret_network_output_mode"] == "direct"
    assert DEFAULT_CONFIG["regret_target_processing"] == "standardize"
    assert DEFAULT_CONFIG["regret_target_clip_value"] == 1.0
    assert DEFAULT_CONFIG["regret_target_standardize_epsilon"] == 1e-6
    assert DEFAULT_CONFIG["variant_id"] == CANDIDATE_VARIANT["variant_id"]
    assert DEFAULT_CONFIG["total_policy_training_events_expected"] >= 1
    assert DEFAULT_CONFIG["policy_gradient_steps_expected"] >= 1

