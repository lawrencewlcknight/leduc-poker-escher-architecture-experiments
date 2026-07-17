"""Correctness tests for Experiment 3's estimator and solver."""

from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_escher import (
    AdaptiveLambdaController,
    AdaptiveResidualPredictiveEscher,
    adaptive_residual_corrected_advantage,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher.config import (
    ADAPTIVE_CONFIG,
    DEFAULT_SEEDS,
    EXPERIMENT_1_NODE_TARGETS,
    REFERENCE_CURVES,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher.run import (
    _load_reference_curves,
)
from experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.config import (
    VR_PAPER_CONFIG,
)
from vr_deep_cfr.logger import Logger


def test_lambda_zero_is_direct_relative_q_and_is_exactly_centred():
    result = adaptive_residual_corrected_advantage(
        [0.2, -0.4, 99.0],
        sampled_action=0,
        sample_probability=0.5,
        sampled_return=1.0,
        lambda_value=0.0,
        policy=[0.25, 0.75, 0.0],
        legal_actions_mask=[1, 1, 0],
    )

    np.testing.assert_allclose(result.q_values, [0.2, -0.4, 0.0])
    assert result.policy_value == pytest.approx(-0.25)
    np.testing.assert_allclose(result.advantages, [0.45, -0.15, 0.0])
    assert result.policy_weighted_advantage == pytest.approx(0.0, abs=1e-15)


def test_lambda_one_residual_correction_is_unbiased_for_each_action_value():
    q_hat = np.array([0.1, -0.2, 0.7])
    true_q = np.array([0.8, -0.5, 0.2])
    sample_policy = np.array([0.2, 0.3, 0.5])
    policy = np.array([0.5, 0.25, 0.25])
    expected_q = np.zeros(3)
    expected_advantage = np.zeros(3)

    for sampled_action, probability in enumerate(sample_policy):
        result = adaptive_residual_corrected_advantage(
            q_hat,
            sampled_action=sampled_action,
            sample_probability=float(probability),
            sampled_return=float(true_q[sampled_action]),
            lambda_value=1.0,
            policy=policy,
            legal_actions_mask=[1, 1, 1],
        )
        expected_q += probability * result.q_values
        expected_advantage += probability * result.advantages

    np.testing.assert_allclose(expected_q, true_q, atol=1e-12)
    true_advantage = true_q - np.dot(policy, true_q)
    np.testing.assert_allclose(expected_advantage, true_advantage, atol=1e-12)


def test_intermediate_lambda_has_the_documented_shrinkage_expectation():
    q_hat = np.array([0.0, 1.0])
    true_q = np.array([1.0, -1.0])
    sample_policy = np.array([0.4, 0.6])
    lambda_value = 0.25
    expectation = np.zeros(2)
    for action, probability in enumerate(sample_policy):
        result = adaptive_residual_corrected_advantage(
            q_hat,
            sampled_action=action,
            sample_probability=float(probability),
            sampled_return=float(true_q[action]),
            lambda_value=lambda_value,
            policy=[0.5, 0.5],
            legal_actions_mask=[1, 1],
        )
        expectation += probability * result.q_values

    np.testing.assert_allclose(
        expectation,
        (1.0 - lambda_value) * q_hat + lambda_value * true_q,
        atol=1e-12,
    )


def test_lambda_uses_past_residual_and_floor_tends_to_one():
    controller = AdaptiveLambdaController(
        2,
        3,
        lambda_start=0.2,
        schedule_half_life=2.0,
        schedule_power=1.0,
        residual_ema_decay=0.0,
        residual_scale=0.25,
        initial_residual=1.0,
    )
    assert controller.schedule_floor(1) == pytest.approx(0.2)
    assert controller.schedule_floor(3) == pytest.approx(0.6)
    before = controller.value(0, 1, 1)
    assert before == pytest.approx(0.8)

    controller.observe(
        0,
        1,
        sampled_residual=0.0,
        lambda_value=before,
        residual_correction=0.0,
        policy_weighted_advantage=0.0,
    )
    # The just-observed sample can affect only future lambda values.
    assert controller.value(0, 1, 1) == pytest.approx(0.2)
    assert controller.schedule_floor(1_000_000) > 0.99999


def _tiny_solver(num_iterations=2):
    traversals = 4
    solver = AdaptiveResidualPredictiveEscher(
        game_name="leduc_poker",
        num_episodes=2 * traversals * num_iterations,
        advantage_buffer_size=128,
        ave_policy_buffer_size=128,
        baseline_buffer_size=128,
        learning_rate=1e-3,
        num_traversals=traversals,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        baseline_network_train_steps=1,
        advantage_batch_size=2,
        ave_policy_batch_size=2,
        baseline_batch_size=2,
        num_layers=1,
        num_hiddens=8,
        evaluation_frequency=1,
        reinitialize_advantage_networks=False,
        reinitialize_imm_regret_networks=True,
        use_baseline=True,
        fit_advantage=True,
        alpha=2.3,
        gamma=2.0,
        device="cpu",
        seed=0,
        logger=Logger(verbose=False),
    )
    solver.evaluate_initial_policy = True
    solver.early_evaluation_node_thresholds = (10,)
    return solver


def test_tiny_solver_preserves_q_model_and_records_safety_invariants():
    solver = _tiny_solver(num_iterations=2)
    online_model_id = id(solver.q_value_trainer.model)
    rows = solver.solve()

    assert id(solver.q_value_trainer.model) == online_model_id
    assert solver.q_value_trainer.target_version == 4
    assert [row["checkpoint_kind"] for row in rows] == [
        "initial_untrained_policy",
        "early_node_threshold",
        "outer_iteration",
        "outer_iteration",
    ]
    for row in rows[1:]:
        assert 0.0 <= row["adaptive_lambda_min"] <= row["adaptive_lambda_max"] <= 1.0
        assert row["full_support_traverser_sampling_min_probability"] >= 1.0 / 3.0
        assert row["policy_weighted_advantage_abs_mean"] < 1e-12
    for online, target in zip(
        solver.q_value_trainer.model.parameters(),
        solver.q_value_trainer.target_model.parameters(),
    ):
        assert torch.equal(online, target)


def test_experiment_config_changes_vr_pdcfr_only_for_new_mechanisms():
    shared_keys = {
        "game_name",
        "advantage_buffer_size",
        "ave_policy_buffer_size",
        "baseline_buffer_size",
        "learning_rate",
        "num_traversals",
        "advantage_network_train_steps",
        "ave_policy_network_train_steps",
        "baseline_network_train_steps",
        "advantage_batch_size",
        "ave_policy_batch_size",
        "baseline_batch_size",
        "num_layers",
        "num_hiddens",
        "reinitialize_advantage_networks",
        "use_regret_matching_argmax",
        "epsilon",
        "fit_advantage",
        "use_baseline",
        "device",
        "evaluation_frequency",
        "max_num_iterations",
        "preserve_evaluation_rng",
    }
    for key in shared_keys:
        assert ADAPTIVE_CONFIG[key] == VR_PAPER_CONFIG[key]
    assert ADAPTIVE_CONFIG["alpha"] == 2.3
    assert ADAPTIVE_CONFIG["gamma"] == 2.0
    assert ADAPTIVE_CONFIG["sampling_mode"] == "fixed_uniform"
    assert ADAPTIVE_CONFIG["evaluate_initial_policy"] is True
    assert ADAPTIVE_CONFIG["early_evaluation_node_thresholds"] == (10_000,)


def test_reference_curves_and_node_targets_are_paired_to_experiment_one():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert EXPERIMENT_1_NODE_TARGETS == {0: 942_635, 1: 939_834, 2: 962_274}
    rows = _load_reference_curves(REFERENCE_CURVES)
    assert len(rows) == 71
    for seed, target in EXPERIMENT_1_NODE_TARGETS.items():
        escher_rows = [
            row
            for row in rows
            if row["algorithm_id"] == "escher_exp28" and row["seed"] == seed
        ]
        assert max(row["nodes_touched"] for row in escher_rows) == target


def test_readme_documents_architecture_state_of_art_case_and_batch_smoke_test():
    readme = (
        Path(__file__).parents[1]
        / "experiments"
        / "leduc_poker"
        / "adaptive_residual_predictive_escher"
        / "README.md"
    ).read_text(encoding="utf-8")
    assert "Why this might be state of the art" in readme
    assert "current return is observed" in readme
    assert "combined_exploitability_by_nodes.png" in readme
    assert "leduc-escher-arch-exp3-adaptive-smoke" in readme
    assert "n2-standard-4 21600 4000 16000 100" in readme

    root_readme = (Path(__file__).parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    assert "Experiment 3 local smoke test" in root_readme
    assert "leduc-escher-arch-exp3-adaptive-smoke" in root_readme
    assert "--early-evaluation-nodes 10" in root_readme
