"""Proof-oriented and integration tests for Experiment 11."""

from pathlib import Path

import numpy as np
import pytest

from advantage_sampling_escher import (
    AdvantageVarianceSamplingEscher,
    advantage_variance_sampling_decision,
)
from experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes import (
    config as experiment_config,
    run as experiment_run,
)
from unbiased_escher import (
    UnbiasedControlVariateEscher,
    control_variate_advantage,
    residual_adaptive_sampling_policy,
)
from vr_deep_cfr.logger import Logger


def _tiny_solver(solver_class=AdvantageVarianceSamplingEscher):
    solver = solver_class(
        game_name="leduc_poker",
        num_episodes=8,
        advantage_buffer_size=128,
        ave_policy_buffer_size=128,
        baseline_buffer_size=128,
        learning_rate=1e-3,
        num_traversals=4,
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
        q_ensemble_size=3,
        calibration_buffer_size=128,
        calibration_batch_size=2,
        calibration_train_steps=1,
    )
    solver.max_num_iterations = 1
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    return solver


def test_centering_influence_norm_matches_explicit_matrix_columns():
    policy = np.array([0.6, 0.3, 0.1])
    decision = advantage_variance_sampling_decision(
        [0.4, -0.2, 0.7],
        beta=[1.2, 0.5, 0.8],
        predicted_residual_means=[0.1, -0.3, 0.2],
        predicted_residual_variances=[0.8, 0.4, 1.1],
        current_policy=policy,
        legal_actions_mask=[1, 1, 1],
        uniform_floor_mass=0.2,
    )
    centering = np.eye(3) - np.ones((3, 1)) @ policy.reshape(1, -1)
    expected = np.linalg.norm(centering, axis=0)
    np.testing.assert_allclose(decision.centering_influence_norms, expected)
    np.testing.assert_allclose(
        np.square(expected),
        1.0 - 2.0 * policy + 3.0 * np.square(policy),
    )


def test_sampling_rule_uses_actual_control_residual_second_moment_and_floor():
    q_values = np.array([0.4, -0.2, 0.7])
    beta = np.array([1.2, 0.5, 0.8])
    residual_mean = np.array([0.1, -0.3, 0.2])
    residual_variance = np.array([0.8, 0.4, 1.1])
    policy = np.array([0.6, 0.3, 0.1])
    floor = 0.2
    decision = advantage_variance_sampling_decision(
        q_values,
        beta=beta,
        predicted_residual_means=residual_mean,
        predicted_residual_variances=residual_variance,
        current_policy=policy,
        legal_actions_mask=[1, 1, 1],
        uniform_floor_mass=floor,
    )
    expected_second_moment = residual_variance + np.square(
        residual_mean + (1.0 - beta) * q_values
    )
    expected_influence = np.sqrt(
        1.0 - 2.0 * policy + 3.0 * np.square(policy)
    )
    expected_score = np.sqrt(expected_second_moment) * expected_influence
    expected_sampling = (
        (1.0 - floor) * expected_score / np.sum(expected_score)
        + floor / 3.0
    )
    np.testing.assert_allclose(
        decision.control_residual_second_moments,
        expected_second_moment,
    )
    np.testing.assert_allclose(decision.scores, expected_score)
    np.testing.assert_allclose(decision.policy, expected_sampling)
    assert np.sum(decision.policy) == pytest.approx(1.0)
    assert np.min(decision.policy) >= floor / 3.0


def test_sampler_preserves_unbiased_q_and_centred_advantage_estimates():
    q_hat = np.array([0.2, -0.5, 0.7])
    true_q = np.array([0.9, -0.1, 0.3])
    beta = np.array([0.0, 0.6, 1.8])
    current_policy = np.array([0.5, 0.25, 0.25])
    sampling = advantage_variance_sampling_decision(
        q_hat,
        beta=beta,
        predicted_residual_means=true_q - q_hat,
        predicted_residual_variances=[0.2, 0.4, 0.8],
        current_policy=current_policy,
        legal_actions_mask=[1, 1, 1],
        uniform_floor_mass=0.2,
    ).policy
    expected_q = np.zeros(3)
    expected_advantage = np.zeros(3)
    for action, probability in enumerate(sampling):
        estimate = control_variate_advantage(
            q_hat,
            beta=beta,
            sampled_action=action,
            sample_probability=float(probability),
            sampled_return=float(true_q[action]),
            policy=current_policy,
            legal_actions_mask=[1, 1, 1],
        )
        expected_q += probability * estimate.q_values
        expected_advantage += probability * estimate.advantages
    np.testing.assert_allclose(expected_q, true_q, atol=1e-12)
    np.testing.assert_allclose(
        expected_advantage,
        true_q - np.dot(current_policy, true_q),
        atol=1e-12,
    )


def test_experiment_6_default_sampling_hook_is_unchanged():
    solver = _tiny_solver(UnbiasedControlVariateEscher)
    variances = np.array([4.0, 1.0, 0.0])
    legal_mask = np.array([1.0, 1.0, 0.0])
    actual = solver._traverser_sampling_policy(
        q_values=np.array([7.0, -9.0, 0.0]),
        beta=np.array([0.2, 1.8, 1.0]),
        residual_means=np.array([10.0, -20.0, 0.0]),
        predicted_variances=variances,
        policy=np.array([0.9, 0.1, 0.0]),
        legal_mask=legal_mask,
    )
    expected = residual_adaptive_sampling_policy(
        variances,
        legal_mask,
        uniform_floor_mass=solver.sampling_uniform_floor_mass,
        minimum_variance=solver.calibration_minimum_variance,
    )
    np.testing.assert_allclose(actual, expected)


def test_tiny_solver_records_sampling_and_unbiasedness_invariants():
    solver = _tiny_solver()
    rows = solver.solve()
    final = rows[-1]
    assert final["advantage_sampler_state_count"] > 0
    assert final["advantage_sampling_probability_min"] >= 0.2 / 3.0
    assert final["advantage_sampling_probability_max"] <= 1.0
    assert final["predicted_advantage_variance_proxy_mean"] > 0.0
    assert np.isfinite(
        final["predicted_advantage_variance_proxy_ratio_vs_experiment_6"]
    )
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert sum(
        final[f"q_fold_{fold}_replay_size"] for fold in range(3)
    ) == final["unbiased_estimator_sample_count"]


def test_experiment_11_reference_is_immutable_and_node_matched():
    curves = experiment_run._load_reference_curves(
        experiment_config.REFERENCE_CURVES
    )
    summaries = experiment_run._load_reference_summaries(
        experiment_config.REFERENCE_SUMMARIES
    )
    assert len(curves) == experiment_config.REFERENCE_CURVE_ROWS == 90
    assert len(summaries) == experiment_config.REFERENCE_SUMMARY_ROWS == 3
    assert experiment_config.EXPERIMENT_2_NODE_TARGETS == {
        0: 4_700_205,
        1: 4_701_540,
        2: 4_684_695,
    }
    assert experiment_config.ADVANTAGE_SAMPLING_CONFIG[
        "sampling_uniform_floor_mass"
    ] == 0.2
    assert experiment_config.BATCH_TIMEOUT_SECONDS == 86_400


def test_experiment_11_readmes_document_smoke_test_and_provenance():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "advantage_variance_sampling_escher_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "advantage_variance_sampling_escher_5x_nodes.run" in readme
        assert "leduc-escher-arch-exp11-adv-sampling-smoke" in readme
        assert "--calibration-train-steps 1" in readme
    assert "n2-standard-8 86400 8000 32000 100" in experiment_readme
    assert experiment_config.REFERENCE_CURVES_SHA256 in experiment_readme
