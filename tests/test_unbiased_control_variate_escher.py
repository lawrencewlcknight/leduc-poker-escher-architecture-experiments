"""Proof-oriented and integration tests for Experiment 6."""

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    REFERENCE_CURVES,
    UNBIASED_CONFIG,
)
from unbiased_escher import (
    UnbiasedControlVariateEscher,
    control_variate_advantage,
    residual_adaptive_sampling_policy,
    variance_optimal_beta,
)
from unbiased_escher.solver import PredictorGateController
from vr_deep_cfr.logger import Logger


def test_control_variate_estimator_is_unbiased_for_arbitrary_beta():
    q_hat = np.array([0.2, -0.5, 0.7])
    true_q = np.array([0.9, -0.1, 0.3])
    beta = np.array([0.0, 0.6, 1.8])
    sample_policy = np.array([0.2, 0.3, 0.5])
    current_policy = np.array([0.5, 0.25, 0.25])
    expected_q = np.zeros(3)
    expected_advantage = np.zeros(3)

    for action, probability in enumerate(sample_policy):
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
        assert estimate.policy_weighted_advantage == pytest.approx(0.0, abs=1e-14)

    np.testing.assert_allclose(expected_q, true_q, atol=1e-12)
    np.testing.assert_allclose(
        expected_advantage,
        true_q - np.dot(current_policy, true_q),
        atol=1e-12,
    )


def test_variance_beta_and_sampling_controls_are_bounded_and_full_support():
    beta = variance_optimal_beta(
        [0.5, 0.0, -0.2],
        [0.5, 100.0, -0.4],
        beta_min=0.0,
        beta_max=2.0,
        ridge=1e-4,
    )
    assert np.all((0.0 <= beta) & (beta <= 2.0))
    assert beta[0] > 1.9
    assert beta[1] == pytest.approx(1.0)

    policy = residual_adaptive_sampling_policy(
        [100.0, 1.0, 0.0],
        [1, 1, 0],
        uniform_floor_mass=0.2,
    )
    assert np.sum(policy) == pytest.approx(1.0)
    assert policy[2] == 0.0
    assert np.min(policy[:2]) >= 0.2 / 2.0


def _tiny_solver(num_iterations=1):
    traversals = 4
    solver = UnbiasedControlVariateEscher(
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
        q_ensemble_size=3,
        calibration_buffer_size=128,
        calibration_batch_size=2,
        calibration_train_steps=1,
    )
    solver.evaluate_initial_policy = True
    solver.early_evaluation_node_thresholds = (10,)
    return solver


def test_cross_fitted_critic_excludes_and_only_trains_on_assigned_fold():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    ensemble.begin_trajectory(1)

    for index, member in enumerate(ensemble.members):
        with torch.no_grad():
            for parameter in member.target_model.parameters():
                parameter.zero_()
            list(member.target_model.parameters())[-1].fill_(float(index + 1))
    mean, disagreement = ensemble.get_baseline_and_disagreement(state, 0)
    for action in state.legal_actions():
        assert mean[action] == pytest.approx(2.0)
        assert disagreement[action] == pytest.approx(1.0)

    next_state = solver.skip_chance_state(state.child(state.legal_actions()[0]))
    ensemble.add_data(
        solver.get_history_tensor(state),
        state.legal_actions()[0],
        solver.get_history_tensor(next_state),
        solver.get_infostate_tensor(next_state),
        next_state.legal_actions_mask(),
        next_state.current_player(),
        0,
        0.0,
    )
    assert ensemble.fold_sizes() == [0, 1, 0]
    assert ensemble.heldout_member_indices() == [0, 2]


def test_prediction_gate_opens_only_when_predictor_beats_zero():
    controller = PredictorGateController(2, ema_decay=0.0, initial_gate=0.0)
    controller.observe(0, prediction_mse=0.25, zero_mse=1.0)
    controller.observe(1, prediction_mse=2.0, zero_mse=1.0)
    assert controller.value(0) == pytest.approx(0.75)
    assert controller.value(1) == 0.0


def test_tiny_solver_records_unbiased_architecture_invariants():
    solver = _tiny_solver()
    rows = solver.solve()
    assert [row["checkpoint_kind"] for row in rows] == [
        "initial_untrained_policy",
        "early_node_threshold",
        "outer_iteration",
    ]
    final = rows[-1]
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert final["full_support_sampling_min_probability"] >= 0.2 / 3.0
    assert final["q_ensemble_target_version_min"] == 1
    assert final["q_ensemble_target_version_max"] == 1
    assert final["calibration_target_version"] == 1
    assert sum(
        final[f"q_fold_{fold}_replay_size"] for fold in range(3)
    ) == final["unbiased_estimator_sample_count"]


def test_experiment_6_uses_experiment_2_horizon_and_documents_provenance():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert EXPERIMENT_2_NODE_TARGETS == {
        0: 4_700_205,
        1: 4_701_540,
        2: 4_684_695,
    }
    assert UNBIASED_CONFIG["q_ensemble_size"] == 3
    assert UNBIASED_CONFIG["sampling_uniform_floor_mass"] == 0.2
    assert BATCH_TIMEOUT_SECONDS == 129_600
    assert REFERENCE_CURVES.exists()

    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "unbiased_control_variate_escher_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "unbiased_control_variate_escher_5x_nodes.run" in readme
        assert "leduc-escher-arch-exp6-unbiased-cv-smoke" in readme
        assert "--calibration-train-steps 1" in readme
    assert "n2-standard-8 129600 8000 32000 100" in experiment_readme
