"""Correctness and integration tests for Experiment 10."""

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes import (
    config as experiment_config,
    run as experiment_run,
)
from monte_carlo_escher import MonteCarloControlCriticEscher
from vr_deep_cfr.logger import Logger


def _tiny_solver():
    solver = MonteCarloControlCriticEscher(
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


def test_mc_returns_are_cross_fitted_and_use_player_zero_value_convention():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    action = state.legal_actions()[0]
    ensemble.begin_iteration(1)
    assert ensemble.begin_trajectory(1) == 1
    assert ensemble.heldout_member_indices() == [0, 2]
    ensemble.observe_control_return(
        state=state,
        player=1,
        action=action,
        sampled_return=0.25,
        iteration=1,
    )
    assert ensemble.fold_sizes() == [0, 1, 0]
    assert ensemble.members[1].buffer.returns[0] == pytest.approx(-0.25)
    assert ensemble.fold_seen_counts() == [0, 1, 0]


def test_one_step_transition_path_is_discarded_and_buffers_are_iteration_local():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    ensemble.begin_iteration(1)
    ensemble.begin_trajectory(0)
    ensemble.observe_control_return(
        state=state,
        player=0,
        action=state.legal_actions()[0],
        sampled_return=0.1,
        iteration=1,
    )
    assert sum(ensemble.fold_sizes()) == 1
    ensemble.add_data("one-step", "transition", "must", "be", "ignored")
    assert sum(ensemble.fold_sizes()) == 1
    ensemble.begin_iteration(2)
    assert sum(ensemble.fold_sizes()) == 0
    assert sum(ensemble.fold_seen_counts()) == 0


def test_direct_regression_never_invokes_td_bootstrapping():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    member = ensemble.members[0]
    state = solver.skip_chance_state(solver.game.new_initial_state())
    member.add_monte_carlo_target(
        member.get_history_tensor(state),
        state.legal_actions()[0],
        0.2,
    )
    member.add_monte_carlo_target(
        member.get_history_tensor(state),
        state.legal_actions()[1],
        -0.1,
    )

    def fail_if_td_is_used(*args, **kwargs):
        del args, kwargs
        raise AssertionError("TD continuation computation was called")

    member._batched_predictive_strategies = fail_if_td_is_used
    assert np.isfinite(member.train_model(1))
    assert member.target_version == 1


def test_both_players_collect_before_either_regret_network_is_fitted():
    solver = _tiny_solver()
    events = []
    solver.collect_training_data = lambda player: events.append(("collect", player))
    solver.train_regret = lambda player: events.append(("train_regret", player))
    solver._predictor_holdout_error = lambda player: (np.nan, np.nan)
    solver.calibration_trainer.train_model = lambda: None
    solver.q_value_trainer.train_model = lambda iteration: None
    solver.evaluation_frequency = 99
    solver.iteration()
    assert events == [
        ("collect", 0),
        ("collect", 1),
        ("train_regret", 0),
        ("train_regret", 1),
    ]


def test_cross_fitted_prediction_excludes_the_active_mc_fold():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    ensemble.begin_iteration(1)
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


def test_tiny_solver_records_mc_and_unbiasedness_invariants():
    solver = _tiny_solver()
    rows = solver.solve()
    final = rows[-1]
    assert final["mc_target_count"] > 0
    assert final["mc_target_version_min"] == 1
    assert final["mc_target_version_max"] == 1
    assert final["mc_target_variance"] >= 0.0
    assert final["mc_target_abs_max"] > 0.0
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert sum(final[f"mc_fold_{fold}_seen_count"] for fold in range(3)) == (
        final["mc_target_count"]
    )
    assert sum(final[f"q_fold_{fold}_replay_size"] for fold in range(3)) == (
        final["mc_target_count"]
    )


def test_experiment_10_reference_is_immutable_and_node_matched():
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
    assert experiment_config.MONTE_CARLO_CONFIG["q_ensemble_size"] == 3
    assert experiment_config.BATCH_TIMEOUT_SECONDS == 86_400


def test_experiment_10_readmes_document_smoke_test_and_provenance():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "monte_carlo_control_critic_escher_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "monte_carlo_control_critic_escher_5x_nodes.run" in readme
        assert "leduc-escher-arch-exp10-mc-critic-smoke" in readme
        assert "--calibration-train-steps 1" in readme
    assert "n2-standard-8 86400 8000 32000 100" in experiment_readme
    assert experiment_config.REFERENCE_CURVES_SHA256 in experiment_readme

