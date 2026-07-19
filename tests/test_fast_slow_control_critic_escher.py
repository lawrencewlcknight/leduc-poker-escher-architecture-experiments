"""Correctness and integration tests for Experiment 9."""

from pathlib import Path

import numpy as np
import pytest

from experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes import (
    config as experiment_config,
    run as experiment_run,
)
from fast_slow_escher import FastSlowControlCriticEscher
from fast_slow_escher.solver import ReservoirTransitionBuffer
from vr_deep_cfr.logger import Logger


def _tiny_solver():
    solver = FastSlowControlCriticEscher(
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
        fast_q_buffer_size=128,
        fast_q_train_steps=1,
        rho_buffer_size=128,
        rho_batch_size=2,
        rho_train_steps=1,
    )
    solver.max_num_iterations = 1
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    return solver


def _one_transition(solver, state):
    action = state.legal_actions()[0]
    next_state = solver.skip_chance_state(state.child(action))
    return (
        solver.get_history_tensor(state),
        action,
        solver.get_history_tensor(next_state),
        solver.get_infostate_tensor(next_state),
        next_state.legal_actions_mask(),
        next_state.current_player(),
        0,
        0.0,
    )


def test_both_timescales_are_strictly_cross_fitted_by_trajectory():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    ensemble.begin_iteration(1)
    assert ensemble.begin_trajectory(1) == 1
    assert ensemble.heldout_member_indices() == [0, 2]
    ensemble.add_data(*_one_transition(solver, state))
    assert ensemble.fast_fold_sizes() == [0, 1, 0]
    assert ensemble.fold_sizes() == [0, 1, 0]


def test_fast_replay_resets_each_iteration_but_slow_reservoir_persists():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    ensemble.begin_iteration(1)
    ensemble.begin_trajectory(0)
    ensemble.add_data(*_one_transition(solver, state))
    assert sum(ensemble.fast_fold_sizes()) == 1
    assert sum(ensemble.fold_sizes()) == 1
    ensemble.begin_iteration(2)
    assert sum(ensemble.fast_fold_sizes()) == 0
    assert sum(ensemble.fold_sizes()) == 1


def test_slow_transition_buffer_counts_lifetime_stream_beyond_capacity():
    buffer = ReservoirTransitionBuffer(2, 2, 2, 2, "cpu")
    for index in range(7):
        buffer.add(
            [index, index],
            0,
            [index + 1, index + 1],
            [0, 0],
            [1, 1],
            0,
            0,
            float(index),
        )
    assert len(buffer) == 2
    assert buffer.seen_count == 7
    assert len(buffer.sample(-1)[0]) == 2


def test_initial_controller_is_convex_and_chosen_before_return():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    ensemble.begin_iteration(1)
    ensemble.begin_trajectory(2)
    _, _ = ensemble.get_baseline_and_disagreement(state, 0)
    diagnostics = ensemble.diagnostics()
    assert diagnostics["fast_slow_rho_mean"] == pytest.approx(0.5)
    assert diagnostics["rho_controller_target_version"] == 0
    assert len(ensemble.rho_controller.buffer) == 0
    ensemble.observe_control_return(
        state=state,
        player=0,
        action=state.legal_actions()[0],
        sampled_return=0.25,
        iteration=1,
    )
    assert len(ensemble.rho_controller.buffer) == 1
    assert diagnostics["rho_controller_target_version"] == 0


def test_tiny_solver_records_fast_slow_and_unbiasedness_invariants():
    solver = _tiny_solver()
    rows = solver.solve()
    final = rows[-1]
    assert 0.0 <= final["fast_slow_rho_min"]
    assert final["fast_slow_rho_max"] <= 1.0
    assert final["rho_controller_target_version"] == 1
    assert final["fast_critic_target_version_min"] == 1
    assert final["slow_critic_target_version_min"] == 1
    assert final["rho_controller_replay_size"] > 0
    assert final["slow_critic_lifetime_seen_count"] > 0
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert sum(final[f"fast_q_fold_{fold}_replay_size"] for fold in range(3)) > 0
    assert sum(final[f"q_fold_{fold}_replay_size"] for fold in range(3)) > 0


def test_experiment_9_reference_is_immutable_and_node_matched():
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
    assert experiment_config.FAST_SLOW_CONFIG["q_ensemble_size"] == 3
    assert experiment_config.FAST_SLOW_CONFIG["fast_q_train_steps"] == 5_000
    assert experiment_config.BATCH_TIMEOUT_SECONDS == 172_800


def test_experiment_9_readmes_document_smoke_test_and_provenance():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "fast_slow_control_critic_escher_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "fast_slow_control_critic_escher_5x_nodes.run" in readme
        assert "leduc-escher-arch-exp9-fast-slow-smoke" in readme
        assert "--rho-train-steps 1" in readme
    assert "n2-standard-8 172800 8000 32000 100" in experiment_readme
    assert experiment_config.REFERENCE_CURVES_SHA256 in experiment_readme

