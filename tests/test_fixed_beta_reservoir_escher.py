"""Correctness and experiment-contract tests for Experiments 13 and 14."""

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.leduc_poker import fixed_beta_reservoir_shared as common
from experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes import (
    config as experiment_13_config,
    run as experiment_13_run,
)
from experiments.leduc_poker.fixed_beta_reservoir_escher_15m_nodes import (
    config as experiment_14_config,
)
from fixed_beta_reservoir_escher import FixedBetaReservoirEscher
from fast_slow_escher.solver import ReservoirTransitionBuffer
from vr_deep_cfr.logger import Logger


ROOT = Path(__file__).parents[1]


def _tiny_solver():
    solver = FixedBetaReservoirEscher(
        game_name="leduc_poker",
        num_episodes=8,
        advantage_buffer_size=128,
        ave_policy_buffer_size=128,
        baseline_buffer_size=12,
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
        fixed_control_variate_beta=1.0,
        calibration_buffer_size=128,
        calibration_batch_size=2,
        calibration_train_steps=1,
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


def test_solver_enforces_fixed_beta_one():
    with pytest.raises(ValueError, match="requires beta=1"):
        FixedBetaReservoirEscher(
            game_name="leduc_poker",
            use_baseline=True,
            fit_advantage=True,
            fixed_control_variate_beta=0.5,
        )


def test_reservoir_critics_remain_strictly_cross_fitted():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    assert all(
        isinstance(member.buffer, ReservoirTransitionBuffer)
        for member in ensemble.members
    )
    state = solver.skip_chance_state(solver.game.new_initial_state())
    assert ensemble.begin_trajectory(1) == 1
    assert ensemble.heldout_member_indices() == [0, 2]
    ensemble.add_data(*_one_transition(solver, state))
    assert ensemble.fold_sizes() == [0, 1, 0]
    assert ensemble.fold_lifetime_seen_counts() == [0, 1, 0]


def test_reservoir_tracks_lifetime_stream_after_reaching_capacity():
    solver = _tiny_solver()
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    for index in range(20):
        ensemble.begin_trajectory(0)
        ensemble.add_data(*_one_transition(solver, state))
    assert ensemble.fold_sizes()[0] == 4
    assert ensemble.fold_lifetime_seen_counts()[0] == 20


def test_tiny_solver_records_fixed_beta_and_lifetime_diagnostics():
    solver = _tiny_solver()
    rows = solver.solve()
    final = rows[-1]
    assert final["control_variate_beta_min"] == pytest.approx(1.0)
    assert final["control_variate_beta_max"] == pytest.approx(1.0)
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert final["q_lifetime_seen_count"] > 0
    assert sum(
        final[f"q_fold_{fold}_lifetime_seen_count"] for fold in range(3)
    ) == final["q_lifetime_seen_count"]
    assert not hasattr(solver.q_value_trainer, "fast_members")
    assert not hasattr(solver.q_value_trainer, "rho_controller")


def test_experiment_13_is_node_matched_and_reuses_immutable_experiment_6():
    curves = common.load_reference_curves(
        experiment_13_config.REFERENCE_CURVES,
        expected_sha256=experiment_13_config.REFERENCE_CURVES_SHA256,
        expected_rows=experiment_13_config.REFERENCE_CURVE_ROWS,
        expected_algorithm_ids=(experiment_13_config.REFERENCE_ALGORITHM_ID,),
        expected_seeds=experiment_13_config.DEFAULT_SEEDS,
        result_source="test",
    )
    summaries = common.load_reference_summaries(
        experiment_13_config.REFERENCE_SUMMARIES,
        expected_sha256=experiment_13_config.REFERENCE_SUMMARIES_SHA256,
        expected_rows=experiment_13_config.REFERENCE_SUMMARY_ROWS,
        expected_algorithm_ids=(experiment_13_config.REFERENCE_ALGORITHM_ID,),
        expected_seeds=experiment_13_config.DEFAULT_SEEDS,
        result_source="test",
    )
    assert len(curves) == 90
    assert len(summaries) == 3
    assert experiment_13_config.EXPERIMENT_2_NODE_TARGETS == {
        0: 4_700_205,
        1: 4_701_540,
        2: 4_684_695,
    }
    assert experiment_13_config.CANDIDATE_CONFIG[
        "fixed_control_variate_beta"
    ] == 1.0
    assert experiment_13_config.CANDIDATE_CONFIG["q_ensemble_size"] == 3


def test_experiment_14_reuses_all_experiment_7_results_at_15m_nodes():
    curves = common.load_reference_curves(
        experiment_14_config.REFERENCE_CURVES,
        expected_sha256=experiment_14_config.REFERENCE_CURVES_SHA256,
        expected_rows=experiment_14_config.REFERENCE_CURVE_ROWS,
        expected_algorithm_ids=experiment_14_config.REFERENCE_ALGORITHM_IDS,
        expected_seeds=experiment_14_config.DEFAULT_SEEDS,
        result_source="test",
    )
    summaries = common.load_reference_summaries(
        experiment_14_config.REFERENCE_SUMMARIES,
        expected_sha256=experiment_14_config.REFERENCE_SUMMARIES_SHA256,
        expected_rows=experiment_14_config.REFERENCE_SUMMARY_ROWS,
        expected_algorithm_ids=experiment_14_config.REFERENCE_ALGORITHM_IDS,
        expected_seeds=experiment_14_config.DEFAULT_SEEDS,
        result_source="test",
    )
    assert len(curves) == 862
    assert len(summaries) == 9
    assert experiment_14_config.TARGET_NODES == 15_000_000
    assert experiment_14_config.CANDIDATE_CONFIG["max_num_iterations"] == 120
    assert {row["algorithm_id"] for row in summaries} == set(
        experiment_14_config.REFERENCE_ALGORITHM_IDS
    )


def test_smoke_overrides_preserve_architecture_defaults():
    args = experiment_13_run._parser().parse_args(
        [
            "--traversals",
            "4",
            "--max-iterations",
            "2",
            "--q-train-steps",
            "1",
            "--calibration-train-steps",
            "1",
            "--batch-size",
            "2",
            "--buffer-size",
            "128",
            "--early-evaluation-nodes",
            "10",
        ]
    )
    config = deepcopy(experiment_13_config.CANDIDATE_CONFIG)
    experiment_13_run._apply_overrides(args, config)
    assert config["num_traversals"] == 4
    assert config["max_num_iterations"] == 2
    assert config["baseline_network_train_steps"] == 1
    assert config["calibration_train_steps"] == 1
    assert config["baseline_buffer_size"] == 128
    assert config["early_evaluation_node_thresholds"] == (10,)
    assert config["fixed_control_variate_beta"] == 1.0
    assert experiment_13_config.CANDIDATE_CONFIG["baseline_buffer_size"] == 1_000_000


def test_runtime_and_batch_timeouts_have_headroom():
    assert experiment_13_config.EXPECTED_SEQUENTIAL_RUNTIME_HOURS == 12
    assert experiment_13_config.BATCH_TIMEOUT_SECONDS == 24 * 60 * 60
    assert experiment_14_config.EXPECTED_SEQUENTIAL_RUNTIME_HOURS == 36
    assert experiment_14_config.BATCH_TIMEOUT_SECONDS == 48 * 60 * 60
    assert experiment_13_config.BATCH_TIMEOUT_SECONDS > (
        experiment_13_config.EXPECTED_SEQUENTIAL_RUNTIME_HOURS * 3600
    )
    assert experiment_14_config.BATCH_TIMEOUT_SECONDS > (
        experiment_14_config.EXPECTED_SEQUENTIAL_RUNTIME_HOURS * 3600
    )


def test_readmes_document_full_and_smoke_batch_jobs():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    specs = {
        13: {
            "directory": "fixed_beta_reservoir_escher_5x_nodes",
            "module": "fixed_beta_reservoir_escher_5x_nodes.run",
            "job": "leduc-escher-arch-exp13-reservoir-smoke",
            "timeout": "86400",
        },
        14: {
            "directory": "fixed_beta_reservoir_escher_15m_nodes",
            "module": "fixed_beta_reservoir_escher_15m_nodes.run",
            "job": "leduc-escher-arch-exp14-reservoir-15m-smoke",
            "timeout": "172800",
        },
    }
    for experiment, spec in specs.items():
        experiment_readme = (
            ROOT
            / "experiments"
            / "leduc_poker"
            / spec["directory"]
            / "README.md"
        ).read_text(encoding="utf-8")
        for readme in (root_readme, experiment_readme):
            assert spec["module"] in readme
            assert spec["job"] in readme
            assert "--seeds 0" in readme
        assert "Full single GCP Batch job" in experiment_readme
        assert spec["timeout"] in experiment_readme
