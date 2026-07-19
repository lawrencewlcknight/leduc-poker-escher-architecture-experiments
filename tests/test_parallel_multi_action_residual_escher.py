"""Proof-oriented and integration tests for Experiment 12."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes import (
    config as experiment_config,
    run as experiment_run,
)
from parallel_multi_action_escher import (
    CoupledRolloutStreams,
    ParallelMultiActionResidualEscher,
    adaptive_nonempty_subset,
    multi_action_control_variate_advantage,
)
from vr_deep_cfr.logger import Logger


def _decision(variances, seed=7):
    return adaptive_nonempty_subset(
        [0.2, -0.5, 0.7],
        beta=[0.0, 0.6, 1.8],
        predicted_residual_means=[0.7, 0.4, -0.4],
        predicted_residual_variances=variances,
        current_policy=[0.5, 0.25, 0.25],
        legal_actions_mask=[1, 1, 1],
        uniform_floor_mass=0.2,
        rollout_cost_scale=2.0,
        rng=np.random.default_rng(seed),
        minimum_variance=1e-5,
    )


def _tiny_solver(parallel_workers=3):
    solver = ParallelMultiActionResidualEscher(
        game_name="leduc_poker",
        num_episodes=8,
        advantage_buffer_size=256,
        ave_policy_buffer_size=256,
        baseline_buffer_size=256,
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
        calibration_buffer_size=256,
        calibration_batch_size=2,
        calibration_train_steps=1,
        subset_rollout_cost_scale=2.0,
        parallel_action_workers=parallel_workers,
    )
    solver.max_num_iterations = 1
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    return solver


def test_conditioned_bernoulli_marginals_and_nonempty_support_are_exact():
    decision = _decision([0.2, 0.4, 0.8])
    raw = decision.raw_inclusion_probabilities
    nonempty_probability = 1.0 - float(np.prod(1.0 - raw))
    np.testing.assert_allclose(
        decision.inclusion_probabilities,
        raw / nonempty_probability,
    )
    total_probability = 0.0
    marginals = np.zeros(3)
    for mask in product((False, True), repeat=3):
        if not any(mask):
            continue
        probability = 1.0
        for action, included in enumerate(mask):
            probability *= raw[action] if included else 1.0 - raw[action]
        probability /= nonempty_probability
        total_probability += probability
        marginals += probability * np.asarray(mask, dtype=float)
    assert total_probability == pytest.approx(1.0)
    np.testing.assert_allclose(marginals, decision.inclusion_probabilities)
    assert decision.selected_actions
    assert np.min(decision.inclusion_probabilities) >= 0.2 / 3.0


def test_multi_action_estimator_is_unbiased_over_every_possible_subset():
    q_hat = np.array([0.2, -0.5, 0.7])
    true_q = np.array([0.9, -0.1, 0.3])
    beta = np.array([0.0, 0.6, 1.8])
    policy = np.array([0.5, 0.25, 0.25])
    decision = _decision([0.2, 0.4, 0.8])
    raw = decision.raw_inclusion_probabilities
    nonempty_probability = 1.0 - float(np.prod(1.0 - raw))
    expected_q = np.zeros(3)
    expected_advantage = np.zeros(3)
    probability_sum = 0.0
    for mask in product((False, True), repeat=3):
        if not any(mask):
            continue
        probability = 1.0
        selected_returns = {}
        for action, included in enumerate(mask):
            probability *= raw[action] if included else 1.0 - raw[action]
            if included:
                selected_returns[action] = float(true_q[action])
        probability /= nonempty_probability
        estimate = multi_action_control_variate_advantage(
            q_hat,
            beta=beta,
            selected_returns=selected_returns,
            inclusion_probabilities=decision.inclusion_probabilities,
            policy=policy,
            legal_actions_mask=[1, 1, 1],
        )
        assert estimate.policy_weighted_advantage == pytest.approx(0.0, abs=1e-14)
        expected_q += probability * estimate.q_values
        expected_advantage += probability * estimate.advantages
        probability_sum += probability
    assert probability_sum == pytest.approx(1.0)
    np.testing.assert_allclose(expected_q, true_q, atol=1e-12)
    np.testing.assert_allclose(
        expected_advantage,
        true_q - np.dot(policy, true_q),
        atol=1e-12,
    )


def test_subset_expands_monotonically_with_predicted_regret_noise():
    low = _decision([0.01, 0.01, 0.01], seed=2)
    high = _decision([9.0, 9.0, 9.0], seed=2)
    assert high.expected_subset_size > low.expected_subset_size
    assert np.all(
        high.raw_inclusion_probabilities >= low.raw_inclusion_probabilities
    )


def test_coupled_streams_align_chance_and_opponent_noise_independently():
    original = CoupledRolloutStreams.from_seed(1234)
    sibling = original.clone()
    np.testing.assert_allclose(original.chance.random(8), sibling.chance.random(8))
    np.testing.assert_allclose(
        original.opponent.random(8),
        sibling.opponent.random(8),
    )

    separated = CoupledRolloutStreams.from_seed(9876)
    reference = separated.clone()
    separated.subset.random(100)
    assert separated.chance.random() == pytest.approx(reference.chance.random())
    assert separated.opponent.random() == pytest.approx(reference.opponent.random())


def test_tiny_solver_executes_parallel_subsets_and_records_invariants():
    solver = _tiny_solver(parallel_workers=3)
    rows = solver.solve()
    final = rows[-1]
    assert final["subset_information_set_count"] > 0
    assert final["sampled_subset_size_mean"] >= 1.0
    assert final["sampled_subset_size_max"] <= 3.0
    assert final["multi_action_information_set_fraction"] > 0.0
    assert final["action_inclusion_probability_min"] >= 0.2 / 3.0
    assert final["actual_parallel_action_batch_count"] > 0
    assert final["common_random_number_group_count"] >= (
        final["actual_parallel_action_batch_count"]
    )
    assert final["ideal_parallel_node_speedup"] >= 1.0
    assert 0.0 <= final["ideal_parallelisable_node_fraction"] < 1.0
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert sum(
        final[f"q_fold_{fold}_replay_size"] for fold in range(3)
    ) == final["unbiased_estimator_sample_count"]


def test_parallel_and_serial_branch_execution_have_identical_training_results():
    serial_rows = _tiny_solver(parallel_workers=1).solve()
    parallel_rows = _tiny_solver(parallel_workers=3).solve()
    serial = serial_rows[-1]
    parallel = parallel_rows[-1]
    for field in (
        "nodes_touched",
        "episode",
        "exp",
        "average_policy_value",
        "unbiased_estimator_sample_count",
        "sampled_subset_size_mean",
        "expected_subset_size_mean",
        "common_random_number_group_count",
    ):
        assert parallel[field] == pytest.approx(serial[field], abs=1e-12)
    assert serial["actual_parallel_action_batch_count"] == 0
    assert parallel["actual_parallel_action_batch_count"] > 0


def test_experiment_12_reference_is_immutable_and_node_matched():
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
    assert experiment_config.PARALLEL_MULTI_ACTION_CONFIG[
        "parallel_action_workers"
    ] == 3
    assert experiment_config.PARALLEL_MULTI_ACTION_CONFIG[
        "subset_rollout_cost_scale"
    ] == 2.0
    assert experiment_config.BATCH_TIMEOUT_SECONDS == 86_400


def test_experiment_12_readmes_document_smoke_test_and_provenance():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "parallel_multi_action_residual_escher_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "parallel_multi_action_residual_escher_5x_nodes.run" in readme
        assert "leduc-escher-arch-exp12-multi-action-smoke" in readme
        assert "--parallel-action-workers 3" in readme
    assert "n2-standard-8 86400 8000 32000 100" in experiment_readme
    assert experiment_config.REFERENCE_CURVES_SHA256 in experiment_readme
