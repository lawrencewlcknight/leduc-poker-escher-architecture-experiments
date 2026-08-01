"""Correctness and experiment-contract tests for Experiment 15."""

from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest

from experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes import (
    config as experiment_config,
    run as experiment_run,
)
from fast_slow_escher.solver import (
    IsolatedCircularTransitionBuffer,
    ReservoirTransitionBuffer,
    RhoReplayBuffer,
)
from fixed_beta_fast_slow_escher import (
    FixedBetaFastSlowControlCriticEscher,
)
from vr_deep_cfr.logger import Logger


ROOT = Path(__file__).parents[1]


def _tiny_solver(seed=0):
    solver = FixedBetaFastSlowControlCriticEscher(
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
        seed=seed,
        logger=Logger(verbose=False),
        q_ensemble_size=3,
        fixed_control_variate_beta=1.0,
        calibration_buffer_size=128,
        calibration_batch_size=2,
        calibration_train_steps=1,
        fast_q_buffer_size=12,
        fast_q_train_steps=1,
        rho_buffer_size=128,
        rho_batch_size=2,
        rho_train_steps=1,
    )
    solver.max_num_iterations = 1
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    return solver


def _transition(index):
    return (
        [index, index],
        index % 2,
        [index + 1, index + 1],
        [0, 0],
        [1, 1],
        0,
        0,
        float(index),
    )


def test_solver_enforces_fixed_beta_one():
    with pytest.raises(ValueError, match="requires beta=1"):
        FixedBetaFastSlowControlCriticEscher(
            game_name="leduc_poker",
            use_baseline=True,
            fit_advantage=True,
            fixed_control_variate_beta=0.5,
        )


def test_complete_fast_slow_architecture_uses_isolated_replay_streams():
    solver = _tiny_solver(seed=2)
    ensemble = solver.q_value_trainer
    assert ensemble.isolated_replay_rng
    assert ensemble.replay_rng_seed == 1_200_009
    assert len(ensemble.fast_members) == 3
    assert len(ensemble.slow_members) == 3
    assert all(
        isinstance(member.buffer, IsolatedCircularTransitionBuffer)
        for member in ensemble.fast_members
    )
    assert all(member.buffer.rng is not None for member in ensemble.fast_members)
    assert all(
        isinstance(member.buffer, ReservoirTransitionBuffer)
        for member in ensemble.slow_members
    )
    assert all(member.buffer.rng is not None for member in ensemble.slow_members)
    assert ensemble.rho_controller.buffer.rng is not None


def test_control_replay_operations_do_not_advance_global_python_rng():
    random.seed(8128)
    expected_state = random.getstate()

    slow = ReservoirTransitionBuffer(
        3,
        2,
        2,
        2,
        "cpu",
        rng=random.Random(10),
    )
    fast = IsolatedCircularTransitionBuffer(
        3,
        2,
        2,
        2,
        "cpu",
        rng=random.Random(20),
    )
    rho = RhoReplayBuffer(4, 2, rng=random.Random(30))
    for index in range(10):
        slow.add(*_transition(index))
        fast.add(*_transition(index))
        rho.add([index, index], index, index + 1, index + 2)
    slow.sample(2)
    fast.sample(2)
    rho.sample(2, "cpu")

    assert random.getstate() == expected_state


def test_component_local_replay_is_deterministic():
    first = ReservoirTransitionBuffer(
        5,
        2,
        2,
        2,
        "cpu",
        rng=random.Random(99),
    )
    second = ReservoirTransitionBuffer(
        5,
        2,
        2,
        2,
        "cpu",
        rng=random.Random(99),
    )
    for index in range(100):
        first.add(*_transition(index))
        second.add(*_transition(index))
    np.testing.assert_array_equal(first.history, second.history)
    np.testing.assert_array_equal(first.reward, second.reward)
    np.testing.assert_array_equal(
        first.sample(3)[0].cpu().numpy(),
        second.sample(3)[0].cpu().numpy(),
    )


def test_tiny_solver_records_fixed_beta_fast_slow_and_rng_invariants():
    solver = _tiny_solver()
    rows = solver.solve()
    final = rows[-1]
    assert final["control_variate_beta_min"] == pytest.approx(1.0)
    assert final["control_variate_beta_max"] == pytest.approx(1.0)
    assert final["control_replay_rng_isolated"] == pytest.approx(1.0)
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12
    assert 0.0 <= final["fast_slow_rho_min"]
    assert final["fast_slow_rho_max"] <= 1.0
    assert final["rho_controller_replay_size"] > 0
    assert final["slow_critic_lifetime_seen_count"] > 0
    assert sum(
        final[f"fast_q_fold_{fold}_replay_size"] for fold in range(3)
    ) > 0
    assert sum(final[f"q_fold_{fold}_replay_size"] for fold in range(3)) > 0


def test_reference_results_are_checksum_validated_and_node_matched():
    curves, summaries = experiment_run._load_references((0, 1, 2))
    assert len(curves) == 274
    assert len(summaries) == 9
    assert {row["algorithm_id"] for row in curves} == {
        experiment_config.EXPERIMENT_6_ALGORITHM_ID,
        experiment_config.EXPERIMENT_9_ALGORITHM_ID,
        experiment_config.EXPERIMENT_13_ALGORITHM_ID,
    }
    assert experiment_config.EXPERIMENT_2_NODE_TARGETS == {
        0: 4_700_205,
        1: 4_701_540,
        2: 4_684_695,
    }
    assert experiment_config.CANDIDATE_CONFIG[
        "fixed_control_variate_beta"
    ] == 1.0
    assert experiment_config.CANDIDATE_CONFIG["fast_q_train_steps"] == 5_000
    assert experiment_config.CANDIDATE_CONFIG["rho_train_steps"] == 2_000


def test_smoke_overrides_preserve_architecture_defaults():
    args = experiment_run._parser().parse_args(
        [
            "--traversals",
            "4",
            "--max-iterations",
            "2",
            "--q-train-steps",
            "1",
            "--fast-q-train-steps",
            "1",
            "--calibration-train-steps",
            "1",
            "--rho-train-steps",
            "1",
            "--batch-size",
            "2",
            "--buffer-size",
            "128",
            "--fast-q-buffer-size",
            "128",
            "--rho-buffer-size",
            "128",
            "--early-evaluation-nodes",
            "10",
        ]
    )
    config = deepcopy(experiment_config.CANDIDATE_CONFIG)
    experiment_run._apply_overrides(args, config)
    assert config["num_traversals"] == 4
    assert config["max_num_iterations"] == 2
    assert config["baseline_network_train_steps"] == 1
    assert config["fast_q_train_steps"] == 1
    assert config["rho_train_steps"] == 1
    assert config["baseline_buffer_size"] == 128
    assert config["fast_q_buffer_size"] == 128
    assert config["early_evaluation_node_thresholds"] == (10,)
    assert config["fixed_control_variate_beta"] == 1.0


def test_runtime_and_batch_timeout_have_headroom():
    assert experiment_config.EXPECTED_SEQUENTIAL_RUNTIME_HOURS == 17
    assert experiment_config.BATCH_TIMEOUT_SECONDS == 36 * 60 * 60
    assert experiment_config.BATCH_TIMEOUT_SECONDS > (
        experiment_config.EXPECTED_SEQUENTIAL_RUNTIME_HOURS * 3600
    )


def test_readmes_document_full_then_smoke_batch_jobs():
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    experiment_readme = (
        ROOT
        / "experiments"
        / "leduc_poker"
        / "fixed_beta_fast_slow_escher_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    module = "fixed_beta_fast_slow_escher_5x_nodes.run"
    smoke_job = "leduc-escher-arch-exp15-fixed-beta-fast-slow-smoke"
    for readme in (root_readme, experiment_readme):
        assert module in readme
        assert smoke_job in readme
        assert "--seeds 0" in readme
        assert "--fast-q-train-steps 1" in readme
        assert "--rho-train-steps 1" in readme
        assert "n2-standard-8 129600 8000 32000 100" in readme
        assert readme.index("Experiment 15 full single GCP Batch job") < (
            readme.index("Experiment 15 GCP Batch smoke test")
        )
    assert experiment_config.EXPERIMENT_9_CURVES_SHA256 in experiment_readme
    assert experiment_config.EXPERIMENT_13_CURVES_SHA256 in experiment_readme
