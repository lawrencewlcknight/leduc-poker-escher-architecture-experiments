"""Mechanism and orchestration tests for Experiment 8."""

from pathlib import Path

import pytest

from experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.config import (
    DEFAULT_VARIANT_IDS,
    EXPERIMENT_2_NODE_TARGETS,
    FIXED_BETA_ONE,
    FIXED_BETA_ONE_NO_PREDICTOR,
    FULL_EXPERIMENT_6,
    LEAN_CANDIDATE,
    PREDICTION_GATE_ZERO,
    SINGLE_FROZEN_TARGET_CRITIC,
    TWO_CROSS_FITTED_CRITICS,
    UNIFORM_FULL_SUPPORT_SAMPLING,
    VARIANTS,
)
from experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run import (
    _variant_config,
)
from experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.config import (
    BASE_CONFIG,
)
from unbiased_escher import UnbiasedControlVariateEscher
from vr_deep_cfr.logger import Logger


def _tiny_solver(**architecture):
    solver = UnbiasedControlVariateEscher(
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
        calibration_buffer_size=128,
        calibration_batch_size=2,
        calibration_train_steps=1,
        **architecture,
    )
    solver.max_num_iterations = 1
    solver.evaluate_initial_policy = False
    solver.early_evaluation_node_thresholds = ()
    return solver


def test_requested_arms_and_combined_lean_candidate_are_declared():
    assert DEFAULT_VARIANT_IDS == (
        FULL_EXPERIMENT_6,
        FIXED_BETA_ONE,
        PREDICTION_GATE_ZERO,
        FIXED_BETA_ONE_NO_PREDICTOR,
        TWO_CROSS_FITTED_CRITICS,
        SINGLE_FROZEN_TARGET_CRITIC,
        UNIFORM_FULL_SUPPORT_SAMPLING,
        LEAN_CANDIDATE,
    )
    assert EXPERIMENT_2_NODE_TARGETS == {
        0: 4_700_205,
        1: 4_701_540,
        2: 4_684_695,
    }
    assert VARIANTS[FULL_EXPERIMENT_6]["overrides"] == {}
    assert VARIANTS[TWO_CROSS_FITTED_CRITICS]["overrides"] == {
        "q_ensemble_size": 2
    }
    assert VARIANTS[SINGLE_FROZEN_TARGET_CRITIC]["overrides"] == {
        "q_ensemble_size": 1
    }


def test_variant_configuration_does_not_mutate_full_control():
    lean = _variant_config(LEAN_CANDIDATE, BASE_CONFIG)
    full = _variant_config(FULL_EXPERIMENT_6, BASE_CONFIG)
    assert lean["fixed_control_variate_beta"] == 1.0
    assert lean["q_ensemble_size"] == 2
    assert not lean["use_instantaneous_predictor"]
    assert not lean["use_residual_calibration"]
    assert lean["sampling_uniform_floor_mass"] == 1.0
    assert full["fixed_control_variate_beta"] is None
    assert full["q_ensemble_size"] == 3
    assert full["use_instantaneous_predictor"]
    assert full["use_residual_calibration"]
    assert full["sampling_uniform_floor_mass"] == 0.2


def test_lean_candidate_really_removes_predictor_and_calibration():
    solver = _tiny_solver(
        fixed_control_variate_beta=1.0,
        use_instantaneous_predictor=False,
        q_ensemble_size=2,
        sampling_uniform_floor_mass=1.0,
        use_residual_calibration=False,
    )
    assert solver.calibration_trainer is None
    assert len(solver.q_value_trainer.members) == 2
    assert all(
        not hasattr(trainer, "imm_model") for trainer in solver.regret_trainers
    )
    rows = solver.solve()
    final = rows[-1]
    assert final["control_variate_beta_min"] == pytest.approx(1.0)
    assert final["control_variate_beta_max"] == pytest.approx(1.0)
    assert final["prediction_gate_player_0"] == 0.0
    assert final["prediction_gate_player_1"] == 0.0
    assert final["calibration_target_version"] == 0
    assert final["full_support_sampling_min_probability"] >= 1.0 / 3.0
    assert final["policy_weighted_advantage_abs_mean"] < 1e-12


def test_single_critic_arm_is_supported_but_not_claimed_as_cross_fitted():
    solver = _tiny_solver(q_ensemble_size=1)
    ensemble = solver.q_value_trainer
    state = solver.skip_chance_state(solver.game.new_initial_state())
    assert ensemble.begin_trajectory(7) == 0
    assert ensemble.heldout_member_indices() == [0]
    values, disagreement = ensemble.get_baseline_and_disagreement(state, 0)
    assert values.shape == (solver.action_size,)
    assert disagreement.shape == (solver.action_size,)
    assert max(abs(value) for value in disagreement) == pytest.approx(0.0)


def test_disabling_calibration_requires_a_complete_lean_sampling_contract():
    with pytest.raises(ValueError, match="fixed control-variate beta"):
        _tiny_solver(use_residual_calibration=False)
    with pytest.raises(ValueError, match="uniform full-support sampling"):
        _tiny_solver(
            use_residual_calibration=False,
            fixed_control_variate_beta=1.0,
        )


def test_experiment_8_smoke_test_is_documented_in_both_readmes():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "unbiased_control_variate_escher_lean_ablation"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "unbiased_control_variate_escher_lean_ablation.run" in readme
        assert "leduc-escher-arch-exp8-lean-smoke" in readme
        assert "--calibration-train-steps 1" in readme
        assert "--target-nodes 50" in readme
    assert "n2-standard-8 345600 8000 32000 100" in experiment_readme

