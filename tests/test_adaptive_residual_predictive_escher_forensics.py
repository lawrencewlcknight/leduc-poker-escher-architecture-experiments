"""Correctness tests for Experiment 5's exact forensic diagnostics."""

from pathlib import Path

import numpy as np

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics import (
    config as forensic_config,
    diagnostics as forensic_diagnostics,
    run as forensic_run,
    solver as forensic_solver,
)
from vr_deep_cfr.logger import Logger


BATCH_TIMEOUT_SECONDS = forensic_config.BATCH_TIMEOUT_SECONDS
CONTROL_VARIANT = forensic_config.CONTROL_VARIANT
DEFAULT_SEEDS = forensic_config.DEFAULT_SEEDS
EXPERIMENT_1_NODE_TARGETS = forensic_config.EXPERIMENT_1_NODE_TARGETS
VARIANTS = forensic_config.VARIANTS
ExactLeducOracle = forensic_diagnostics.ExactLeducOracle
build_policy_table = forensic_diagnostics.build_policy_table
ForensicAdaptiveResidualPredictiveEscher = (
    forensic_solver.ForensicAdaptiveResidualPredictiveEscher
)


def _tiny_solver(
    *,
    variant_id="lambda_one",
    lambda_mode="fixed_one",
    use_predictive_accumulator=True,
    q_mode="persistent",
):
    traversals = 4
    solver = ForensicAdaptiveResidualPredictiveEscher(
        game_name="leduc_poker",
        num_episodes=2 * traversals,
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
        diagnostic_variant_id=variant_id,
        lambda_mode=lambda_mode,
        use_predictive_accumulator=use_predictive_accumulator,
        q_mode=q_mode,
    )
    solver.evaluate_initial_policy = True
    solver.early_evaluation_node_thresholds = (10,)
    return solver


def test_six_arms_are_one_factor_changes_from_control():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert EXPERIMENT_1_NODE_TARGETS == {0: 942_635, 1: 939_834, 2: 962_274}
    assert BATCH_TIMEOUT_SECONDS == 86_400
    assert len(VARIANTS) == 6

    mechanism_fields = {
        "lambda_mode",
        "use_predictive_accumulator",
        "q_mode",
    }
    control = VARIANTS[CONTROL_VARIANT]
    for variant_id, variant in VARIANTS.items():
        differences = {
            field
            for field in mechanism_fields
            if variant[field] != control[field]
        }
        if variant_id == CONTROL_VARIANT:
            assert differences == set()
        else:
            assert len(differences) == 1


def test_exact_estimator_moments_identify_bias_variance_tradeoff():
    solver = _tiny_solver()
    policy = build_policy_table(solver.game, solver.predictive_strategy)

    lambda_one_rows = ExactLeducOracle(solver, policy).estimator_rows()
    assert lambda_one_rows
    assert max(abs(row["estimator_bias"]) for row in lambda_one_rows) < 1e-12
    assert any(row["estimator_variance"] > 0.0 for row in lambda_one_rows)

    solver.lambda_controller.mode = "fixed_zero"
    lambda_zero_rows = ExactLeducOracle(solver, policy).estimator_rows()
    assert any(abs(row["estimator_bias"]) > 1e-5 for row in lambda_zero_rows)
    assert np.mean([row["estimator_variance"] for row in lambda_zero_rows]) < (
        np.mean([row["estimator_variance"] for row in lambda_one_rows])
    )


def test_one_iteration_exact_average_equals_the_observed_current_strategy():
    solver = _tiny_solver()
    expected = build_policy_table(solver.game, solver.predictive_strategy)
    solver.exact_average_strategy.observe_iteration(1, solver.predictive_strategy)
    actual = solver.exact_average_strategy.table()

    # Zero-own-reach information sets have zero denominator and intentionally
    # use the exact tabular policy's uniform fallback.
    assert actual.keys() <= expected.keys()
    assert actual
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)


def test_nonpredictive_arm_exposes_the_cumulative_strategy_consistently():
    solver = _tiny_solver(
        variant_id="nonpredictive_accumulator",
        lambda_mode="scheduled",
        use_predictive_accumulator=False,
    )
    state = solver.game.new_initial_state()
    while state.is_chance_node():
        state = state.child(state.chance_outcomes()[0][0])
    np.testing.assert_allclose(
        solver.predictive_strategy(state),
        solver.nonpredictive_strategy(state),
        atol=1e-12,
    )


def test_predictor_diagnostic_compares_matched_architectural_arms():
    shared = {
        "seed": 2,
        "checkpoint_index": 3,
        "nodes_touched": 100.0,
    }
    rows = forensic_run._predictor_ablation_rows(
        [
            {
                **shared,
                "variant_id": CONTROL_VARIANT,
                "predictor_preupdate_mse": 0.25,
                "predictor_postupdate_mse": 0.10,
                "current_predictive_exploitability": 0.7,
            },
            {
                **shared,
                "variant_id": "nonpredictive_accumulator",
                "current_nonpredictive_exploitability": 1.1,
            },
        ]
    )

    assert len(rows) == 1
    assert rows[0]["predictor_preupdate_mse"] == 0.25
    assert rows[0]["predictor_postupdate_mse"] == 0.10
    assert np.isclose(
        rows[0]["predictive_update_exploitability_improvement"], 0.4
    )


def test_readmes_document_all_six_branches_and_smoke_tests():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "adaptive_residual_predictive_escher_forensics"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")

    for variant_id in VARIANTS:
        assert variant_id in experiment_readme
    for readme in (experiment_readme, root_readme):
        assert "adaptive_residual_predictive_escher_forensics.run" in readme
        assert "leduc-escher-arch-exp5-forensics-smoke" in readme
        assert "--early-evaluation-nodes 10" in readme
    assert "n2-standard-8 86400 8000 32000 100" in experiment_readme
