"""Contract tests for the five-times-longer matched-node experiment."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXP28_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.config import (
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_SEEDS,
    ESCHER_CONFIG,
    ESCHER_NUM_ITERATIONS,
    EXPECTED_BATCH_RUNTIME_HOURS,
    NODE_BUDGET_MULTIPLIER,
    VR_PAPER_CONFIG,
)
from experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.config import (
    VR_PAPER_CONFIG as EXPERIMENT_1_VR_CONFIG,
)
from vr_deep_cfr import VRDeepDCFRPlus
from vr_deep_cfr.logger import Logger
from vr_deep_cfr.solver import DeepCumuAdv


def test_five_times_budget_uses_three_paired_seeds_and_exact_cycle_ratio():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert NODE_BUDGET_MULTIPLIER == 5
    assert ESCHER_NUM_ITERATIONS == 404
    assert ESCHER_CONFIG["num_iterations"] + 1 == 5 * (
        EXP28_DEFAULT_CONFIG["num_iterations"] + 1
    )


def test_escher_preserves_exp28_settings_except_horizon_and_observation():
    permitted_changes = {
        "experiment_name",
        "variant_id",
        "variant_label",
        "variant_description",
        "num_iterations",
        "evaluate_initial_policy",
        "intermediate_policy_training_events_expected",
        "final_policy_training_events_expected",
        "total_policy_training_events_expected",
        "policy_gradient_steps_expected",
    }
    for key, value in EXP28_DEFAULT_CONFIG.items():
        if key not in permitted_changes:
            assert ESCHER_CONFIG[key] == value
    assert ESCHER_CONFIG["evaluate_initial_policy"] is True


def test_vr_keeps_paper_training_settings_and_adds_only_early_observations():
    additions = {"evaluate_initial_policy", "early_evaluation_node_thresholds"}
    for key, value in EXPERIMENT_1_VR_CONFIG.items():
        assert VR_PAPER_CONFIG[key] == value
    assert set(VR_PAPER_CONFIG) == set(EXPERIMENT_1_VR_CONFIG) | additions
    assert VR_PAPER_CONFIG["evaluate_initial_policy"] is True
    assert VR_PAPER_CONFIG["early_evaluation_node_thresholds"] == (10_000,)


def test_batch_timeout_has_twelve_hours_headroom_over_projection():
    assert EXPECTED_BATCH_RUNTIME_HOURS == 24
    assert BATCH_TIMEOUT_SECONDS == 36 * 60 * 60
    assert BATCH_TIMEOUT_SECONDS >= (EXPECTED_BATCH_RUNTIME_HOURS + 12) * 60 * 60


def test_smoke_cli_can_lower_only_the_early_evaluation_threshold():
    from copy import deepcopy

    from experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run import (
        _apply_overrides,
        _parser,
    )

    args = _parser().parse_args(["--vr-early-evaluation-nodes", "10"])
    escher = deepcopy(ESCHER_CONFIG)
    vr = deepcopy(VR_PAPER_CONFIG)
    _apply_overrides(args, escher, vr)

    assert vr["early_evaluation_node_thresholds"] == (10,)
    assert escher == ESCHER_CONFIG


def test_vr_threshold_scheduler_fires_once_just_after_10k_nodes():
    solver = DeepCumuAdv.__new__(DeepCumuAdv)
    solver.early_evaluation_node_thresholds = (10_000,)
    solver._next_early_evaluation_index = 0
    solver.ave_policy_trainer = SimpleNamespace(buffer=[object()])
    calls = []
    solver._run_checkpoint = lambda **kwargs: calls.append(kwargs)
    solver._prepare_early_evaluation_schedule()

    solver.nodes_touched = 9_999
    solver._maybe_run_early_node_checkpoint()
    assert calls == []

    solver.nodes_touched = 10_006
    solver._maybe_run_early_node_checkpoint()
    solver._maybe_run_early_node_checkpoint()
    assert calls == [
        {
            "checkpoint_kind": "early_node_threshold",
            "checkpoint_target_nodes": 10_000,
        }
    ]


def test_vr_thresholds_must_be_positive():
    solver = DeepCumuAdv.__new__(DeepCumuAdv)
    solver.early_evaluation_node_thresholds = (0,)
    with pytest.raises(ValueError, match="must be positive"):
        solver._prepare_early_evaluation_schedule()


def _run_tiny_vr_solver(*, early_thresholds):
    solver = VRDeepDCFRPlus(
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
        use_baseline=True,
        device="cpu",
        seed=0,
        logger=Logger(verbose=False),
    )
    solver.evaluate_initial_policy = True
    solver.early_evaluation_node_thresholds = early_thresholds
    return solver.solve()


def test_early_vr_evaluation_preserves_the_final_training_result():
    without_early = _run_tiny_vr_solver(early_thresholds=())
    with_early = _run_tiny_vr_solver(early_thresholds=(10,))

    assert [row["checkpoint_kind"] for row in with_early] == [
        "initial_untrained_policy",
        "early_node_threshold",
        "outer_iteration",
    ]
    assert with_early[0]["nodes_touched"] == 0
    assert 10 <= with_early[1]["nodes_touched"] < 20
    for key in ("nodes_touched", "exp", "average_policy_value", "average_policy_loss"):
        assert with_early[-1][key] == without_early[-1][key]


def test_readme_records_full_36_hour_batch_command():
    readme = (
        Path(__file__).parents[1]
        / "experiments"
        / "leduc_poker"
        / "escher_vs_vr_deep_cfr_5x_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    assert "escher_vs_vr_deep_cfr_5x_nodes.run" in readme
    assert "n2-standard-8 129600 8000 32000 100" in readme
    assert "leduc-escher-arch-exp2-5x-smoke" in readme
    assert "--vr-early-evaluation-nodes 10" in readme
    assert "n2-standard-4 21600 4000 16000 100" in readme
