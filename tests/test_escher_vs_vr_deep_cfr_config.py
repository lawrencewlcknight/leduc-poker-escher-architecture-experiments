"""Contract tests for the matched-node algorithm comparison."""

from pathlib import Path

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXP28_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.config import (
    ALGORITHMS,
    DEFAULT_SEEDS,
    ESCHER_CONFIG,
    UPSTREAM,
    VR_PAPER_CONFIG,
)


def test_comparison_uses_three_paired_seeds():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert set(ALGORITHMS) == {
        "escher_exp28",
        "vr_deep_dcfr_plus",
        "vr_deep_pdcfr_plus",
    }


def test_escher_arm_preserves_experiment_28_training_contract():
    excluded = {"experiment_name", "variant_id", "variant_label", "variant_description"}
    for key, value in EXP28_DEFAULT_CONFIG.items():
        if key not in excluded:
            assert ESCHER_CONFIG[key] == value


def test_vr_defaults_use_paper_settings_and_dense_evaluation():
    assert VR_PAPER_CONFIG["num_traversals"] == 10_000
    assert VR_PAPER_CONFIG["advantage_buffer_size"] == 1_000_000
    assert VR_PAPER_CONFIG["ave_policy_buffer_size"] == 1_000_000
    assert VR_PAPER_CONFIG["baseline_buffer_size"] == 1_000_000
    assert VR_PAPER_CONFIG["advantage_network_train_steps"] == 750
    assert VR_PAPER_CONFIG["ave_policy_network_train_steps"] == 5_000
    assert VR_PAPER_CONFIG["baseline_network_train_steps"] == 10_000
    assert VR_PAPER_CONFIG["evaluation_frequency"] == 1
    assert VR_PAPER_CONFIG["preserve_evaluation_rng"] is True


def test_vr_variant_parameters_match_reported_algorithms():
    assert ALGORITHMS["vr_deep_dcfr_plus"]["alpha"] == 2.0
    assert ALGORITHMS["vr_deep_dcfr_plus"]["gamma"] == 2.0
    assert ALGORITHMS["vr_deep_pdcfr_plus"]["alpha"] == 2.3
    assert ALGORITHMS["vr_deep_pdcfr_plus"]["gamma"] == 2.0
    assert ALGORITHMS["vr_deep_pdcfr_plus"]["reinitialize_imm_regret_networks"] is True


def test_integrated_source_records_exact_upstream_commit():
    assert UPSTREAM["commit"] == "9f156c9fcdac7f8c9bd0debf94c9432d222858d3"
    upstream_notice = Path(__file__).parents[1] / "vr_deep_cfr" / "UPSTREAM.md"
    assert UPSTREAM["commit"] in upstream_notice.read_text(encoding="utf-8")
