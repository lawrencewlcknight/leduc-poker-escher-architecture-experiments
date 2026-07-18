"""Experiment 4 horizon and saved-reference correctness tests."""

from collections import Counter
import hashlib
from pathlib import Path

from experiments.leduc_poker.adaptive_residual_predictive_escher.config import (
    ADAPTIVE_CONFIG as EXPERIMENT_3_ADAPTIVE_CONFIG,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.config import (
    ADAPTIVE_CONFIG,
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_2_SOURCE,
    REFERENCE_CURVE_ROWS,
    REFERENCE_CURVES,
    REFERENCE_CURVES_SHA256,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run import (
    _load_reference_curves,
    _reference_summaries,
)


def test_experiment_4_is_a_horizon_only_extension_of_experiment_3():
    assert ADAPTIVE_CONFIG == EXPERIMENT_3_ADAPTIVE_CONFIG
    assert ADAPTIVE_CONFIG is not EXPERIMENT_3_ADAPTIVE_CONFIG
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert EXPERIMENT_2_NODE_TARGETS == {
        0: 4_700_205,
        1: 4_701_540,
        2: 4_684_695,
    }
    assert ADAPTIVE_CONFIG["max_num_iterations"] == 100
    assert BATCH_TIMEOUT_SECONDS == 64_800


def test_bundled_reference_is_the_exact_experiment_2_curve_file():
    assert hashlib.sha256(REFERENCE_CURVES.read_bytes()).hexdigest() == (
        REFERENCE_CURVES_SHA256
    )
    rows = _load_reference_curves(REFERENCE_CURVES)
    assert len(rows) == REFERENCE_CURVE_ROWS == 323
    assert {row["algorithm_id"] for row in rows} == {
        "escher_exp28",
        "vr_deep_dcfr_plus",
        "vr_deep_pdcfr_plus",
    }
    assert {row["result_source"] for row in rows} == {"saved_experiment_2"}
    assert Counter((row["algorithm_id"], row["seed"]) for row in rows) == {
        ("escher_exp28", 0): 43,
        ("escher_exp28", 1): 43,
        ("escher_exp28", 2): 43,
        ("vr_deep_dcfr_plus", 0): 32,
        ("vr_deep_dcfr_plus", 1): 32,
        ("vr_deep_dcfr_plus", 2): 32,
        ("vr_deep_pdcfr_plus", 0): 33,
        ("vr_deep_pdcfr_plus", 1): 33,
        ("vr_deep_pdcfr_plus", 2): 32,
    }

    summaries = _reference_summaries(rows)
    assert len(summaries) == 9
    escher = {
        row["seed"]: int(row["final_nodes_touched"])
        for row in summaries
        if row["algorithm_id"] == "escher_exp28"
    }
    assert escher == EXPERIMENT_2_NODE_TARGETS


def test_experiment_2_provenance_and_experiment_4_commands_are_documented():
    assert EXPERIMENT_2_SOURCE["batch_job"].endswith(
        "leduc-escher-arch-exp2-20260717-105458"
    )
    experiment_readme = REFERENCE_CURVES.with_name("README.md").read_text(
        encoding="utf-8"
    )
    root_readme = (Path(__file__).parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    for readme in (experiment_readme, root_readme):
        assert (
            "experiments.leduc_poker."
            "adaptive_residual_predictive_escher_5x_nodes.run"
        ) in readme
        assert "leduc-escher-arch-exp4-adaptive-5x-smoke" in readme
        assert "--early-evaluation-nodes 10" in readme
    assert REFERENCE_CURVES_SHA256 in experiment_readme
    assert "n2-standard-8 64800 8000 32000 100" in experiment_readme

