"""Contract tests for Experiment 7's 15-million-node comparison."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.config import (
    VR_PAPER_CONFIG,
)
from experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.config import (
    UNBIASED_CONFIG,
)
from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    CANDIDATE_ALGORITHM_ID,
    CANDIDATE_CONFIG,
    DEFAULT_ALGORITHM_IDS,
    DEFAULT_SEEDS,
    EXPECTED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS,
    EXPECTED_SEQUENTIAL_RUNTIME_HOURS,
    MAX_NUM_ITERATIONS,
    PARALLEL_BATCH_TIMEOUT_SECONDS,
    SEQUENTIAL_BATCH_TIMEOUT_SECONDS,
    TARGET_NODES,
    VR_CONFIG,
)
from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run import (
    _apply_overrides,
    _load_aggregate_results,
    _parse_algorithms,
    _parser,
)


def test_experiment_7_uses_three_seeds_three_algorithms_and_15m_nodes():
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert DEFAULT_ALGORITHM_IDS == (
        "vr_deep_dcfr_plus",
        "vr_deep_pdcfr_plus",
        CANDIDATE_ALGORITHM_ID,
    )
    assert TARGET_NODES == 15_000_000


def test_production_configs_only_raise_the_iteration_safety_cap():
    assert MAX_NUM_ITERATIONS == 120
    for key, value in VR_PAPER_CONFIG.items():
        if key != "max_num_iterations":
            assert VR_CONFIG[key] == value
    for key, value in UNBIASED_CONFIG.items():
        if key != "max_num_iterations":
            assert CANDIDATE_CONFIG[key] == value
    assert VR_CONFIG["max_num_iterations"] == 120
    assert CANDIDATE_CONFIG["max_num_iterations"] == 120


def test_runtime_estimates_have_conservative_timeout_headroom():
    assert EXPECTED_SEQUENTIAL_RUNTIME_HOURS == 78
    assert SEQUENTIAL_BATCH_TIMEOUT_SECONDS == 96 * 60 * 60
    assert EXPECTED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS == 42
    assert PARALLEL_BATCH_TIMEOUT_SECONDS == 48 * 60 * 60
    assert SEQUENTIAL_BATCH_TIMEOUT_SECONDS > EXPECTED_SEQUENTIAL_RUNTIME_HOURS * 3600
    assert PARALLEL_BATCH_TIMEOUT_SECONDS > (
        EXPECTED_PARALLEL_BY_ALGORITHM_RUNTIME_HOURS * 3600
    )


def test_algorithm_subset_validation():
    assert _parse_algorithms(None) == list(DEFAULT_ALGORITHM_IDS)
    assert _parse_algorithms("vr_deep_dcfr_plus") == ["vr_deep_dcfr_plus"]
    with pytest.raises(ValueError, match="Unknown algorithm ids"):
        _parse_algorithms("not_an_algorithm")


def test_smoke_overrides_apply_to_both_families_without_mutating_defaults():
    args = _parser().parse_args(
        [
            "--traversals",
            "4",
            "--max-iterations",
            "2",
            "--advantage-train-steps",
            "1",
            "--policy-train-steps",
            "1",
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
    vr = deepcopy(VR_CONFIG)
    candidate = deepcopy(CANDIDATE_CONFIG)
    _apply_overrides(args, vr, candidate)

    for config in (vr, candidate):
        assert config["num_traversals"] == 4
        assert config["max_num_iterations"] == 2
        assert config["advantage_network_train_steps"] == 1
        assert config["ave_policy_network_train_steps"] == 1
        assert config["baseline_network_train_steps"] == 1
        assert config["advantage_batch_size"] == 2
        assert config["ave_policy_batch_size"] == 2
        assert config["baseline_batch_size"] == 2
        assert config["advantage_buffer_size"] == 128
        assert config["early_evaluation_node_thresholds"] == (10,)
    assert candidate["calibration_train_steps"] == 1
    assert candidate["calibration_batch_size"] == 2
    assert candidate["calibration_buffer_size"] == 128
    assert VR_CONFIG["num_traversals"] == 10_000
    assert CANDIDATE_CONFIG["calibration_train_steps"] == 2_000


def _write_worker_result(path: Path, algorithm_id: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary": {"algorithm_id": algorithm_id, "seed": seed},
                "curves": [],
            }
        ),
        encoding="utf-8",
    )


def test_partial_job_aggregator_finds_results_and_rejects_duplicates(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_worker_result(
        first / "run" / "worker_results" / "dcfr_seed_0.json",
        "vr_deep_dcfr_plus",
        0,
    )
    _write_worker_result(
        second / "run" / "worker_results" / "candidate_seed_0.json",
        CANDIDATE_ALGORITHM_ID,
        0,
    )
    results = _load_aggregate_results([first, second])
    assert {
        (result["summary"]["algorithm_id"], result["summary"]["seed"])
        for result in results
    } == {
        ("vr_deep_dcfr_plus", 0),
        (CANDIDATE_ALGORITHM_ID, 0),
    }

    duplicate = tmp_path / "duplicate"
    _write_worker_result(
        duplicate / "run" / "worker_results" / "dcfr_seed_0.json",
        "vr_deep_dcfr_plus",
        0,
    )
    with pytest.raises(ValueError, match="Duplicate aggregate result"):
        _load_aggregate_results([first, duplicate])


def test_root_and_experiment_readmes_document_smoke_and_parallel_full_runs():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "unbiased_escher_vs_vr_deep_cfr_15m_nodes"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "unbiased_escher_vs_vr_deep_cfr_15m_nodes.run" in readme
        assert "leduc-escher-arch-exp7-15m-smoke" in readme
        assert "--calibration-train-steps 1" in readme
    assert "n2-standard-8 345600 8000 32000 100" in experiment_readme
    assert "--algorithms vr_deep_dcfr_plus" in experiment_readme
    assert "--aggregate-run-dir" in experiment_readme

