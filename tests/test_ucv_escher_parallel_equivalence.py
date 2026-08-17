"""Contracts for Experiment 18 and the UCV-ESCHER parallel backend."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (
    CANDIDATE_CONFIG,
)
from experiments.leduc_poker.ucv_escher_parallel_equivalence.config import (
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_CONFIG,
    DEFAULT_SEEDS,
    FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN,
    FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN,
    PARALLEL_NUM_WORKERS,
    PARALLEL_VARIANT_ID,
    RECOMMENDED_BATCH_TIMEOUT_MINUTES,
    SEQUENTIAL_VARIANT_ID,
    TARGET_NODES,
    VARIANTS,
)
from experiments.leduc_poker.ucv_escher_parallel_equivalence.run import (
    _apply_overrides,
    _paired_rows,
    _parse_variant_ids,
    _parser,
)
from unbiased_escher.parallel_solver import _append_circular
from unbiased_escher.parallel_utils import (
    equivalence_summary,
    partition_total,
    worker_seed,
)


def test_experiment_18_reuses_exact_experiment_7_ucv_config():
    assert DEFAULT_CONFIG == CANDIDATE_CONFIG
    assert DEFAULT_CONFIG is not CANDIDATE_CONFIG
    assert DEFAULT_SEEDS == [0, 1, 2]
    assert TARGET_NODES == 15_000_000
    assert [variant["variant_id"] for variant in VARIANTS] == [
        SEQUENTIAL_VARIANT_ID,
        PARALLEL_VARIANT_ID,
    ]
    assert VARIANTS[0]["execution_backend"] == "sequential"
    assert VARIANTS[1]["execution_backend"] == "ray_parallel"
    assert PARALLEL_NUM_WORKERS == 3


def test_equivalence_margins_are_predeclared_and_tighter_than_legacy_checks():
    assert FINAL_EXPLOITABILITY_EQUIVALENCE_MARGIN == 0.02
    assert FINAL_POLICY_VALUE_EQUIVALENCE_MARGIN == 0.01
    assert RECOMMENDED_BATCH_TIMEOUT_MINUTES == 84 * 60
    assert BATCH_TIMEOUT_SECONDS == RECOMMENDED_BATCH_TIMEOUT_MINUTES * 60


def test_parallel_budget_partition_and_worker_seeds_are_exact():
    assert partition_total(10_000, 3) == [3334, 3333, 3333]
    assert sum(partition_total(10_000, 3)) == 10_000
    assert len({worker_seed(7, index) for index in range(3)}) == 3
    with pytest.raises(ValueError, match="non-negative"):
        worker_seed(0, -1)


def test_smoke_overrides_do_not_mutate_production_config():
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
            "--evaluation-frequency",
            "1",
            "--early-evaluation-nodes",
            "10",
        ]
    )
    config = deepcopy(DEFAULT_CONFIG)
    _apply_overrides(args, config)
    assert config["num_traversals"] == 4
    assert config["max_num_iterations"] == 2
    assert config["calibration_train_steps"] == 1
    assert config["calibration_buffer_size"] == 128
    assert config["early_evaluation_node_thresholds"] == (10,)
    assert DEFAULT_CONFIG["num_traversals"] == 10_000
    assert DEFAULT_CONFIG["max_num_iterations"] == 120


def test_variant_parser_and_paired_delta_direction():
    assert _parse_variant_ids(None) == [
        SEQUENTIAL_VARIANT_ID,
        PARALLEL_VARIANT_ID,
    ]
    with pytest.raises(ValueError, match="Unknown variant"):
        _parse_variant_ids("missing")
    rows = _paired_rows(
        [
            {
                "variant_id": SEQUENTIAL_VARIANT_ID,
                "seed": 0,
                "final_exploitability": 0.08,
                "final_policy_value": -0.09,
                "final_nodes_touched": 100,
                "solver_initialization_seconds": 2,
                "training_seconds": 20,
                "end_to_end_seconds": 22,
                "final_cumulative_experience_collection_seconds": 8,
            },
            {
                "variant_id": PARALLEL_VARIANT_ID,
                "seed": 0,
                "final_exploitability": 0.09,
                "final_policy_value": -0.085,
                "final_nodes_touched": 102,
                "solver_initialization_seconds": 4,
                "training_seconds": 10,
                "end_to_end_seconds": 14,
                "final_cumulative_experience_collection_seconds": 4,
            },
        ]
    )
    assert rows[0]["exploitability_delta_parallel_minus_sequential"] == pytest.approx(0.01)
    assert rows[0]["policy_value_delta_parallel_minus_sequential"] == pytest.approx(0.005)
    assert rows[0]["training_seconds_speedup_sequential_over_parallel"] == 2.0


def test_equivalence_summary_uses_paired_90_percent_interval():
    result = equivalence_summary([0.001, 0.002, 0.0015], margin=0.02)
    assert result["n"] == 3
    assert result["all_seed_deltas_within_margin"] is True
    assert result["tost_equivalent"] is True
    assert result["ci_lower"] < result["mean_delta"] < result["ci_upper"]


class _TinyCircular:
    def __init__(self, capacity=5):
        self.buffer_size = capacity
        self.cur_id = 0
        self.size = 0
        self.history_buf = np.zeros((capacity, 2))
        self.action_buf = np.zeros(capacity, dtype=int)
        self.next_history_buf = np.zeros((capacity, 2))
        self.next_state_buf = np.zeros((capacity, 3))
        self.next_legal_actions_mask_buf = np.zeros((capacity, 2), dtype=int)
        self.next_player_buf = np.zeros(capacity, dtype=int)
        self.done_buf = np.zeros(capacity, dtype=int)
        self.reward_buf = np.zeros(capacity)


def _transition_payload(actions):
    count = len(actions)
    return {
        "histories": np.arange(count * 2).reshape(count, 2),
        "actions": np.asarray(actions),
        "next_histories": np.arange(count * 2).reshape(count, 2) + 10,
        "next_states": np.arange(count * 3).reshape(count, 3),
        "next_legal_masks": np.ones((count, 2), dtype=int),
        "next_players": np.zeros(count, dtype=int),
        "dones": np.zeros(count, dtype=int),
        "rewards": np.arange(count, dtype=float),
    }


def test_vectorised_circular_merge_preserves_capacity_and_wraps():
    buffer = _TinyCircular(capacity=5)
    _append_circular(buffer, _transition_payload([0, 1, 2]))
    _append_circular(buffer, _transition_payload([3, 4, 5, 6]))
    assert buffer.size == 5
    assert buffer.cur_id == 2
    assert sorted(buffer.action_buf.tolist()) == [2, 3, 4, 5, 6]


def test_root_and_experiment_readmes_include_gcp_smoke_test():
    root = Path(__file__).parents[1]
    readmes = [
        (root / "README.md").read_text(encoding="utf-8"),
        (
            root
            / "experiments"
            / "leduc_poker"
            / "ucv_escher_parallel_equivalence"
            / "README.md"
        ).read_text(encoding="utf-8"),
    ]
    for readme in readmes:
        assert "ucv_escher_parallel_equivalence.run" in readme
        assert "leduc-escher-arch-exp18-ucv-par-smoke" in readme
        assert "--parallel-num-workers 2" in readme
        assert "n2-standard-4 21600 4000 16000 100" in readme
