"""Tests for configurable ESCHER regret replay backends."""

from __future__ import annotations

import random

import numpy as np
import pytest

from escher_poker.replay import (
    ALL_SAMPLES,
    COUNTERFACTUAL_REACH_WEIGHTED,
    INFOSET_STRATIFIED,
    RARE_HISTORY_QUOTA,
    RESERVOIR,
    make_regret_replay_buffer,
)


def test_all_samples_removes_finite_capacity_approximation():
    buffer = make_regret_replay_buffer(ALL_SAMPLES, 5)
    for index in range(20):
        buffer.add(index, key=f"infoset-{index % 2}")

    assert buffer.get_data() == list(range(20))
    diagnostics = buffer.diagnostics()
    assert diagnostics["stored_count"] == 20
    assert diagnostics["stream_count"] == 20
    assert diagnostics["retention_fraction"] == pytest.approx(1.0)


def test_uniform_reservoir_retains_capacity_and_tracks_infosets():
    np.random.seed(1)
    buffer = make_regret_replay_buffer(RESERVOIR, 10)
    for index in range(100):
        buffer.add(index, key=f"infoset-{index % 4}")

    diagnostics = buffer.diagnostics()
    assert len(buffer.get_data()) == 10
    assert diagnostics["stream_count"] == 100
    assert diagnostics["retention_fraction"] == pytest.approx(0.1)
    assert 1 <= diagnostics["unique_infosets"] <= 4


def test_infoset_stratified_replay_equalizes_saturated_strata():
    np.random.seed(2)
    random.seed(2)
    buffer = make_regret_replay_buffer(INFOSET_STRATIFIED, 12)
    for key, count in [("common", 100), ("medium", 20), ("rare", 5)]:
        for index in range(count):
            buffer.add((key, index), key=key)

    diagnostics = buffer.diagnostics()
    assert len(buffer.get_data()) == 12
    assert diagnostics["unique_infosets"] == 3
    assert diagnostics["samples_per_infoset_min"] == pytest.approx(4.0)
    assert diagnostics["samples_per_infoset_max"] == pytest.approx(4.0)


def test_rare_history_quota_protects_small_infosets_and_stays_bounded():
    np.random.seed(3)
    random.seed(3)
    buffer = make_regret_replay_buffer(
        RARE_HISTORY_QUOTA,
        20,
        rare_history_quota=3,
    )
    for key, count in [("common", 100), ("rare-a", 2), ("rare-b", 1)]:
        for index in range(count):
            buffer.add((key, index), key=key)

    diagnostics = buffer.diagnostics()
    assert len(buffer.get_data()) <= 20
    assert diagnostics["unique_infosets"] == 3
    assert diagnostics["samples_per_infoset_min"] >= 1.0
    assert diagnostics["rare_history_quota"] == 3


def test_counterfactual_reach_weighted_reservoir_prefers_high_reach():
    random.seed(4)
    buffer = make_regret_replay_buffer(
        COUNTERFACTUAL_REACH_WEIGHTED,
        10,
        weight_floor=1e-6,
    )
    for index in range(100):
        buffer.add(("low", index), key="low", weight=0.001)
    for index in range(10):
        buffer.add(("high", index), key="high", weight=1.0)

    diagnostics = buffer.diagnostics()
    assert len(buffer.get_data()) == 10
    assert diagnostics["stored_weight_mean"] > 0.5


def test_counterfactual_reach_weighted_reservoir_sanitizes_invalid_weights():
    random.seed(6)
    buffer = make_regret_replay_buffer(
        COUNTERFACTUAL_REACH_WEIGHTED,
        3,
        weight_floor=1e-4,
    )
    for index, weight in enumerate([np.nan, np.inf, 0.0, -1.0]):
        buffer.add(index, key="infoset", weight=weight)

    diagnostics = buffer.diagnostics()
    assert len(buffer.get_data()) == 3
    assert diagnostics["stored_weight_min"] == pytest.approx(1e-4)
    assert diagnostics["stored_weight_max"] == pytest.approx(1e-4)


@pytest.mark.parametrize(
    "mode",
    [
        RESERVOIR,
        ALL_SAMPLES,
        INFOSET_STRATIFIED,
        RARE_HISTORY_QUOTA,
        COUNTERFACTUAL_REACH_WEIGHTED,
    ],
)
def test_replay_state_round_trip_preserves_stored_samples(mode):
    np.random.seed(5)
    random.seed(5)
    original = make_regret_replay_buffer(mode, 10, rare_history_quota=2)
    for index in range(30):
        original.add(index, key=f"infoset-{index % 3}")
    restored = make_regret_replay_buffer(mode, 10, rare_history_quota=2)
    restored.load_state_dict(original.state_dict())

    assert sorted(restored.get_data()) == sorted(original.get_data())
    assert restored.get_num_calls() == original.get_num_calls()


def test_restored_rare_history_buffer_remains_globally_bounded():
    np.random.seed(7)
    random.seed(7)
    original = make_regret_replay_buffer(
        RARE_HISTORY_QUOTA,
        10,
        rare_history_quota=2,
    )
    for index in range(60):
        original.add(index, key=f"infoset-{index % 3}")

    restored = make_regret_replay_buffer(
        RARE_HISTORY_QUOTA,
        10,
        rare_history_quota=2,
    )
    restored.load_state_dict(original.state_dict())
    for index in range(60, 160):
        restored.add(index, key=f"infoset-{index % 3}")

    assert len(restored.get_data()) == 10
