"""Unit tests for ESCHER instantaneous regret-target definitions."""

from __future__ import annotations

import numpy as np
import pytest

from escher_poker.regret_targets import (
    AUTHOR_STATE_VALUE,
    PAPER_POLICY_WEIGHTED_Q,
    compute_regret_target,
)


def test_author_target_reproduces_state_value_subtraction():
    result = compute_regret_target(
        action_values=np.array([2.0, -1.0, 99.0]),
        state_value=0.25,
        policy=np.array([0.25, 0.75, 0.0]),
        legal_actions_mask=np.array([1.0, 1.0, 0.0]),
        baseline_mode=AUTHOR_STATE_VALUE,
    )

    np.testing.assert_allclose(result.target, [1.75, -1.25, 0.0])
    assert result.baseline == pytest.approx(0.25)
    assert result.policy_weighted_q == pytest.approx(-0.25)
    assert result.bellman_residual == pytest.approx(0.5)
    assert result.policy_weighted_target == pytest.approx(-0.5)


def test_paper_target_is_policy_weighted_and_internally_centered():
    result = compute_regret_target(
        action_values=np.array([2.0, -1.0, 99.0]),
        state_value=0.25,
        policy=np.array([0.25, 0.75, 0.0]),
        legal_actions_mask=np.array([1.0, 1.0, 0.0]),
        baseline_mode=PAPER_POLICY_WEIGHTED_Q,
    )

    np.testing.assert_allclose(result.target, [2.25, -0.75, 0.0])
    assert result.baseline == pytest.approx(-0.25)
    assert result.policy_weighted_target == pytest.approx(0.0, abs=1e-12)
    assert result.bellman_residual == pytest.approx(0.5)


def test_target_definitions_agree_when_value_predictions_are_bellman_consistent():
    action_values = np.array([3.0, 1.0, -2.0])
    policy = np.array([0.2, 0.3, 0.5])
    state_value = float(np.dot(policy, action_values))

    author = compute_regret_target(
        action_values,
        state_value,
        policy,
        np.ones(3),
        baseline_mode=AUTHOR_STATE_VALUE,
    )
    paper = compute_regret_target(
        action_values,
        state_value,
        policy,
        np.ones(3),
        baseline_mode=PAPER_POLICY_WEIGHTED_Q,
    )

    np.testing.assert_allclose(author.target, paper.target)
    assert author.bellman_residual == pytest.approx(0.0, abs=1e-12)


def test_regret_target_rejects_unknown_mode():
    with pytest.raises(ValueError, match="baseline_mode"):
        compute_regret_target(
            [1.0, 2.0],
            1.5,
            [0.5, 0.5],
            [1.0, 1.0],
            baseline_mode="unknown",
        )
