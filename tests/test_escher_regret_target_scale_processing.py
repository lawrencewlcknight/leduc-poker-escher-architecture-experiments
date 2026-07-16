"""Tests for sign-preserving ESCHER regret-target scaling modes."""

from __future__ import annotations

import numpy as np
import pytest

from escher_poker.regret_target_processing import (
    BATCH_RMS,
    BATCH_STANDARDIZE,
    EMA_STD,
    FIXED_UTILITY_SCALE,
    RAW,
    process_regret_targets,
)


def test_batch_centering_can_flip_target_signs():
    result = process_regret_targets(
        [[1.0, 2.0]],
        [[1.0, 1.0]],
        mode=BATCH_STANDARDIZE,
    )

    np.testing.assert_allclose(result.target, [[-1.0, 1.0]])
    assert result.applied_mean == pytest.approx(1.5)
    assert result.applied_scale == pytest.approx(0.5)
    assert result.sign_flip_fraction == pytest.approx(0.5)
    assert result.raw_positive_fraction == pytest.approx(1.0)
    assert result.processed_positive_fraction == pytest.approx(0.5)


@pytest.mark.parametrize("mode", [RAW, FIXED_UTILITY_SCALE, BATCH_RMS])
def test_non_centered_static_modes_preserve_every_legal_sign(mode):
    result = process_regret_targets(
        [[-4.0, 2.0, 99.0], [1.0, -3.0, 99.0]],
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        mode=mode,
        fixed_scale=26.0,
    )

    assert result.sign_flip_fraction == pytest.approx(0.0)
    assert result.raw_positive_fraction == result.processed_positive_fraction
    np.testing.assert_allclose(result.target[:, 2], 0.0)


def test_fixed_utility_scaling_uses_one_game_wide_divisor():
    result = process_regret_targets(
        [[-13.0, 13.0]],
        [[1.0, 1.0]],
        mode=FIXED_UTILITY_SCALE,
        fixed_scale=26.0,
    )

    np.testing.assert_allclose(result.target, [[-0.5, 0.5]])
    assert result.applied_mean == pytest.approx(0.0)
    assert result.applied_scale == pytest.approx(26.0)


def test_batch_rms_divides_without_subtracting_mean():
    result = process_regret_targets(
        [[3.0, 4.0]],
        [[1.0, 1.0]],
        mode=BATCH_RMS,
    )

    expected_scale = np.sqrt((9.0 + 16.0) / 2.0)
    np.testing.assert_allclose(result.target, [[3.0, 4.0]] / expected_scale)
    assert result.applied_mean == pytest.approx(0.0)
    assert result.applied_scale == pytest.approx(expected_scale)


def test_ema_std_persists_moments_but_does_not_center_targets():
    first = process_regret_targets(
        [[1.0, 3.0]],
        [[1.0, 1.0]],
        mode=EMA_STD,
        ema_decay=0.5,
    )
    second = process_regret_targets(
        [[3.0, 5.0]],
        [[1.0, 1.0]],
        mode=EMA_STD,
        ema_decay=0.5,
        persistent_moments=first.persistent_moments,
    )

    assert first.persistent_moments.mean == pytest.approx(2.0)
    assert first.persistent_moments.second_moment == pytest.approx(5.0)
    assert second.persistent_moments.mean == pytest.approx(3.0)
    assert second.persistent_moments.second_moment == pytest.approx(11.0)
    assert second.applied_mean == pytest.approx(0.0)
    assert second.applied_scale == pytest.approx(np.sqrt(2.0))
    np.testing.assert_allclose(second.target, [[3.0, 5.0]] / np.sqrt(2.0))
    assert second.sign_flip_fraction == pytest.approx(0.0)
