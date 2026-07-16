"""Tests for fixed uniform, balanced, and tempered ESCHER sampling."""

from __future__ import annotations

import numpy as np
import pytest

from escher_poker.fixed_sampling import fixed_sampling_policy


MASK = np.asarray([1.0, 1.0, 0.0])
BALANCED = np.asarray([0.8, 0.2, 0.0])


def test_zero_mix_is_exactly_uniform_over_legal_actions():
    result = fixed_sampling_policy(MASK, BALANCED, 0.0)
    np.testing.assert_allclose(result, [0.5, 0.5, 0.0])


def test_unit_mix_is_exactly_balanced():
    result = fixed_sampling_policy(MASK, BALANCED, 1.0)
    np.testing.assert_allclose(result, BALANCED)


def test_half_mix_is_fixed_convex_tempering_with_full_support():
    result = fixed_sampling_policy(MASK, BALANCED, 0.5)
    np.testing.assert_allclose(result, [0.65, 0.35, 0.0])
    assert np.all(result[MASK > 0.0] > 0.0)


@pytest.mark.parametrize("mix", [-0.1, 1.1, np.nan])
def test_invalid_mix_is_rejected(mix):
    with pytest.raises(ValueError, match="balanced_mix"):
        fixed_sampling_policy(MASK, BALANCED, mix)


def test_illegal_balanced_mass_is_rejected():
    with pytest.raises(ValueError, match="illegal"):
        fixed_sampling_policy(MASK, [0.4, 0.4, 0.2], 0.5)
