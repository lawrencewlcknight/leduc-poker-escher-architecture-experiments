"""Helpers for fixed ESCHER sampling policies."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def fixed_sampling_policy(
    legal_actions_mask: Iterable[float],
    balanced_action_probs: Iterable[float],
    balanced_mix: float,
) -> np.ndarray:
    """Blend uniform and leaf-balanced action policies on legal actions.

    ``balanced_mix=0`` is exactly uniform, ``balanced_mix=1`` is exactly the
    supplied balanced policy, and intermediate values preserve full support
    through a convex mixture.
    """
    mask = np.asarray(legal_actions_mask, dtype=np.float64)
    balanced = np.asarray(balanced_action_probs, dtype=np.float64)
    mix = float(balanced_mix)

    if mask.ndim != 1 or balanced.ndim != 1 or mask.shape != balanced.shape:
        raise ValueError("legal mask and balanced probabilities must be 1-D peers.")
    if not np.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError("balanced_mix must be finite and in [0, 1].")

    legal = mask > 0.0
    if not np.any(legal):
        raise ValueError("fixed sampling requires at least one legal action.")
    if np.any(~np.isfinite(balanced)) or np.any(balanced < 0.0):
        raise ValueError("balanced probabilities must be finite and non-negative.")
    if np.any(balanced[~legal] != 0.0):
        raise ValueError("balanced probabilities must be zero on illegal actions.")

    uniform = legal.astype(np.float64) / np.count_nonzero(legal)
    balanced = balanced * legal
    balanced_sum = float(np.sum(balanced))
    if balanced_sum <= 0.0:
        raise ValueError("balanced probabilities must have positive legal mass.")
    balanced = balanced / balanced_sum

    policy = (1.0 - mix) * uniform + mix * balanced
    policy *= legal
    return policy / np.sum(policy)
