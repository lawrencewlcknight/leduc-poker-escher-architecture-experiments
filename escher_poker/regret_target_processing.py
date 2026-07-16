"""Reference definitions for ESCHER regret-target processing modes.

The NumPy implementation is a small test oracle for the TensorFlow training
path in :mod:`escher_poker.solver`. Scale-only modes deliberately never add to
or subtract from a target, so they preserve every target sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


RAW = "none"
BATCH_STANDARDIZE = "standardize"
CLIP = "clip"
BATCH_STANDARDIZE_CLIP = "standardize_clip"
FIXED_UTILITY_SCALE = "fixed_utility_scale"
BATCH_RMS = "batch_rms"
EMA_STD = "ema_std"

VALID_REGRET_TARGET_PROCESSING = {
    RAW,
    BATCH_STANDARDIZE,
    CLIP,
    BATCH_STANDARDIZE_CLIP,
    FIXED_UTILITY_SCALE,
    BATCH_RMS,
    EMA_STD,
}


@dataclass(frozen=True)
class PersistentMoments:
    """Exponentially weighted moments used by persistent scale-only processing."""

    mean: float = 0.0
    second_moment: float = 0.0
    initialized: bool = False


@dataclass(frozen=True)
class RegretTargetProcessingResult:
    """Processed targets, applied transform, and sign diagnostics."""

    target: np.ndarray
    applied_mean: float
    applied_scale: float
    sign_flip_fraction: float
    raw_positive_fraction: float
    processed_positive_fraction: float
    persistent_moments: PersistentMoments


def process_regret_targets(
    regrets,
    legal_actions_mask,
    *,
    mode: str = RAW,
    fixed_scale: float = 1.0,
    clip_value: float = 1.0,
    epsilon: float = 1e-6,
    ema_decay: float = 0.99,
    persistent_moments: Optional[PersistentMoments] = None,
) -> RegretTargetProcessingResult:
    """Apply one target-processing mode to a NumPy minibatch.

    ``applied_mean`` is non-zero only for the legacy batch-centred modes.
    ``EMA_STD`` tracks a persistent standard deviation, but only divides the
    targets by it; the tracked mean is never subtracted from the targets.
    """
    processing = str(mode).lower()
    if processing not in VALID_REGRET_TARGET_PROCESSING:
        raise ValueError(
            "mode must be one of "
            f"{sorted(VALID_REGRET_TARGET_PROCESSING)}, got {mode!r}."
        )
    if not np.isfinite(fixed_scale) or fixed_scale <= 0.0:
        raise ValueError("fixed_scale must be positive and finite.")
    if not np.isfinite(clip_value) or clip_value <= 0.0:
        raise ValueError("clip_value must be positive and finite.")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1).")

    raw = np.asarray(regrets, dtype=np.float64)
    mask = np.asarray(legal_actions_mask, dtype=np.float64)
    if raw.shape != mask.shape:
        raise ValueError("regrets and legal_actions_mask must have matching shapes.")
    legal = mask > 0.0
    if not np.any(legal):
        raise ValueError("legal_actions_mask must contain a legal target.")
    if not np.all(np.isfinite(raw[legal])):
        raise ValueError("Legal regret targets must be finite.")

    raw = np.where(legal, raw, 0.0)
    legal_raw = raw[legal]
    processed = raw.copy()
    applied_mean = 0.0
    applied_scale = 1.0
    moments = persistent_moments or PersistentMoments()

    if processing in {BATCH_STANDARDIZE, BATCH_STANDARDIZE_CLIP}:
        applied_mean = float(np.mean(legal_raw))
        applied_scale = max(float(np.std(legal_raw)), float(epsilon))
        processed = np.where(legal, (raw - applied_mean) / applied_scale, 0.0)
    elif processing == FIXED_UTILITY_SCALE:
        applied_scale = float(fixed_scale)
        processed = np.where(legal, raw / applied_scale, 0.0)
    elif processing == BATCH_RMS:
        applied_scale = max(
            float(np.sqrt(np.mean(np.square(legal_raw)))),
            float(epsilon),
        )
        processed = np.where(legal, raw / applied_scale, 0.0)
    elif processing == EMA_STD:
        batch_mean = float(np.mean(legal_raw))
        batch_second_moment = float(np.mean(np.square(legal_raw)))
        if moments.initialized:
            updated_mean = (
                ema_decay * moments.mean + (1.0 - ema_decay) * batch_mean
            )
            updated_second_moment = (
                ema_decay * moments.second_moment
                + (1.0 - ema_decay) * batch_second_moment
            )
        else:
            updated_mean = batch_mean
            updated_second_moment = batch_second_moment
        moments = PersistentMoments(
            mean=float(updated_mean),
            second_moment=float(updated_second_moment),
            initialized=True,
        )
        variance = max(
            moments.second_moment - moments.mean * moments.mean,
            float(epsilon) ** 2,
        )
        applied_scale = float(np.sqrt(variance))
        processed = np.where(legal, raw / applied_scale, 0.0)

    if processing in {CLIP, BATCH_STANDARDIZE_CLIP}:
        processed = np.where(
            legal,
            np.clip(processed, -clip_value, clip_value),
            0.0,
        )

    legal_processed = processed[legal]
    sign_flip_fraction = float(
        np.mean(np.sign(legal_raw) != np.sign(legal_processed))
    )
    return RegretTargetProcessingResult(
        target=processed,
        applied_mean=applied_mean,
        applied_scale=applied_scale,
        sign_flip_fraction=sign_flip_fraction,
        raw_positive_fraction=float(np.mean(legal_raw > 0.0)),
        processed_positive_fraction=float(np.mean(legal_processed > 0.0)),
        persistent_moments=moments,
    )
