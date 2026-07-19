"""Advantage-variance-aligned sampling for unbiased ESCHER."""

from .solver import (
    AdvantageVarianceSamplingDecision,
    AdvantageVarianceSamplingEscher,
    advantage_variance_sampling_decision,
    advantage_variance_sampling_policy,
)

__all__ = [
    "AdvantageVarianceSamplingDecision",
    "AdvantageVarianceSamplingEscher",
    "advantage_variance_sampling_decision",
    "advantage_variance_sampling_policy",
]

