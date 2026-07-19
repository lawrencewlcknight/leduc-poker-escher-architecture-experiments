"""Parallel multi-action residual correction for unbiased ESCHER."""

from .solver import (
    AdaptiveSubsetDecision,
    CoupledRolloutStreams,
    MultiActionControlVariateEstimate,
    ParallelMultiActionResidualEscher,
    adaptive_nonempty_subset,
    multi_action_control_variate_advantage,
)

__all__ = [
    "AdaptiveSubsetDecision",
    "CoupledRolloutStreams",
    "MultiActionControlVariateEstimate",
    "ParallelMultiActionResidualEscher",
    "adaptive_nonempty_subset",
    "multi_action_control_variate_advantage",
]
