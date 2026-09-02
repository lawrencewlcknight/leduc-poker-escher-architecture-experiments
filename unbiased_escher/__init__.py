"""Always-unbiased, uncertainty-adaptive control-variate ESCHER."""

from .estimator import (
    ControlVariateEstimate,
    control_variate_advantage,
    residual_adaptive_sampling_policy,
    variance_optimal_beta,
)
from .solver import UnbiasedControlVariateEscher
from .stability import StableUnbiasedControlVariateEscher


def __getattr__(name):
    if name == "ParallelUnbiasedControlVariateEscher":
        from .parallel_solver import ParallelUnbiasedControlVariateEscher

        return ParallelUnbiasedControlVariateEscher
    raise AttributeError(name)

__all__ = [
    "ControlVariateEstimate",
    "UnbiasedControlVariateEscher",
    "StableUnbiasedControlVariateEscher",
    "ParallelUnbiasedControlVariateEscher",
    "control_variate_advantage",
    "residual_adaptive_sampling_policy",
    "variance_optimal_beta",
]
