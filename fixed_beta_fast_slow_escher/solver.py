"""Fixed-beta ESCHER with the complete fast/slow cross-fitted control critic."""

from __future__ import annotations

import math

from fast_slow_escher import FastSlowControlCriticEscher


class FixedBetaFastSlowControlCriticEscher(FastSlowControlCriticEscher):
    """Experiment 9's full control critic with beta fixed exactly at one.

    The estimator remains always unbiased. Imperfect fast/slow critic and rho
    predictions can change its variance but not its expectation. Control-side
    replay receives deterministic component-local RNG streams so reservoir
    replacement and critic/controller minibatches cannot advance the global
    Python RNG used by the regret, calibration and average-policy learners.
    """

    def __init__(
        self,
        *args,
        fixed_control_variate_beta: float = 1.0,
        control_replay_seed: int | None = None,
        **kwargs,
    ):
        if not math.isclose(float(fixed_control_variate_beta), 1.0):
            raise ValueError(
                "FixedBetaFastSlowControlCriticEscher requires beta=1"
            )
        if control_replay_seed is None:
            # Keep the control-side streams reproducible and distinct from the
            # process-wide seed without relying on Python's randomised hash().
            run_seed = int(kwargs.get("seed", 0))
            control_replay_seed = 1_000_003 + 100_003 * run_seed
        super().__init__(
            *args,
            fixed_control_variate_beta=1.0,
            control_replay_seed=int(control_replay_seed),
            **kwargs,
        )
