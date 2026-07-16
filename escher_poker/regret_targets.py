"""Regret-target constructions used by the ESCHER solver.

Keeping these formulae in a small NumPy-only module makes the algorithmic
choice explicit, testable, and independent from traversal/replay mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AUTHOR_STATE_VALUE = "author_state_value"
PAPER_POLICY_WEIGHTED_Q = "paper_policy_weighted_q"
VALID_REGRET_TARGET_BASELINES = {
    AUTHOR_STATE_VALUE,
    PAPER_POLICY_WEIGHTED_Q,
}


@dataclass(frozen=True)
class RegretTargetResult:
    """A raw instantaneous regret target and its consistency diagnostics."""

    target: np.ndarray
    baseline: float
    policy_weighted_q: float
    policy_weighted_target: float
    bellman_residual: float


def compute_regret_target(
    action_values,
    state_value,
    policy,
    legal_actions_mask,
    *,
    baseline_mode: str = AUTHOR_STATE_VALUE,
) -> RegretTargetResult:
    """Construct one ESCHER instantaneous regret vector.

    ``author_state_value`` reproduces the public Sandholm-Lab implementation:
    ``Q_hat(h, a) - V_hat(h)``. ``paper_policy_weighted_q`` implements Equation
    7 / Algorithm 2 of the paper: ``Q_hat(h, a) - pi dot Q_hat(h, .)``.

    The policy is normalised over legal actions before it is used. Illegal
    target entries are always exactly zero.
    """
    mode = str(baseline_mode).lower()
    if mode not in VALID_REGRET_TARGET_BASELINES:
        raise ValueError(
            "baseline_mode must be one of "
            f"{sorted(VALID_REGRET_TARGET_BASELINES)}, got {baseline_mode!r}."
        )

    values = np.asarray(action_values, dtype=np.float64)
    policy_array = np.asarray(policy, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    if values.ndim != 1 or policy_array.shape != values.shape or legal_mask.shape != values.shape:
        raise ValueError(
            "action_values, policy, and legal_actions_mask must be matching 1-D arrays."
        )

    legal = legal_mask > 0.0
    if not np.any(legal):
        raise ValueError("legal_actions_mask must contain at least one legal action.")
    if not np.all(np.isfinite(values[legal])):
        raise ValueError("Legal action values must be finite.")

    legal_policy = np.where(legal, policy_array, 0.0)
    policy_mass = float(np.sum(legal_policy))
    if not np.isfinite(policy_mass) or policy_mass <= 0.0:
        raise ValueError("Policy must assign positive finite mass to legal actions.")
    legal_policy = legal_policy / policy_mass

    scalar_state_value = float(np.asarray(state_value, dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(scalar_state_value):
        raise ValueError("state_value must be finite.")

    policy_weighted_q = float(np.dot(legal_policy, values))
    baseline = (
        scalar_state_value
        if mode == AUTHOR_STATE_VALUE
        else policy_weighted_q
    )
    target = np.where(legal, values - baseline, 0.0)
    policy_weighted_target = float(np.dot(legal_policy, target))

    return RegretTargetResult(
        target=target,
        baseline=float(baseline),
        policy_weighted_q=policy_weighted_q,
        policy_weighted_target=policy_weighted_target,
        bellman_residual=float(scalar_state_value - policy_weighted_q),
    )
