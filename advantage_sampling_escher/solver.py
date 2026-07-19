"""Sample actions to minimise centred-advantage target variance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from unbiased_escher import residual_adaptive_sampling_policy
from unbiased_escher.solver import UnbiasedControlVariateEscher


@dataclass(frozen=True)
class AdvantageVarianceSamplingDecision:
    """Predictable sampling distribution and its variance components."""

    policy: np.ndarray
    control_residual_second_moments: np.ndarray
    centering_influence_norms: np.ndarray
    scores: np.ndarray
    variance_proxy: float


def advantage_variance_sampling_decision(
    q_values,
    *,
    beta,
    predicted_residual_means,
    predicted_residual_variances,
    current_policy,
    legal_actions_mask,
    uniform_floor_mass: float,
    minimum_variance: float = 1e-6,
) -> AdvantageVarianceSamplingDecision:
    """Construct the centred-advantage second-moment proposal.

    The calibration model predicts moments of ``R=G-Q_hat``. For the actual
    control ``C=beta*Q_hat``, the residual is
    ``G-C = R + (1-beta)*Q_hat``. Its predicted second moment is multiplied by
    ``||(I-1*pi^T)e_a||^2`` before taking the square root.
    """

    if not 0.0 < uniform_floor_mass <= 1.0:
        raise ValueError("uniform_floor_mass must lie in (0, 1]")
    if minimum_variance <= 0.0:
        raise ValueError("minimum_variance must be positive")
    q_values = np.asarray(q_values, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    residual_means = np.asarray(predicted_residual_means, dtype=np.float64)
    residual_variances = np.asarray(
        predicted_residual_variances,
        dtype=np.float64,
    )
    policy = np.asarray(current_policy, dtype=np.float64)
    legal_mask = np.asarray(legal_actions_mask, dtype=np.float64)
    arrays = (beta, residual_means, residual_variances, policy, legal_mask)
    if q_values.ndim != 1 or any(value.shape != q_values.shape for value in arrays):
        raise ValueError("All sampling inputs must be matching one-dimensional arrays")
    legal = legal_mask > 0.0
    legal_count = int(np.sum(legal))
    if legal_count == 0:
        raise ValueError("legal_actions_mask must contain a legal action")
    for name, values in (
        ("q_values", q_values),
        ("beta", beta),
        ("predicted_residual_means", residual_means),
        ("predicted_residual_variances", residual_variances),
        ("current_policy", policy),
    ):
        if not np.all(np.isfinite(values[legal])):
            raise ValueError(f"{name} must be finite on legal actions")
    legal_policy = np.where(legal, policy, 0.0)
    policy_mass = float(np.sum(legal_policy))
    if not np.isfinite(policy_mass) or policy_mass <= 0.0:
        raise ValueError("current_policy must have positive finite legal mass")
    legal_policy /= policy_mass

    safe_variances = np.where(
        legal,
        np.maximum(
            residual_variances,
            minimum_variance,
        ),
        0.0,
    )
    control_residual_means = residual_means + (1.0 - beta) * q_values
    second_moments = np.where(
        legal,
        safe_variances + np.square(control_residual_means),
        0.0,
    )
    influence_squared = np.where(
        legal,
        np.maximum(
            1.0
            - 2.0 * legal_policy
            + float(legal_count) * np.square(legal_policy),
            0.0,
        ),
        0.0,
    )
    influence_norms = np.sqrt(influence_squared)
    scores = np.where(legal, np.sqrt(second_moments) * influence_norms, 0.0)
    score_mass = float(np.sum(scores))
    uniform = np.where(legal, 1.0 / float(legal_count), 0.0)
    adaptive = scores / score_mass if score_mass > 0.0 else uniform
    sample_policy = (
        (1.0 - uniform_floor_mass) * adaptive
        + uniform_floor_mass * uniform
    )
    sample_policy = np.where(legal, sample_policy, 0.0)
    sample_policy /= float(np.sum(sample_policy))
    variance_proxy = float(
        np.sum(
            second_moments[legal]
            * influence_squared[legal]
            / sample_policy[legal]
        )
    )
    return AdvantageVarianceSamplingDecision(
        policy=sample_policy,
        control_residual_second_moments=second_moments,
        centering_influence_norms=influence_norms,
        scores=scores,
        variance_proxy=variance_proxy,
    )


def advantage_variance_sampling_policy(*args, **kwargs) -> np.ndarray:
    """Return only the action distribution from the full decision."""

    return advantage_variance_sampling_decision(*args, **kwargs).policy


class AdvantageVarianceSamplingEscher(UnbiasedControlVariateEscher):
    """Experiment 6 with centred-advantage-aligned action sampling."""

    def _reset_architecture_diagnostics(self) -> None:
        super()._reset_architecture_diagnostics()
        self._advantage_sampling_stats: Dict[str, float] = {
            "state_count": 0.0,
            "action_count": 0.0,
            "second_moment_sum": 0.0,
            "influence_norm_sum": 0.0,
            "influence_norm_min": float("inf"),
            "influence_norm_max": float("-inf"),
            "score_sum": 0.0,
            "probability_min": 1.0,
            "probability_max": 0.0,
            "entropy_sum": 0.0,
            "advantage_proxy_sum": 0.0,
            "experiment_6_proxy_sum": 0.0,
        }

    def _traverser_sampling_policy(
        self,
        *,
        q_values,
        beta,
        residual_means,
        predicted_variances,
        policy,
        legal_mask,
    ) -> np.ndarray:
        decision = advantage_variance_sampling_decision(
            q_values,
            beta=beta,
            predicted_residual_means=residual_means,
            predicted_residual_variances=predicted_variances,
            current_policy=policy,
            legal_actions_mask=legal_mask,
            uniform_floor_mass=self.sampling_uniform_floor_mass,
            minimum_variance=self.calibration_minimum_variance,
        )
        experiment_6_policy = residual_adaptive_sampling_policy(
            predicted_variances,
            legal_mask,
            uniform_floor_mass=self.sampling_uniform_floor_mass,
            minimum_variance=self.calibration_minimum_variance,
        )
        legal = np.asarray(legal_mask, dtype=np.float64) > 0.0
        influence_squared = np.square(decision.centering_influence_norms)
        experiment_6_proxy = float(
            np.sum(
                decision.control_residual_second_moments[legal]
                * influence_squared[legal]
                / experiment_6_policy[legal]
            )
        )
        legal_probabilities = decision.policy[legal]
        legal_influences = decision.centering_influence_norms[legal]
        stats = self._advantage_sampling_stats
        stats["state_count"] += 1.0
        stats["action_count"] += float(np.sum(legal))
        stats["second_moment_sum"] += float(
            np.sum(decision.control_residual_second_moments[legal])
        )
        stats["influence_norm_sum"] += float(np.sum(legal_influences))
        stats["influence_norm_min"] = min(
            stats["influence_norm_min"],
            float(np.min(legal_influences)),
        )
        stats["influence_norm_max"] = max(
            stats["influence_norm_max"],
            float(np.max(legal_influences)),
        )
        stats["score_sum"] += float(np.sum(decision.scores[legal]))
        stats["probability_min"] = min(
            stats["probability_min"],
            float(np.min(legal_probabilities)),
        )
        stats["probability_max"] = max(
            stats["probability_max"],
            float(np.max(legal_probabilities)),
        )
        stats["entropy_sum"] += float(
            -np.sum(legal_probabilities * np.log(legal_probabilities))
        )
        stats["advantage_proxy_sum"] += decision.variance_proxy
        stats["experiment_6_proxy_sum"] += experiment_6_proxy
        return decision.policy

    def evaluate(self, **kwargs):
        stats = self._advantage_sampling_stats
        state_count = stats["state_count"]
        action_count = stats["action_count"]
        action_denominator = max(action_count, 1.0)
        state_denominator = max(state_count, 1.0)
        baseline_proxy = stats["experiment_6_proxy_sum"]
        self.logger.record("advantage_sampler_state_count", state_count)
        self.logger.record(
            "predicted_control_residual_second_moment_mean",
            stats["second_moment_sum"] / action_denominator,
        )
        self.logger.record(
            "centering_influence_norm_mean",
            stats["influence_norm_sum"] / action_denominator,
        )
        self.logger.record(
            "centering_influence_norm_min",
            stats["influence_norm_min"] if action_count else np.nan,
        )
        self.logger.record(
            "centering_influence_norm_max",
            stats["influence_norm_max"] if action_count else np.nan,
        )
        self.logger.record(
            "advantage_sampling_score_mean",
            stats["score_sum"] / action_denominator,
        )
        self.logger.record(
            "advantage_sampling_probability_min",
            stats["probability_min"] if action_count else np.nan,
        )
        self.logger.record(
            "advantage_sampling_probability_max",
            stats["probability_max"] if action_count else np.nan,
        )
        self.logger.record(
            "advantage_sampling_entropy_mean",
            stats["entropy_sum"] / state_denominator,
        )
        self.logger.record(
            "predicted_advantage_variance_proxy_mean",
            stats["advantage_proxy_sum"] / state_denominator,
        )
        self.logger.record(
            "experiment_6_sampling_variance_proxy_mean",
            baseline_proxy / state_denominator,
        )
        self.logger.record(
            "predicted_advantage_variance_proxy_ratio_vs_experiment_6",
            (
                stats["advantage_proxy_sum"] / baseline_proxy
                if baseline_proxy > 0.0
                else np.nan
            ),
        )
        return super().evaluate(**kwargs)
