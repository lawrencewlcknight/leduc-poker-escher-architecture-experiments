"""Low-overhead online mechanism diagnostics for Experiment 22."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from unbiased_escher import UnbiasedControlVariateEscher
from unbiased_escher.estimator import control_variate_advantage

from .config import BETA_HISTOGRAM_EDGES


@dataclass
class OnlineMoments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    absolute_sum: float = 0.0

    def add(self, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.absolute_sum += abs(value)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    @property
    def mean_absolute(self) -> float:
        return self.absolute_sum / self.count if self.count else math.nan


@dataclass
class InformationActionDiagnostics:
    observed_residual: OnlineMoments = field(default_factory=OnlineMoments)
    predicted_residual_mean: OnlineMoments = field(default_factory=OnlineMoments)
    predicted_residual_variance: OnlineMoments = field(default_factory=OnlineMoments)
    residual_prediction_error: OnlineMoments = field(default_factory=OnlineMoments)
    beta: OnlineMoments = field(default_factory=OnlineMoments)
    importance_correction: OnlineMoments = field(default_factory=OnlineMoments)
    sample_probability: OnlineMoments = field(default_factory=OnlineMoments)
    current_target: OnlineMoments = field(default_factory=OnlineMoments)
    fixed_beta_one_target: OnlineMoments = field(default_factory=OnlineMoments)


@dataclass
class IterationDiagnostics:
    critic_squared_error: OnlineMoments = field(default_factory=OnlineMoments)
    local_regret_absolute: OnlineMoments = field(default_factory=OnlineMoments)


class DiagnosticUCVSolver(UnbiasedControlVariateEscher):
    """UCV solver that observes estimator inputs without changing training."""

    def __init__(self, *args, **kwargs):
        self._information_action = defaultdict(InformationActionDiagnostics)
        self._iteration_information_action = defaultdict(IterationDiagnostics)
        self._beta_histogram = np.zeros(len(BETA_HISTOGRAM_EDGES) - 1, dtype=np.int64)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _information_state(state, player: int) -> str:
        try:
            return str(state.information_state_string(player))
        except Exception:
            tensor = np.asarray(state.information_state_tensor(player), dtype=float)
            return "tensor:" + ",".join(f"{value:.6g}" for value in tensor)

    def _record_detailed_estimate_diagnostics(self, **context: Any) -> None:
        player = int(context["traverser"])
        sampled_action = int(context["sampled_action"])
        legal_mask = np.asarray(context["legal_mask"], dtype=float)
        info_state = self._information_state(context["state"], player)
        q_values = np.asarray(context["q_values"], dtype=float)
        beta = np.asarray(context["beta"], dtype=float)
        residual_means = np.asarray(context["residual_means"], dtype=float)
        predicted_variances = np.asarray(context["predicted_variances"], dtype=float)
        policy = np.asarray(context["policy"], dtype=float)
        estimate = context["estimate"]
        fixed_beta_one = control_variate_advantage(
            q_values,
            beta=np.ones_like(beta),
            sampled_action=sampled_action,
            sample_probability=float(context["sample_probability"]),
            sampled_return=float(context["sampled_return"]),
            policy=policy,
            legal_actions_mask=legal_mask,
        )
        iteration = int(self.num_iteration)
        legal_actions = np.flatnonzero(legal_mask > 0.0)
        for action in legal_actions:
            key = (player, info_state, int(action))
            cell = self._information_action[key]
            cell.current_target.add(float(estimate.advantages[action]))
            cell.fixed_beta_one_target.add(float(fixed_beta_one.advantages[action]))
            iteration_cell = self._iteration_information_action[
                (iteration, player, info_state, int(action))
            ]
            iteration_cell.local_regret_absolute.add(
                abs(float(estimate.advantages[action]))
            )

        key = (player, info_state, sampled_action)
        cell = self._information_action[key]
        observed_residual = float(estimate.q_residual)
        predicted_mean = float(residual_means[sampled_action])
        sampled_beta = float(beta[sampled_action])
        cell.observed_residual.add(observed_residual)
        cell.predicted_residual_mean.add(predicted_mean)
        cell.predicted_residual_variance.add(
            float(predicted_variances[sampled_action])
        )
        cell.residual_prediction_error.add(observed_residual - predicted_mean)
        cell.beta.add(sampled_beta)
        cell.importance_correction.add(float(estimate.importance_correction))
        cell.sample_probability.add(float(context["sample_probability"]))
        self._iteration_information_action[
            (iteration, player, info_state, sampled_action)
        ].critic_squared_error.add(observed_residual * observed_residual)

        bin_index = int(
            np.searchsorted(BETA_HISTOGRAM_EDGES, sampled_beta, side="right") - 1
        )
        bin_index = max(0, min(bin_index, len(self._beta_histogram) - 1))
        self._beta_histogram[bin_index] += 1

    def information_action_rows(self) -> list[dict]:
        rows = []
        for (player, info_state, action), cell in sorted(self._information_action.items()):
            target_variance = cell.current_target.variance
            beta_one_variance = cell.fixed_beta_one_target.variance
            rows.append(
                {
                    "player": player,
                    "information_state": info_state,
                    "action": action,
                    "sample_count": cell.observed_residual.count,
                    "target_count": cell.current_target.count,
                    "observed_residual_mean": cell.observed_residual.mean,
                    "observed_residual_variance": cell.observed_residual.variance,
                    "predicted_residual_mean": cell.predicted_residual_mean.mean,
                    "residual_mean_calibration_error": cell.residual_prediction_error.mean,
                    "residual_mean_calibration_rmse": math.sqrt(
                        cell.residual_prediction_error.variance
                        + cell.residual_prediction_error.mean**2
                    ),
                    "mean_predicted_residual_variance": (
                        cell.predicted_residual_variance.mean
                    ),
                    "residual_variance_calibration_ratio": (
                        cell.observed_residual.variance
                        / cell.predicted_residual_variance.mean
                        if cell.predicted_residual_variance.mean > 0.0
                        else math.nan
                    ),
                    "beta_mean": cell.beta.mean,
                    "beta_standard_deviation": cell.beta.standard_deviation,
                    "beta_min": cell.beta.minimum if cell.beta.count else math.nan,
                    "beta_max": cell.beta.maximum if cell.beta.count else math.nan,
                    "importance_correction_abs_mean": (
                        cell.importance_correction.mean_absolute
                    ),
                    "sample_probability_mean": cell.sample_probability.mean,
                    "realised_target_variance": target_variance,
                    "counterfactual_fixed_beta_one_target_variance": beta_one_variance,
                    "target_variance_ratio_vs_fixed_beta_one": (
                        target_variance / beta_one_variance
                        if beta_one_variance > 0.0
                        else math.nan
                    ),
                }
            )
        return rows

    def beta_histogram_rows(self) -> list[dict]:
        total = int(np.sum(self._beta_histogram))
        rows = []
        for index, count in enumerate(self._beta_histogram):
            rows.append(
                {
                    "bin_lower": float(BETA_HISTOGRAM_EDGES[index]),
                    "bin_upper": float(BETA_HISTOGRAM_EDGES[index + 1]),
                    "count": int(count),
                    "fraction": float(count / total) if total else 0.0,
                }
            )
        return rows

    def critic_subsequent_regret_rows(self) -> list[dict]:
        rows = []
        cells = self._iteration_information_action
        for (iteration, player, info_state, action), current in sorted(cells.items()):
            following = cells.get((iteration + 1, player, info_state, action))
            if current.critic_squared_error.count == 0 or following is None:
                continue
            if following.local_regret_absolute.count == 0:
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "next_iteration": iteration + 1,
                    "player": player,
                    "information_state": info_state,
                    "action": action,
                    "critic_sample_count": current.critic_squared_error.count,
                    "sampled_critic_target_rmse": math.sqrt(
                        current.critic_squared_error.mean
                    ),
                    "next_local_regret_sample_count": (
                        following.local_regret_absolute.count
                    ),
                    "next_local_regret_target_abs_mean": (
                        following.local_regret_absolute.mean
                    ),
                }
            )
        return rows


__all__ = ["DiagnosticUCVSolver", "OnlineMoments"]
