"""Fixed-beta, lifetime-reservoir control-variate ESCHER."""

from __future__ import annotations

import math

import numpy as np

from fast_slow_escher.solver import ReservoirTransitionBuffer
from unbiased_escher.solver import (
    CrossFittedQEnsemble,
    UnbiasedControlVariateEscher,
)


class LifetimeReservoirCrossFittedQEnsemble(CrossFittedQEnsemble):
    """Experiment 6's cross-fitted critics with uniform lifetime replay.

    Networks, optimisers, target snapshots, fold assignment, held-out
    prediction and training work are unchanged. Only each fold's circular
    recent-transition buffer is replaced by a uniform reservoir over the
    lifetime transition stream.
    """

    def __init__(
        self,
        *,
        ensemble_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers,
        learning_rate: float,
        total_buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger,
        regret_trainers,
        device: str,
        gradient_clip_norm: float,
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            history_size=history_size,
            state_size=state_size,
            action_size=action_size,
            network_layers=network_layers,
            learning_rate=learning_rate,
            total_buffer_size=total_buffer_size,
            batch_size=batch_size,
            train_steps=train_steps,
            logger=logger,
            regret_trainers=regret_trainers,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        member_capacity = max(1, int(total_buffer_size) // int(ensemble_size))
        for member in self.members:
            member.buffer = ReservoirTransitionBuffer(
                member_capacity,
                history_size,
                state_size,
                action_size,
                device,
            )

    def fold_lifetime_seen_counts(self):
        return [int(member.buffer.seen_count) for member in self.members]


class FixedBetaReservoirEscher(UnbiasedControlVariateEscher):
    """Recommended post-Experiment-12 architecture.

    The always-unbiased residual correction is fixed at beta=1. The three
    persistent frozen-target critics remain strictly cross-fitted, but their
    replay represents the complete transition history rather than only the
    most recent buffer window. Experiment 6's residual calibration, adaptive
    full-support sampling, gated predictor, regret learner and average-policy
    learner are otherwise unchanged.
    """

    def __init__(
        self,
        *args,
        fixed_control_variate_beta: float = 1.0,
        **kwargs,
    ):
        if not math.isclose(float(fixed_control_variate_beta), 1.0):
            raise ValueError("FixedBetaReservoirEscher requires beta=1")
        super().__init__(
            *args,
            fixed_control_variate_beta=1.0,
            **kwargs,
        )

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = LifetimeReservoirCrossFittedQEnsemble(
            ensemble_size=self.q_ensemble_size,
            history_size=history_size,
            state_size=self.infostate_size,
            action_size=self.action_size,
            network_layers=self.network_layers,
            learning_rate=self.learning_rate,
            total_buffer_size=self.baseline_buffer_size,
            batch_size=self.baseline_batch_size,
            train_steps=self.baseline_network_train_steps,
            logger=self.logger,
            regret_trainers=self.regret_trainers,
            device=self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
        )

    def evaluate(self, **kwargs):
        lifetime_counts = self.q_value_trainer.fold_lifetime_seen_counts()
        self.logger.record(
            "q_lifetime_seen_count",
            int(sum(lifetime_counts)),
        )
        for fold, count in enumerate(lifetime_counts):
            self.logger.record(f"q_fold_{fold}_lifetime_seen_count", count)
        return super().evaluate(**kwargs)
