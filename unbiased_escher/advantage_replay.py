"""Persistent direct-advantage replay for a model-free UCV-ESCHER variant.

The estimator and external-sampling traversal are unchanged.  The only
algorithmic intervention in this module is the regret-learning formulation:
instantaneous UCV advantage observations are retained in a lifetime reservoir
and fitted directly, with Deep-CFR-style iteration weighting.  In particular,
the preceding fitted regret network is never used as a supervised target.
"""

from __future__ import annotations

import numpy as np
import torch

from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .stability import StableUnbiasedControlVariateEscher


class PersistentAdvantageReplayTrainer(VRDCFRPlusRegretTrainer):
    """Fit utility-normalised instantaneous advantages from lifetime replay.

    The reservoir is uniform over every observation seen so far.  Linear
    iteration weights reproduce Deep CFR's weighted-regression objective up to
    an outer-iteration-wide positive scale factor, which keeps the optimiser's
    effective learning rate stable.  Since regret matching is
    invariant to a common positive scale, the learned weighted mean represents
    the corresponding weighted cumulative advantages for policy construction.
    """

    def __init__(self, *args, replay_weight_exponent: float = 1.0, **kwargs):
        self.replay_weight_exponent = float(replay_weight_exponent)
        if self.replay_weight_exponent < 0.0:
            raise ValueError("replay_weight_exponent cannot be negative")
        self.uses_recursive_target = False
        self.replay_is_persistent = True
        self.target_normalization = "game_utility_bound"
        self.last_replay_diagnostics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def reset_buffer(self):
        """Retain all preceding iterations; reservoir replacement is uniform."""

    def train_model(self, T):
        """Fit direct targets without creating or refreshing a target network."""
        loss = None
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] advantage-replay loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, advantages, legal_actions_mask, iterations = samples
        outputs = self.predict(self.model, infostates, legal_actions_mask)

        legal_count = torch.clamp(legal_actions_mask.sum(dim=1), min=1.0)
        per_record_squared_error = (
            torch.square(outputs - advantages) * legal_actions_mask
        ).sum(dim=1) / legal_count

        iteration_values = torch.clamp(iterations.reshape(-1), min=1.0)
        current_iteration = max(float(T), 1.0)
        weights = (self.replay_weight_exponent + 1.0) * torch.pow(
            iteration_values / current_iteration,
            self.replay_weight_exponent,
        )
        loss = torch.mean(weights * per_record_squared_error)

        with torch.no_grad():
            retained = min(int(self.buffer.cur_id), int(self.buffer.buffer_size))
            self.last_replay_diagnostics = {
                "advantage_replay_seen": float(self.buffer.cur_id),
                "advantage_replay_retained": float(retained),
                "advantage_replay_capacity": float(self.buffer.buffer_size),
                "advantage_replay_weight_mean": float(weights.mean().item()),
                "advantage_replay_weight_max": float(weights.max().item()),
                "advantage_target_abs_mean": float(
                    (torch.abs(advantages) * legal_actions_mask).sum().item()
                    / max(float(legal_actions_mask.sum().item()), 1.0)
                ),
                "advantage_target_rms": float(
                    torch.sqrt(
                        (torch.square(advantages) * legal_actions_mask).sum()
                        / torch.clamp(legal_actions_mask.sum(), min=1.0)
                    ).item()
                ),
            }
        return loss


class AdvantageReplayUnbiasedControlVariateEscher(
    StableUnbiasedControlVariateEscher
):
    """UCV estimator with persistent direct instantaneous-advantage replay."""

    def __init__(
        self,
        *args,
        advantage_replay_weight_exponent: float = 1.0,
        **kwargs,
    ):
        self.advantage_replay_weight_exponent = float(
            advantage_replay_weight_exponent
        )
        super().__init__(*args, **kwargs)

    def init_regret_trainers(self):
        if self.use_instantaneous_predictor:
            raise ValueError(
                "Direct advantage replay is defined for the non-predictive UCV arm"
            )
        if self.regret_policy_gradient_clip_norm is not None:
            raise ValueError(
                "Experiment 26 isolates replay and cannot enable gradient clipping"
            )
        self.regret_trainers = [
            PersistentAdvantageReplayTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
                self.alpha,
                replay_weight_exponent=self.advantage_replay_weight_exponent,
            )
            for _ in range(self.num_players)
        ]
        for trainer in self.regret_trainers:
            trainer.predictor_enabled = False

    def evaluate(self, **kwargs):
        for player, trainer in enumerate(self.regret_trainers):
            diagnostics = trainer.last_replay_diagnostics
            retained = min(int(trainer.buffer.cur_id), int(trainer.buffer.buffer_size))
            self.logger.record(
                f"advantage_replay_seen_player_{player}",
                float(trainer.buffer.cur_id),
            )
            self.logger.record(
                f"advantage_replay_retained_player_{player}", float(retained)
            )
            self.logger.record(
                f"advantage_replay_capacity_player_{player}",
                float(trainer.buffer.buffer_size),
            )
            for name in (
                "advantage_replay_weight_mean",
                "advantage_replay_weight_max",
                "advantage_target_abs_mean",
                "advantage_target_rms",
            ):
                self.logger.record(
                    f"{name}_player_{player}", diagnostics.get(name, np.nan)
                )
        self.logger.record(
            "advantage_replay_weight_exponent",
            self.advantage_replay_weight_exponent,
        )
        self.logger.record("uses_recursive_regret_target", 0.0)
        return super().evaluate(**kwargs)


__all__ = [
    "AdvantageReplayUnbiasedControlVariateEscher",
    "PersistentAdvantageReplayTrainer",
]
