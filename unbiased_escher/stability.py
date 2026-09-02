"""Opt-in optimisation controls for long-horizon UCV-ESCHER studies."""

from __future__ import annotations

import math

import torch

from vr_deep_cfr.solver import AvePolicyTrainer
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .solver import UnbiasedControlVariateEscher


class GradientClippedVRDCFRPlusRegretTrainer(VRDCFRPlusRegretTrainer):
    """VR-DCFR+ regret fitting with an explicit global gradient-norm cap."""

    def __init__(self, *args, gradient_clip_norm: float, **kwargs):
        self.gradient_clip_norm = float(gradient_clip_norm)
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        super().__init__(*args, **kwargs)

    def train_model(self, T):
        loss = None
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] clipped regret loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()


class GradientClippedAvePolicyTrainer(AvePolicyTrainer):
    """Average-policy fitting with an explicit global gradient-norm cap."""

    def __init__(self, *args, gradient_clip_norm: float, **kwargs):
        self.gradient_clip_norm = float(gradient_clip_norm)
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        super().__init__(*args, **kwargs)

    def train_model(self, T):
        loss = None
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] clipped policy loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        return loss.item()


class StableUnbiasedControlVariateEscher(UnbiasedControlVariateEscher):
    """UCV-ESCHER with optional late LR annealing and fit-time clipping.

    Defaults reproduce :class:`UnbiasedControlVariateEscher`.  The controls are
    deliberately restricted to the cumulative-regret and average-policy heads;
    critic and residual-calibration learning rates remain unchanged.
    """

    def __init__(
        self,
        *args,
        regret_policy_gradient_clip_norm: float | None = None,
        anneal_start_nodes: int | None = None,
        anneal_end_nodes: int | None = None,
        anneal_final_learning_rate: float | None = None,
        **kwargs,
    ):
        self.regret_policy_gradient_clip_norm = (
            None
            if regret_policy_gradient_clip_norm is None
            else float(regret_policy_gradient_clip_norm)
        )
        self.anneal_start_nodes = (
            None if anneal_start_nodes is None else int(anneal_start_nodes)
        )
        self.anneal_end_nodes = (
            None if anneal_end_nodes is None else int(anneal_end_nodes)
        )
        self.anneal_final_learning_rate = (
            None
            if anneal_final_learning_rate is None
            else float(anneal_final_learning_rate)
        )
        super().__init__(*args, **kwargs)
        self.initial_regret_policy_learning_rate = float(self.learning_rate)
        self.current_regret_policy_learning_rate = float(self.learning_rate)
        self._validate_optimisation_controls()

    def _validate_optimisation_controls(self) -> None:
        if self.regret_policy_gradient_clip_norm is not None:
            if self.regret_policy_gradient_clip_norm <= 0.0:
                raise ValueError("regret_policy_gradient_clip_norm must be positive")
            if self.use_instantaneous_predictor:
                raise ValueError(
                    "The clipped development arm must disable the instantaneous predictor"
                )
        values = (
            self.anneal_start_nodes,
            self.anneal_end_nodes,
            self.anneal_final_learning_rate,
        )
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError("All annealing fields must be set together")
        if self.anneal_start_nodes is not None:
            if self.anneal_start_nodes < 0:
                raise ValueError("anneal_start_nodes cannot be negative")
            if self.anneal_end_nodes <= self.anneal_start_nodes:
                raise ValueError("anneal_end_nodes must exceed anneal_start_nodes")
            if not 0.0 < self.anneal_final_learning_rate <= self.learning_rate:
                raise ValueError(
                    "anneal_final_learning_rate must be positive and no larger than the initial rate"
                )

    def init_ave_policy_trainer(self):
        if self.regret_policy_gradient_clip_norm is None:
            return super().init_ave_policy_trainer()
        self.ave_policy_trainer = GradientClippedAvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.device,
            self.gamma,
            gradient_clip_norm=self.regret_policy_gradient_clip_norm,
        )

    def init_regret_trainers(self):
        if self.regret_policy_gradient_clip_norm is None:
            return super().init_regret_trainers()
        if self.use_instantaneous_predictor:
            raise ValueError("Clipping is implemented for the non-predictive arm")
        self.regret_trainers = [
            GradientClippedVRDCFRPlusRegretTrainer(
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
                gradient_clip_norm=self.regret_policy_gradient_clip_norm,
            )
            for _ in range(self.num_players)
        ]
        for trainer in self.regret_trainers:
            trainer.predictor_enabled = False

    def scheduled_learning_rate(self, nodes_touched: int | None = None) -> float:
        nodes = self.nodes_touched if nodes_touched is None else int(nodes_touched)
        if self.anneal_start_nodes is None or nodes <= self.anneal_start_nodes:
            return self.initial_regret_policy_learning_rate
        if nodes >= self.anneal_end_nodes:
            return float(self.anneal_final_learning_rate)
        progress = (nodes - self.anneal_start_nodes) / (
            self.anneal_end_nodes - self.anneal_start_nodes
        )
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(
            self.anneal_final_learning_rate
            + (self.initial_regret_policy_learning_rate - self.anneal_final_learning_rate)
            * cosine_weight
        )

    @staticmethod
    def _set_trainer_learning_rate(trainer, learning_rate: float) -> None:
        trainer.learning_rate = float(learning_rate)
        for group in trainer.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def _apply_scheduled_learning_rate(self) -> None:
        learning_rate = self.scheduled_learning_rate()
        self.current_regret_policy_learning_rate = learning_rate
        self._set_trainer_learning_rate(self.ave_policy_trainer, learning_rate)
        for trainer in self.regret_trainers:
            self._set_trainer_learning_rate(trainer, learning_rate)

    def iteration(self):
        self._apply_scheduled_learning_rate()
        return super().iteration()

    def evaluate(self, **kwargs):
        self.logger.record(
            "regret_policy_learning_rate",
            self.current_regret_policy_learning_rate,
        )
        self.logger.record(
            "regret_policy_gradient_clip_norm",
            self.regret_policy_gradient_clip_norm,
        )
        return super().evaluate(**kwargs)


__all__ = [
    "GradientClippedAvePolicyTrainer",
    "GradientClippedVRDCFRPlusRegretTrainer",
    "StableUnbiasedControlVariateEscher",
]
