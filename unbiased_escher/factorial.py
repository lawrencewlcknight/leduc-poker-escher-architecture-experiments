"""Opt-in network and target interventions for the UCV stability factorial.

The defaults are deliberately equivalent to the selected Experiment 23 solver:
the cumulative-regret approximators use the original MLP and each critic target
is the most recently completed online fit.  The two interventions can therefore
be enabled independently without changing the estimator or sampling rule.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
import time
from typing import Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vr_deep_cfr.solver import SonnetLinear, ZeroInitLinear
from vr_deep_cfr.variants import VRDCFRPlusRegretTrainer

from .solver import CrossFittedQEnsemble, CrossFittedQMember
from .stability import StableUnbiasedControlVariateEscher


class PreNormResidualBlock(nn.Module):
    """Two-layer, same-width pre-LayerNorm residual block."""

    def __init__(self, width: int):
        super().__init__()
        self.width = int(width)
        if self.width <= 0:
            raise ValueError("Residual width must be positive")
        self.norm_1 = nn.LayerNorm(self.width)
        self.linear_1 = SonnetLinear(self.width, self.width, activation=False)
        self.norm_2 = nn.LayerNorm(self.width)
        self.linear_2 = SonnetLinear(self.width, self.width, activation=False)

    def reset_parameters(self) -> None:
        self.norm_1.reset_parameters()
        self.linear_1.reset_parameters()
        self.norm_2.reset_parameters()
        self.linear_2.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.linear_1(self.norm_1(inputs)))
        hidden = self.linear_2(self.norm_2(hidden))
        return F.relu(inputs + hidden)


class ResidualLayerNormRegretMLP(nn.Module):
    """Regret-only residual network with a zero-output initial policy head."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        width: int = 64,
        blocks: int = 4,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.width = int(width)
        self.block_count = int(blocks)
        if self.block_count <= 0:
            raise ValueError("Residual block count must be positive")
        self.input_projection = SonnetLinear(
            self.input_size, self.width, activation=True
        )
        self.blocks = nn.ModuleList(
            [PreNormResidualBlock(self.width) for _ in range(self.block_count)]
        )
        self.output_layer = ZeroInitLinear(
            self.width, self.output_size, activation=False
        )

    def reset_parameters(self) -> None:
        self.input_projection.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.output_layer.reset_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_layer(hidden)


class ResidualLayerNormVRDCFRPlusRegretTrainer(VRDCFRPlusRegretTrainer):
    """VR-DCFR+ trainer changing only the cumulative-regret model class."""

    def __init__(
        self,
        *args,
        residual_width: int,
        residual_blocks: int,
        **kwargs,
    ):
        self.residual_width = int(residual_width)
        self.residual_blocks = int(residual_blocks)
        super().__init__(*args, **kwargs)

    def init_model(self):
        return ResidualLayerNormRegretMLP(
            self.input_size,
            self.output_size,
            width=self.residual_width,
            blocks=self.residual_blocks,
        ).to(self.device)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


class TemporallyAveragedCrossFittedQMember(CrossFittedQMember):
    """Critic whose frozen target is the mean of recent completed online fits."""

    def __init__(self, *args, target_average_window: int, **kwargs):
        self.target_average_window = int(target_average_window)
        if self.target_average_window <= 0:
            raise ValueError("target_average_window must be positive")
        self.target_history: deque[dict[str, torch.Tensor]] = deque(
            maxlen=self.target_average_window
        )
        self.last_target_update_l2 = 0.0
        super().__init__(*args, **kwargs)

    def _install_temporal_average(
        self, previous: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        if previous is None:
            previous = _cpu_state_dict(self.target_model)
        self.target_history.append(_cpu_state_dict(self.model))
        averaged = {}
        for name in self.target_history[0]:
            tensors = [snapshot[name] for snapshot in self.target_history]
            if torch.is_floating_point(tensors[0]):
                averaged[name] = torch.stack(tensors, dim=0).mean(dim=0)
            else:
                averaged[name] = tensors[-1]
        self.target_model.load_state_dict(averaged)
        self.target_model.eval()
        squared = 0.0
        for name, value in averaged.items():
            if torch.is_floating_point(value):
                difference = value - previous[name]
                squared += float(torch.sum(difference * difference).item())
        self.last_target_update_l2 = math.sqrt(squared)

    def train_model(self, iteration: int):
        previous = _cpu_state_dict(self.target_model)
        loss = super().train_model(iteration)
        if loss is not None:
            # ``super`` first completes the online fit and hard-copies it.  The
            # copy is then replaced atomically by the requested history mean.
            self._install_temporal_average(previous)
        return loss


class TemporallyAveragedCrossFittedQEnsemble(CrossFittedQEnsemble):
    """Cross-fitted ensemble using temporal averaging within every fold."""

    def __init__(
        self,
        *,
        target_average_window: int,
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
        if int(ensemble_size) < 1:
            raise ValueError("At least one critic is required")
        self.ensemble_size = int(ensemble_size)
        self.target_average_window = int(target_average_window)
        member_buffer_size = max(1, int(total_buffer_size) // self.ensemble_size)
        self.members = [
            TemporallyAveragedCrossFittedQMember(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                member_buffer_size,
                batch_size,
                train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
                target_average_window=self.target_average_window,
            )
            for _ in range(self.ensemble_size)
        ]
        self.active_fold = 0
        self._active_add_count = 0


class FactorialUnbiasedControlVariateEscher(StableUnbiasedControlVariateEscher):
    """Selected UCV core with two independently switchable interventions."""

    REGRET_NETWORK_TYPES = {"mlp", "residual_layer_norm"}

    def __init__(
        self,
        *args,
        regret_network_type: str = "mlp",
        regret_residual_width: int = 64,
        regret_residual_blocks: int = 4,
        critic_target_average_window: int = 1,
        **kwargs,
    ):
        self.regret_network_type = str(regret_network_type)
        self.regret_residual_width = int(regret_residual_width)
        self.regret_residual_blocks = int(regret_residual_blocks)
        self.critic_target_average_window = int(critic_target_average_window)
        if self.regret_network_type not in self.REGRET_NETWORK_TYPES:
            raise ValueError(
                f"Unknown regret network type {self.regret_network_type!r}"
            )
        if self.critic_target_average_window <= 0:
            raise ValueError("critic_target_average_window must be positive")
        self._resume_elapsed_seconds = 0.0
        self._checkpoint_resume_rng_state = None
        super().__init__(*args, **kwargs)
        if self.use_instantaneous_predictor:
            raise ValueError("The factorial requires the selected non-predictive core")

    def init_regret_trainers(self):
        if self.regret_network_type == "mlp":
            return super().init_regret_trainers()
        if self.use_instantaneous_predictor:
            raise ValueError("Residual regret networks require non-predictive UCV")
        self.regret_trainers = [
            ResidualLayerNormVRDCFRPlusRegretTrainer(
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
                residual_width=self.regret_residual_width,
                residual_blocks=self.regret_residual_blocks,
            )
            for _ in range(self.num_players)
        ]
        for trainer in self.regret_trainers:
            trainer.predictor_enabled = False

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = TemporallyAveragedCrossFittedQEnsemble(
            target_average_window=self.critic_target_average_window,
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

    def _run_checkpoint(
        self,
        *,
        checkpoint_kind="outer_iteration",
        checkpoint_target_nodes=None,
    ):
        """Expose the pre-evaluation RNG state for exact continuation saves."""
        rng_state = self._capture_rng_state() if self.preserve_evaluation_rng else None
        try:
            self.train_average_policy()
            self.evaluate(
                checkpoint_kind=checkpoint_kind,
                checkpoint_target_nodes=checkpoint_target_nodes,
            )
            self._checkpoint_resume_rng_state = deepcopy(rng_state)
            callback = getattr(self, "_post_checkpoint_callback", None)
            if callback is not None:
                callback(self, dict(self.checkpoint_rows[-1]))
        finally:
            if rng_state is not None:
                self._restore_rng_state(rng_state)

    def solve(self, post_checkpoint_callback=None):
        """Continue from a restored active-time offset when one is present."""
        self._post_checkpoint_callback = post_checkpoint_callback
        self._solve_start_time = time.perf_counter() - float(
            self._resume_elapsed_seconds
        )
        self._prepare_early_evaluation_schedule()
        if self.evaluate_initial_policy and self.num_iteration == 0:
            self.evaluate(checkpoint_kind="initial_untrained_policy")
        iteration_cap = self.num_iterations
        if self.max_num_iterations is not None:
            iteration_cap = min(iteration_cap, int(self.max_num_iterations))
        while self.num_iteration < iteration_cap:
            self.iteration()
            if (
                self.target_nodes_touched is not None
                and self.nodes_touched >= int(self.target_nodes_touched)
            ):
                break
        if not self.checkpoint_rows or (
            self.checkpoint_rows[-1]["nodes_touched"] != self.nodes_touched
        ):
            self._run_checkpoint(checkpoint_kind="final_node_budget")
        return list(self.checkpoint_rows)


__all__ = [
    "FactorialUnbiasedControlVariateEscher",
    "PreNormResidualBlock",
    "ResidualLayerNormRegretMLP",
    "ResidualLayerNormVRDCFRPlusRegretTrainer",
    "TemporallyAveragedCrossFittedQEnsemble",
    "TemporallyAveragedCrossFittedQMember",
]
