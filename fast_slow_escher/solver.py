"""Two-timescale, cross-fitted control critics for unbiased neural CFR."""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from unbiased_escher.solver import (
    CrossFittedQMember,
    UnbiasedControlVariateEscher,
)
from vr_deep_cfr.solver import MLP


class ReservoirTransitionBuffer:
    """Uniform lifetime reservoir with the Q trainer's transition interface."""

    def __init__(
        self,
        capacity: int,
        history_size: int,
        state_size: int,
        action_size: int,
        device: str,
    ):
        if capacity <= 0:
            raise ValueError("Reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.history_size = int(history_size)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = str(device)
        self.history = np.zeros((capacity, history_size), dtype=np.float32)
        self.next_history = np.zeros((capacity, history_size), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_size), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.legal_mask = np.zeros((capacity, action_size), dtype=np.int64)
        self.next_player = np.zeros(capacity, dtype=np.int64)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.done = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.seen_count = 0

    def add(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ) -> None:
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            candidate = random.randrange(self.seen_count + 1)
            if candidate >= self.capacity:
                self.seen_count += 1
                return
            index = candidate
        self.history[index] = history
        self.action[index] = action
        self.next_history[index] = next_history
        self.next_state[index] = next_state
        self.legal_mask[index] = next_legal_actions_mask
        self.next_player[index] = next_player
        self.done[index] = done
        self.reward[index] = reward
        self.seen_count += 1

    def sample(self, count: int = -1):
        if count == -1 or count > self.size:
            indices = list(range(self.size))
        else:
            indices = random.sample(range(self.size), int(count))
        floats = (
            self.history[indices],
            self.next_history[indices],
            self.next_state[indices],
            self.reward[indices],
        )
        integers = (
            self.legal_mask[indices],
            self.next_player[indices],
            self.action[indices],
            self.done[indices],
        )
        return (
            *(
                torch.as_tensor(value, dtype=torch.float32, device=self.device)
                for value in floats
            ),
            *(
                torch.as_tensor(value, dtype=torch.int64, device=self.device)
                for value in integers
            ),
        )

    def __len__(self) -> int:
        return self.size


class RhoReplayBuffer:
    """Recent out-of-fold critic predictions and sampled returns."""

    def __init__(self, capacity: int, feature_size: int):
        if capacity <= 0:
            raise ValueError("Rho replay capacity must be positive")
        self.capacity = int(capacity)
        self.features = np.zeros((capacity, feature_size), dtype=np.float32)
        self.fast_values = np.zeros(capacity, dtype=np.float32)
        self.slow_values = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.cursor = 0
        self.size = 0

    def add(self, features, fast: float, slow: float, sampled_return: float) -> None:
        self.features[self.cursor] = np.asarray(features, dtype=np.float32)
        self.fast_values[self.cursor] = float(fast)
        self.slow_values[self.cursor] = float(slow)
        self.returns[self.cursor] = float(sampled_return)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, device: str):
        count = min(int(count), self.size)
        indices = random.sample(range(self.size), count)
        return tuple(
            torch.as_tensor(value[indices], dtype=torch.float32, device=device)
            for value in (
                self.features,
                self.fast_values,
                self.slow_values,
                self.returns,
            )
        )

    def __len__(self) -> int:
        return self.size


class HeldOutRhoController:
    """Frozen controller fitted only to out-of-fold critic residuals."""

    def __init__(
        self,
        *,
        infostate_size: int,
        action_size: int,
        hidden_layers,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        device: str,
        gradient_clip_norm: float,
    ):
        self.infostate_size = int(infostate_size)
        self.action_size = int(action_size)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.train_steps = int(train_steps)
        self.gradient_clip_norm = float(gradient_clip_norm)
        # state, action one-hot, log iteration, player, fast/slow predictions,
        # their two held-out disagreements, and their absolute gap.
        self.feature_size = self.infostate_size + self.action_size + 7
        self.model = MLP(self.feature_size, list(hidden_layers), 1).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
        )
        self.loss_fn = nn.MSELoss()
        self.buffer = RhoReplayBuffer(buffer_size, self.feature_size)
        self.target_version = 0

    def feature(
        self,
        infostate,
        action: int,
        iteration: int,
        player: int,
        fast: float,
        slow: float,
        fast_disagreement: float,
        slow_disagreement: float,
    ) -> np.ndarray:
        action_one_hot = np.zeros(self.action_size, dtype=np.float32)
        action_one_hot[int(action)] = 1.0
        scalars = np.asarray(
            [
                math.log1p(max(int(iteration), 0)) / math.log(101.0),
                float(player),
                float(fast),
                float(slow),
                math.log1p(max(float(fast_disagreement), 0.0)),
                math.log1p(max(float(slow_disagreement), 0.0)),
                abs(float(fast) - float(slow)),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [np.asarray(infostate, dtype=np.float32), action_one_hot, scalars]
        )

    def predict_all(
        self,
        infostate,
        iteration: int,
        player: int,
        fast_values,
        slow_values,
        fast_disagreement,
        slow_disagreement,
    ):
        features = np.stack(
            [
                self.feature(
                    infostate,
                    action,
                    iteration,
                    player,
                    fast_values[action],
                    slow_values[action],
                    fast_disagreement[action],
                    slow_disagreement[action],
                )
                for action in range(self.action_size)
            ]
        )
        with torch.no_grad():
            logits = self.target_model(
                torch.as_tensor(features, dtype=torch.float32, device=self.device)
            ).squeeze(1)
            rho = torch.sigmoid(logits)
        return rho.cpu().numpy().astype(np.float64), features

    def add(
        self,
        features,
        fast: float,
        slow: float,
        sampled_return: float,
    ) -> None:
        self.buffer.add(features, fast, slow, sampled_return)

    def train_model(self):
        if len(self.buffer) < max(1, self.batch_size) or self.train_steps <= 0:
            return None
        self.model.train()
        final_loss = None
        for _ in range(self.train_steps):
            features, fast, slow, sampled_returns = self.buffer.sample(
                self.batch_size,
                self.device,
            )
            rho = torch.sigmoid(self.model(features).squeeze(1))
            mixture = rho * fast + (1.0 - rho) * slow
            loss = self.loss_fn(mixture, sampled_returns)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )
            self.optimizer.step()
            final_loss = float(loss.item())
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_version += 1
        return final_loss


class FastSlowCrossFittedQEnsemble:
    """Paired recent/lifetime critics with leave-one-trajectory-fold-out use."""

    def __init__(
        self,
        *,
        ensemble_size: int,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers,
        learning_rate: float,
        slow_buffer_size: int,
        fast_buffer_size: int,
        batch_size: int,
        slow_train_steps: int,
        fast_train_steps: int,
        logger,
        regret_trainers,
        device: str,
        gradient_clip_norm: float,
        rho_buffer_size: int,
        rho_batch_size: int,
        rho_train_steps: int,
        rho_learning_rate: float,
    ):
        if ensemble_size < 2:
            raise ValueError("Fast/slow cross-fitting requires at least two folds")
        self.ensemble_size = int(ensemble_size)
        self.action_size = int(action_size)
        self.active_fold = 0
        self.current_iteration = 0
        slow_member_capacity = max(1, int(slow_buffer_size) // ensemble_size)
        fast_member_capacity = max(1, int(fast_buffer_size) // ensemble_size)
        self.slow_members = [
            CrossFittedQMember(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                slow_member_capacity,
                batch_size,
                slow_train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
            )
            for _ in range(ensemble_size)
        ]
        for member in self.slow_members:
            member.buffer = ReservoirTransitionBuffer(
                slow_member_capacity,
                history_size,
                state_size,
                action_size,
                device,
            )
        self.fast_members = [
            CrossFittedQMember(
                history_size,
                state_size,
                action_size,
                network_layers,
                learning_rate,
                fast_member_capacity,
                batch_size,
                fast_train_steps,
                logger,
                regret_trainers,
                device,
                gradient_clip_norm=gradient_clip_norm,
            )
            for _ in range(ensemble_size)
        ]
        # Compatibility with the Experiment 6 diagnostics: these are the
        # long-horizon folds, with explicit fast-fold fields logged separately.
        self.members = self.slow_members
        self.rho_controller = HeldOutRhoController(
            infostate_size=state_size,
            action_size=action_size,
            hidden_layers=network_layers,
            learning_rate=rho_learning_rate,
            buffer_size=rho_buffer_size,
            batch_size=rho_batch_size,
            train_steps=rho_train_steps,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        self.last_fast_loss = None
        self.last_slow_loss = None
        self.last_rho_loss = None
        self._pending_decisions = {}
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.stats: Dict[str, float] = {
            "decision_count": 0.0,
            "rho_sum": 0.0,
            "rho_min": float("inf"),
            "rho_max": float("-inf"),
            "fast_disagreement_sum": 0.0,
            "slow_disagreement_sum": 0.0,
            "fast_slow_gap_sum": 0.0,
            "return_count": 0.0,
            "fast_squared_error_sum": 0.0,
            "slow_squared_error_sum": 0.0,
            "mixture_squared_error_sum": 0.0,
        }

    def begin_iteration(self, iteration: int) -> None:
        self.current_iteration = int(iteration)
        self._reset_diagnostics()
        # The fast model persists, but its replay contains only transitions
        # collected in this outer iteration. Its target remains last iteration's
        # frozen snapshot during collection.
        for member in self.fast_members:
            member.buffer.cur_id = 0
            member.buffer.size = 0

    def begin_trajectory(self, trajectory_id: int) -> int:
        self.active_fold = int(trajectory_id) % self.ensemble_size
        self._pending_decisions.clear()
        return self.active_fold

    def heldout_member_indices(self) -> List[int]:
        return [
            index for index in range(self.ensemble_size) if index != self.active_fold
        ]

    def _component_predictions(self, state, player: int):
        heldout = self.heldout_member_indices()
        fast_predictions = np.stack(
            [self.fast_members[index].get_baseline(state, player) for index in heldout]
        )
        slow_predictions = np.stack(
            [self.slow_members[index].get_baseline(state, player) for index in heldout]
        )
        return (
            np.mean(fast_predictions, axis=0),
            np.mean(slow_predictions, axis=0),
            np.var(fast_predictions, axis=0),
            np.var(slow_predictions, axis=0),
        )

    def _decision(self, state, player: int, iteration: int, *, record: bool):
        fast, slow, fast_var, slow_var = self._component_predictions(state, player)
        rho, features = self.rho_controller.predict_all(
            state.information_state_tensor(player),
            iteration,
            player,
            fast,
            slow,
            fast_var,
            slow_var,
        )
        legal = np.asarray(state.legal_actions_mask(), dtype=bool)
        mixture = rho * fast + (1.0 - rho) * slow
        disagreement = (
            rho * fast_var
            + (1.0 - rho) * slow_var
            + rho * (1.0 - rho) * np.square(fast - slow)
        )
        if record:
            legal_rho = rho[legal]
            count = float(legal_rho.size)
            self.stats["decision_count"] += count
            self.stats["rho_sum"] += float(np.sum(legal_rho))
            self.stats["rho_min"] = min(
                self.stats["rho_min"],
                float(np.min(legal_rho)),
            )
            self.stats["rho_max"] = max(
                self.stats["rho_max"],
                float(np.max(legal_rho)),
            )
            self.stats["fast_disagreement_sum"] += float(np.sum(fast_var[legal]))
            self.stats["slow_disagreement_sum"] += float(np.sum(slow_var[legal]))
            self.stats["fast_slow_gap_sum"] += float(
                np.sum(np.abs(fast[legal] - slow[legal]))
            )
        return mixture, disagreement, fast, slow, rho, features

    def get_baseline_and_disagreement(self, state, player: int):
        decision = self._decision(
            state,
            player,
            self.current_iteration,
            record=True,
        )
        self._pending_decisions[(id(state), int(player))] = decision
        mixture, disagreement, *_ = decision
        return mixture, disagreement

    def get_baseline(self, state, player: int) -> np.ndarray:
        values, _ = self.get_baseline_and_disagreement(state, player)
        return values

    def observe_control_return(
        self,
        *,
        state,
        player: int,
        action: int,
        sampled_return: float,
        iteration: int,
    ) -> None:
        key = (id(state), int(player))
        decision = self._pending_decisions.pop(key, None)
        if decision is None:
            decision = self._decision(
                state,
                player,
                iteration,
                record=False,
            )
        mixture, _, fast, slow, _, features = decision
        self.rho_controller.add(
            features[action],
            fast[action],
            slow[action],
            sampled_return,
        )
        self.stats["return_count"] += 1.0
        self.stats["fast_squared_error_sum"] += float(
            np.square(sampled_return - fast[action])
        )
        self.stats["slow_squared_error_sum"] += float(
            np.square(sampled_return - slow[action])
        )
        self.stats["mixture_squared_error_sum"] += float(
            np.square(sampled_return - mixture[action])
        )

    def add_data(self, *args) -> None:
        self.fast_members[self.active_fold].add_data(*args)
        self.slow_members[self.active_fold].add_data(*args)

    @staticmethod
    def _mean_finite(losses):
        finite = [float(loss) for loss in losses if loss is not None]
        return float(np.mean(finite)) if finite else None

    def train_model(self, iteration: int):
        self.last_fast_loss = self._mean_finite(
            member.train_model(iteration) for member in self.fast_members
        )
        self.last_slow_loss = self._mean_finite(
            member.train_model(iteration) for member in self.slow_members
        )
        self.last_rho_loss = self.rho_controller.train_model()
        return self._mean_finite([self.last_fast_loss, self.last_slow_loss])

    def fold_sizes(self) -> List[int]:
        return [len(member.buffer) for member in self.slow_members]

    def fast_fold_sizes(self) -> List[int]:
        return [len(member.buffer) for member in self.fast_members]

    def diagnostics(self) -> Dict[str, float]:
        decision_count = max(self.stats["decision_count"], 1.0)
        return_count = max(self.stats["return_count"], 1.0)
        return {
            "rho_controller_loss": self.last_rho_loss,
            "rho_controller_target_version": self.rho_controller.target_version,
            "rho_controller_replay_size": len(self.rho_controller.buffer),
            "fast_slow_rho_mean": self.stats["rho_sum"] / decision_count,
            "fast_slow_rho_min": (
                self.stats["rho_min"]
                if self.stats["decision_count"]
                else np.nan
            ),
            "fast_slow_rho_max": (
                self.stats["rho_max"]
                if self.stats["decision_count"]
                else np.nan
            ),
            "fast_critic_disagreement_mean": (
                self.stats["fast_disagreement_sum"] / decision_count
            ),
            "slow_critic_disagreement_mean": (
                self.stats["slow_disagreement_sum"] / decision_count
            ),
            "fast_slow_prediction_gap_mean": (
                self.stats["fast_slow_gap_sum"] / decision_count
            ),
            "fast_critic_sampled_mse": (
                self.stats["fast_squared_error_sum"] / return_count
            ),
            "slow_critic_sampled_mse": (
                self.stats["slow_squared_error_sum"] / return_count
            ),
            "mixture_critic_sampled_mse": (
                self.stats["mixture_squared_error_sum"] / return_count
            ),
            "fast_critic_loss": self.last_fast_loss,
            "slow_critic_loss": self.last_slow_loss,
            "fast_critic_target_version_min": min(
                member.target_version for member in self.fast_members
            ),
            "fast_critic_target_version_max": max(
                member.target_version for member in self.fast_members
            ),
            "slow_critic_target_version_min": min(
                member.target_version for member in self.slow_members
            ),
            "slow_critic_target_version_max": max(
                member.target_version for member in self.slow_members
            ),
            "slow_critic_lifetime_seen_count": sum(
                member.buffer.seen_count for member in self.slow_members
            ),
        }


class FastSlowControlCriticEscher(UnbiasedControlVariateEscher):
    """Experiment 6 with predictable fast/slow cross-fitted Q controls."""

    def __init__(
        self,
        *args,
        fast_q_buffer_size: int = 250_000,
        fast_q_train_steps: int = 5_000,
        rho_buffer_size: int = 250_000,
        rho_batch_size: int = 2_048,
        rho_train_steps: int = 2_000,
        rho_learning_rate: float = 1e-3,
        **kwargs,
    ):
        self.fast_q_buffer_size = int(fast_q_buffer_size)
        self.fast_q_train_steps = int(fast_q_train_steps)
        self.rho_buffer_size = int(rho_buffer_size)
        self.rho_batch_size = int(rho_batch_size)
        self.rho_train_steps = int(rho_train_steps)
        self.rho_learning_rate = float(rho_learning_rate)
        super().__init__(*args, **kwargs)

    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_size = len(
            np.append(
                root_state.information_state_tensor(0),
                root_state.information_state_tensor(1),
            )
        )
        self.q_value_trainer = FastSlowCrossFittedQEnsemble(
            ensemble_size=self.q_ensemble_size,
            history_size=history_size,
            state_size=self.infostate_size,
            action_size=self.action_size,
            network_layers=self.network_layers,
            learning_rate=self.learning_rate,
            slow_buffer_size=self.baseline_buffer_size,
            fast_buffer_size=self.fast_q_buffer_size,
            batch_size=self.baseline_batch_size,
            slow_train_steps=self.baseline_network_train_steps,
            fast_train_steps=self.fast_q_train_steps,
            logger=self.logger,
            regret_trainers=self.regret_trainers,
            device=self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
            rho_buffer_size=self.rho_buffer_size,
            rho_batch_size=self.rho_batch_size,
            rho_train_steps=self.rho_train_steps,
            rho_learning_rate=self.rho_learning_rate,
        )

    def iteration(self):
        self.q_value_trainer.begin_iteration(self.num_iteration + 1)
        return super().iteration()

    def evaluate(self, **kwargs):
        diagnostics = self.q_value_trainer.diagnostics()
        for key, value in diagnostics.items():
            self.logger.record(key, value)
        for fold, size in enumerate(self.q_value_trainer.fast_fold_sizes()):
            self.logger.record(f"fast_q_fold_{fold}_replay_size", size)
        return super().evaluate(**kwargs)
