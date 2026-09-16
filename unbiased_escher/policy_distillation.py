"""Average-policy distillation variants promoted by the offline studies."""

from __future__ import annotations

import random

import numpy as np
import torch

from vr_deep_cfr.solver import AvePolicyTrainer

from .factorial import FactorialUnbiasedControlVariateEscher


SOFT_TARGET_CROSS_ENTROPY = "soft_target_cross_entropy"
GROUPED_SOFT_TARGET_CROSS_ENTROPY = "grouped_soft_target_cross_entropy"


class SoftTargetCrossEntropyAvePolicyTrainer(AvePolicyTrainer):
    """Fit sampled strategy distributions with iteration-weighted cross-entropy."""

    def compute_loss(self, samples, T):
        infostates, target_policy, legal_actions_mask, iterations = samples
        logits = self.model(infostates)
        legal_logits = logits.masked_fill(legal_actions_mask != 1, -1e20)
        per_sample = -(
            target_policy * torch.log_softmax(legal_logits, dim=-1)
        ).sum(dim=1)
        weights = torch.pow(iterations.reshape(-1) / float(T) * 2.0, self.gamma)
        return torch.mean(per_sample * weights)


class GroupedSoftTargetCrossEntropyAvePolicyTrainer(
    SoftTargetCrossEntropyAvePolicyTrainer
):
    """Fit one sufficient-statistic target per observed information set.

    Repeated reservoir rows are collapsed to their iteration-weighted mean
    target and total objective mass.  The resulting full-batch objective is
    algebraically identical to the original row-wise soft-target CE objective,
    while removing avoidable within-information-set sampling noise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Production trajectories evaluate the uniform initial policy before
        # the reservoir has any rows.  Keep the empirical-policy diagnostic
        # well-defined until the first grouped fit populates this lookup.
        self.grouped_target_lookup = {}
        self.grouped_num_rows = 0
        self.grouped_num_information_sets = 0
        self.grouped_reduction_ratio = 0.0

    def _grouped_training_data(self, T):
        size = min(int(self.buffer.cur_id), int(self.buffer.buffer_size))
        if size <= 0:
            raise ValueError("Cannot fit an average policy from an empty reservoir")
        features = np.asarray(self.buffer.infostate_buf[:size], dtype=np.float32)
        policies = np.asarray(self.buffer.q_value_buf[:size], dtype=np.float64)
        masks = np.asarray(self.buffer.q_value_mask_buf[:size], dtype=np.float32)
        iterations = np.asarray(
            self.buffer.iteration_buf[:size], dtype=np.float64
        ).reshape(-1)
        raw_weights = np.power(
            np.maximum(iterations, 0.0) / float(T) * 2.0,
            self.gamma,
            dtype=np.float64,
        )
        unique, inverse, counts = np.unique(
            features, axis=0, return_inverse=True, return_counts=True
        )
        masses = np.zeros(len(unique), dtype=np.float64)
        numerators = np.zeros((len(unique), policies.shape[1]), dtype=np.float64)
        np.add.at(masses, inverse, raw_weights)
        np.add.at(numerators, inverse, policies * raw_weights[:, None])
        if np.any(masses <= 0.0):
            raise ValueError("Grouped average-policy target has non-positive mass")
        targets = numerators / masses[:, None]

        order = np.argsort(inverse, kind="stable")
        first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
        grouped_masks = masks[order[first]]
        for row_index, group_index in enumerate(inverse):
            if not np.array_equal(masks[row_index], grouped_masks[group_index]):
                raise ValueError(
                    "Legal-action masks differ within an information set"
                )

        # mean(group_weight * CE) is exactly mean(row_weight * CE).
        objective_weights = masses * (float(len(unique)) / float(size))
        self.grouped_num_rows = int(size)
        self.grouped_num_information_sets = int(len(unique))
        self.grouped_reduction_ratio = float(len(unique)) / float(size)
        self.grouped_target_lookup = {
            np.asarray(feature, dtype=np.float32).tobytes(): np.asarray(
                targets[index], dtype=np.float64
            )
            for index, feature in enumerate(unique)
        }
        return tuple(
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in (unique, targets, grouped_masks, objective_weights)
        )

    def train_model(self, T):
        features, targets, masks, weights = self._grouped_training_data(T)
        size = len(features)
        loss = None
        for train_step in range(self.train_steps):
            if self.batch_size == -1 or self.batch_size >= size:
                indices = torch.arange(size, device=self.device)
            else:
                indices = torch.as_tensor(
                    random.sample(range(size), int(self.batch_size)),
                    dtype=torch.long,
                    device=self.device,
                )
            logits = self.model(features.index_select(0, indices))
            selected_masks = masks.index_select(0, indices)
            logits = logits.masked_fill(selected_masks != 1, -1e20)
            per_sample = -(
                targets.index_select(0, indices)
                * torch.log_softmax(logits, dim=-1)
            ).sum(dim=1)
            loss = torch.mean(per_sample * weights.index_select(0, indices))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    f"[{train_step}/{self.train_steps}] grouped policy loss: "
                    f"{loss.item()}"
                )
        if loss is None:  # pragma: no cover - configuration validation prevents this
            raise ValueError("Average-policy training requires at least one update")
        return loss.item()

    def grouped_action_probabilities(self, state, probs_as_dict=True):
        key = np.asarray(
            state.information_state_tensor(), dtype=np.float32
        ).tobytes()
        probabilities = self.grouped_target_lookup.get(key)
        if probabilities is None:
            probabilities = np.zeros(self.output_size, dtype=np.float64)
            legal = state.legal_actions()
            probabilities[legal] = 1.0 / float(len(legal))
        if probs_as_dict:
            return {
                action: float(probabilities[action])
                for action in state.legal_actions()
            }
        return probabilities.copy()


class CrossEntropyAveragePolicyFactorialUCV(
    FactorialUnbiasedControlVariateEscher
):
    """Factorial UCV solver with reset soft-target CE average-policy fits."""

    def __init__(
        self,
        *args,
        average_policy_loss: str = SOFT_TARGET_CROSS_ENTROPY,
        average_policy_reset_each_fit: bool = True,
        **kwargs,
    ):
        if str(average_policy_loss) != SOFT_TARGET_CROSS_ENTROPY:
            raise ValueError("Only soft-target cross-entropy is supported")
        if not bool(average_policy_reset_each_fit):
            raise ValueError("The promoted candidate requires reset policy fits")
        self.average_policy_loss = str(average_policy_loss)
        self.average_policy_reset_each_fit = True
        super().__init__(*args, **kwargs)

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = SoftTargetCrossEntropyAvePolicyTrainer(
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
        )


__all__ = [
    "CrossEntropyAveragePolicyFactorialUCV",
    "GROUPED_SOFT_TARGET_CROSS_ENTROPY",
    "GroupedSoftTargetCrossEntropyAvePolicyTrainer",
    "SOFT_TARGET_CROSS_ENTROPY",
    "SoftTargetCrossEntropyAvePolicyTrainer",
]
