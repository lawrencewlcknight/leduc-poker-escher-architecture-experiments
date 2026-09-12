"""Average-policy distillation variants promoted by the offline studies."""

from __future__ import annotations

import torch

from vr_deep_cfr.solver import AvePolicyTrainer

from .factorial import FactorialUnbiasedControlVariateEscher


SOFT_TARGET_CROSS_ENTROPY = "soft_target_cross_entropy"


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
    "SOFT_TARGET_CROSS_ENTROPY",
    "SoftTargetCrossEntropyAvePolicyTrainer",
]
