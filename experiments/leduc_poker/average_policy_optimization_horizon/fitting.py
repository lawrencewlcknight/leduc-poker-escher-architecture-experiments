"""Replay-only grouped targets and continued full-batch policy fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class GroupedReplayDataset:
    """One iteration-weighted soft target per observed information state."""

    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    replay_rows: int

    def __post_init__(self) -> None:
        size = len(self.features)
        if size <= 0:
            raise ValueError("Grouped replay dataset is empty")
        if len(self.targets) != size or len(self.masks) != size:
            raise ValueError("Grouped replay arrays have different lengths")
        if not np.all(np.isfinite(self.targets)):
            raise ValueError("Grouped replay targets must be finite")
        if np.any(self.targets < 0.0):
            raise ValueError("Grouped replay targets must be non-negative")


def grouped_replay_dataset(source) -> GroupedReplayDataset:
    """Collapse replay rows exactly as Experiment 40's uniform repair target."""

    unique, inverse, counts = np.unique(
        source.infostates, axis=0, return_inverse=True, return_counts=True
    )
    raw_weights = np.power(
        np.maximum(source.iterations, 0.0), source.gamma, dtype=np.float64
    )
    numerators = np.zeros((len(unique), source.policies.shape[1]), dtype=np.float64)
    masses = np.zeros(len(unique), dtype=np.float64)
    np.add.at(numerators, inverse, source.policies * raw_weights[:, None])
    np.add.at(masses, inverse, raw_weights)
    if np.any(masses <= 0.0):
        raise ValueError("An observed information state has zero iteration mass")
    targets = numerators / masses[:, None]

    order = np.argsort(inverse, kind="stable")
    first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    first_rows = order[first]
    masks = np.asarray(source.legal_masks[first_rows], dtype=np.float32)
    for row_index, group_index in enumerate(inverse):
        if not np.array_equal(source.legal_masks[row_index], masks[group_index]):
            raise ValueError("Legal-action mask differs within an information state")
    legal_mass = np.sum(targets * masks, axis=1)
    if not np.allclose(legal_mass, 1.0, atol=1e-5):
        raise ValueError("Grouped targets do not sum to one over legal actions")
    return GroupedReplayDataset(
        features=unique.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        masks=masks,
        replay_rows=int(len(source.infostates)),
    )


def full_batch_cross_entropy(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    logits = model(features).masked_fill(~masks, -1e20)
    per_information_state = -(
        targets * torch.log_softmax(logits, dim=-1)
    ).sum(dim=1)
    return torch.mean(per_information_state)


def fit_continuously(
    *,
    model: torch.nn.Module,
    dataset: GroupedReplayDataset,
    learning_rate: float,
    gradient_clip_norm: float,
    evaluation_updates: Sequence[int],
    callback,
) -> None:
    """Fit one trajectory continuously and invoke ``callback`` at fixed updates."""

    updates = tuple(sorted({int(value) for value in evaluation_updates}))
    if not updates or updates[0] != 0 or updates[-1] <= 0:
        raise ValueError("Evaluation updates must start at zero and include training")
    features = torch.from_numpy(dataset.features)
    targets = torch.from_numpy(dataset.targets)
    masks = torch.from_numpy(dataset.masks).to(dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    callback(0, float("nan"), model)
    for update in range(1, updates[-1] + 1):
        model.train()
        loss = full_batch_cross_entropy(model, features, targets, masks)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(gradient_clip_norm)
        )
        optimizer.step()
        if update in updates:
            callback(update, float(loss.detach()), model)


def objective_value(model: torch.nn.Module, dataset: GroupedReplayDataset) -> float:
    model.eval()
    with torch.no_grad():
        return float(
            full_batch_cross_entropy(
                model,
                torch.from_numpy(dataset.features),
                torch.from_numpy(dataset.targets),
                torch.from_numpy(dataset.masks).to(dtype=torch.bool),
            )
        )


__all__ = [
    "GroupedReplayDataset",
    "fit_continuously",
    "full_batch_cross_entropy",
    "grouped_replay_dataset",
    "objective_value",
]
