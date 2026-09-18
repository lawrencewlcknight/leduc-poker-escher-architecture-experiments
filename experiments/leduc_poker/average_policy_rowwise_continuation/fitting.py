"""Streaming row-sampled fitting without information-set grouping."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .config import EQUAL_EXAMPLE_ARM, EQUAL_UPDATE_ARM


def _row_loss(model, source, indices: np.ndarray) -> torch.Tensor:
    features = torch.as_tensor(source.infostates[indices], dtype=torch.float32)
    targets = torch.as_tensor(source.policies[indices], dtype=torch.float32)
    masks = torch.as_tensor(
        source.legal_masks[indices], dtype=torch.bool
    )
    iterations = torch.as_tensor(
        source.iterations[indices], dtype=torch.float32
    )
    logits = model(features).masked_fill(~masks, -1e20)
    per_row = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=1)
    scale = torch.clamp(iterations, min=0.0) / float(source.iteration) * 2.0
    weights = torch.pow(scale, float(source.gamma))
    return torch.mean(per_row * weights)


def diagnostic_row_objective(model, source, *, maximum_rows: int = 8192) -> float:
    """Evaluate a deterministic row sample without constructing grouped data."""

    size = int(len(source.infostates))
    count = min(size, int(maximum_rows))
    if count <= 0:
        raise ValueError("Cannot evaluate an empty replay reservoir")
    indices = np.linspace(0, size - 1, num=count, dtype=np.int64)
    model.eval()
    weighted_total = 0.0
    rows = 0
    with torch.no_grad():
        for start in range(0, count, 2048):
            batch = indices[start:start + 2048]
            loss = _row_loss(model, source, batch)
            weighted_total += float(loss) * len(batch)
            rows += len(batch)
    return weighted_total / float(rows)


def fit_rowwise_continuously(
    *,
    model: torch.nn.Module,
    source,
    arm_id: str,
    unique_information_states: int,
    batch_size: int,
    learning_rate: float,
    gradient_clip_norm: float,
    diagnostic_equivalent_updates: Sequence[int],
    sampling_seed: int,
    callback,
) -> None:
    """Fit one arm continuously and hit every frozen resource boundary exactly."""

    if arm_id not in {EQUAL_EXAMPLE_ARM, EQUAL_UPDATE_ARM}:
        raise ValueError(f"Unknown row-wise arm {arm_id}")
    updates = tuple(int(value) for value in diagnostic_equivalent_updates)
    if not updates or updates[0] != 0 or tuple(sorted(set(updates))) != updates:
        raise ValueError("Diagnostic updates must be sorted, unique and start at zero")
    if int(unique_information_states) <= 0 or int(batch_size) <= 0:
        raise ValueError("Training sizes must be positive")
    if len(source.infostates) <= 0 or int(source.iteration) <= 0:
        raise ValueError("Source replay and iteration must be non-empty")

    rng = np.random.default_rng(int(sampling_seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    optimizer_steps = 0
    examples_seen = 0
    callback(0, examples_seen, optimizer_steps, float("nan"), model)

    for progress in updates[1:]:
        if arm_id == EQUAL_EXAMPLE_ARM:
            target_examples = int(progress) * int(unique_information_states)
            while examples_seen < target_examples:
                current_batch = min(int(batch_size), target_examples - examples_seen)
                indices = rng.integers(
                    0, len(source.infostates), size=current_batch, dtype=np.int64
                )
                model.train()
                loss = _row_loss(model, source, indices)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip_norm)
                )
                optimizer.step()
                examples_seen += current_batch
                optimizer_steps += 1
        else:
            target_steps = int(progress)
            while optimizer_steps < target_steps:
                indices = rng.integers(
                    0, len(source.infostates), size=int(batch_size), dtype=np.int64
                )
                model.train()
                loss = _row_loss(model, source, indices)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(gradient_clip_norm)
                )
                optimizer.step()
                examples_seen += int(batch_size)
                optimizer_steps += 1
        callback(
            progress,
            examples_seen,
            optimizer_steps,
            float(loss.detach()),
            model,
        )


__all__ = [
    "diagnostic_row_objective",
    "fit_rowwise_continuously",
]
