"""Fitting primitives for Experiment 34.

The grouped target is an exact sufficient statistic for the row-wise
soft-target cross-entropy objective: repeated rows for an information set are
replaced by their iteration-weighted mean policy and total weight.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import itertools
import random
import re
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    SourceData,
    _policy_errors,
)
from experiments.leduc_poker.causal_rare_state_audit.audit import _infoset_catalog
from vr_deep_cfr.solver import MLP

from .config import (
    ARCHITECTURE_LAYERS,
    CURRENT_SHARED,
    GROUPED_TARGETS,
    PLAYER_ROUND_GROUPS,
    ROUTED_EXPERTS,
    ROW_TARGETS,
    WIDE_SHARED,
)


_ROUND_PATTERN = re.compile(r"\[Round\s+(\d+)\]")


def _tensor_key(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _round_from_information_set(information_set: str) -> int:
    match = _ROUND_PATTERN.search(str(information_set))
    if match is None:
        raise ValueError(f"Cannot determine betting round from {information_set!r}")
    round_index = int(match.group(1)) - 1
    if round_index not in (0, 1):
        raise ValueError(f"Unexpected Leduc betting round {round_index + 1}")
    return round_index


def group_id(player: int, information_set: str) -> int:
    value = int(player) * 2 + _round_from_information_set(information_set)
    if value < 0 or value >= PLAYER_ROUND_GROUPS:
        raise ValueError(f"Invalid player-by-round group {value}")
    return value


@dataclass(frozen=True)
class PolicyDataset:
    features: np.ndarray
    targets: np.ndarray
    masks: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    target_id: str

    def __post_init__(self) -> None:
        size = len(self.features)
        if size <= 0:
            raise ValueError("Policy dataset is empty")
        if any(
            len(values) != size
            for values in (self.targets, self.masks, self.weights, self.groups)
        ):
            raise ValueError("Policy dataset arrays differ in length")
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights <= 0.0):
            raise ValueError("Policy dataset weights must be finite and positive")


@dataclass(frozen=True)
class Catalog:
    rows: Mapping
    tensor_to_key: Mapping[bytes, tuple[int, str]]
    tensor_to_group: Mapping[bytes, int]


def build_catalog(game) -> Catalog:
    rows = _infoset_catalog(game)
    tensor_to_key: dict[bytes, tuple[int, str]] = {}
    tensor_to_group: dict[bytes, int] = {}
    for key, row in rows.items():
        tensor = _tensor_key(row["tensor"])
        previous = tensor_to_key.get(tensor)
        if previous is not None and previous != key:
            raise ValueError("Information-state tensor collision prevents deterministic routing")
        tensor_to_key[tensor] = key
        tensor_to_group[tensor] = group_id(key[0], key[1])
    return Catalog(rows=rows, tensor_to_key=tensor_to_key, tensor_to_group=tensor_to_group)


def _row_weights(source: SourceData) -> np.ndarray:
    return np.power(
        np.maximum(source.iterations, 0.0) / float(source.iteration) * 2.0,
        source.gamma,
        dtype=np.float64,
    )


def row_dataset(source: SourceData, catalog: Catalog) -> PolicyDataset:
    groups = np.asarray(
        [catalog.tensor_to_group[_tensor_key(row)] for row in source.infostates],
        dtype=np.int64,
    )
    return PolicyDataset(
        features=np.asarray(source.infostates, dtype=np.float32),
        targets=np.asarray(source.policies, dtype=np.float32),
        masks=np.asarray(source.legal_masks, dtype=np.float32),
        weights=_row_weights(source).astype(np.float32),
        groups=groups,
        target_id=ROW_TARGETS,
    )


def grouped_dataset(source: SourceData, catalog: Catalog) -> PolicyDataset:
    """Collapse replay rows without changing the full empirical CE objective."""
    unique, inverse, counts = np.unique(
        source.infostates, axis=0, return_inverse=True, return_counts=True
    )
    raw_weights = _row_weights(source)
    numerators = np.zeros((len(unique), source.policies.shape[1]), dtype=np.float64)
    masses = np.zeros(len(unique), dtype=np.float64)
    np.add.at(numerators, inverse, source.policies * raw_weights[:, None])
    np.add.at(masses, inverse, raw_weights)
    if np.any(masses <= 0.0):
        raise ValueError("Grouped target has zero iteration mass")
    targets = numerators / masses[:, None]
    first = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    order = np.argsort(inverse, kind="stable")
    first_rows = order[first]
    masks = source.legal_masks[first_rows]
    for row_index, group_index in enumerate(inverse):
        if not np.array_equal(source.legal_masks[row_index], masks[group_index]):
            raise ValueError("Legal-action mask differs within an information set")
    groups = np.asarray(
        [catalog.tensor_to_group[_tensor_key(row)] for row in unique], dtype=np.int64
    )
    # mean(group_weights * CE) is exactly mean(raw_weights * CE).
    objective_weights = masses * (float(len(unique)) / float(len(source.infostates)))
    return PolicyDataset(
        features=unique.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        masks=np.asarray(masks, dtype=np.float32),
        weights=objective_weights.astype(np.float32),
        groups=groups,
        target_id=GROUPED_TARGETS,
    )


def exact_teacher_dataset(source: SourceData, catalog: Catalog) -> PolicyDataset:
    features, targets, masks, weights, groups = [], [], [], [], []
    for key, row in catalog.rows.items():
        target = source.exact_table.get(key)
        mass = float(source.exact_denominators.get(key, 0.0))
        if target is None or mass <= 0.0:
            continue
        mask = np.zeros(source.policies.shape[1], dtype=np.float32)
        mask[list(row["legal_actions"])] = 1.0
        features.append(np.asarray(row["tensor"], dtype=np.float32))
        targets.append(np.asarray(target, dtype=np.float32))
        masks.append(mask)
        weights.append(mass)
        groups.append(group_id(key[0], key[1]))
    weights_array = np.asarray(weights, dtype=np.float64)
    weights_array /= max(float(np.mean(weights_array)), 1e-30)
    return PolicyDataset(
        features=np.asarray(features, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
        weights=weights_array.astype(np.float32),
        groups=np.asarray(groups, dtype=np.int64),
        target_id="exact_tabular_teacher",
    )


class RoutedPolicyNetwork(nn.Module):
    """Four deterministic experts: player 0/1 crossed with betting round 1/2."""

    def __init__(self, input_size: int, hidden_layers: Sequence[int], output_size: int):
        super().__init__()
        self.output_size = int(output_size)
        self.experts = nn.ModuleList(
            MLP(input_size, list(hidden_layers), output_size)
            for _ in range(PLAYER_ROUND_GROUPS)
        )

    def forward(self, features: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        outputs = features.new_zeros((len(features), self.output_size))
        for value, expert in enumerate(self.experts):
            indices = torch.nonzero(groups == value, as_tuple=False).reshape(-1)
            if len(indices):
                outputs = outputs.index_copy(0, indices, expert(features.index_select(0, indices)))
        return outputs


def build_model(
    architecture_id: str, input_size: int, output_size: int
) -> nn.Module:
    layers = list(ARCHITECTURE_LAYERS[architecture_id])
    if architecture_id in {CURRENT_SHARED, WIDE_SHARED}:
        return MLP(input_size, layers, output_size)
    if architecture_id == ROUTED_EXPERTS:
        return RoutedPolicyNetwork(input_size, layers, output_size)
    raise ValueError(f"Unknown architecture {architecture_id!r}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _logits(
    model: nn.Module,
    architecture_id: str,
    features: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    if architecture_id == ROUTED_EXPERTS:
        return model(features, groups)
    return model(features)


def _loss(
    model: nn.Module,
    architecture_id: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    logits = _logits(model, architecture_id, features, groups).masked_fill(
        masks != 1, -1e20
    )
    per_sample = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=1)
    return torch.mean(per_sample * weights)


def full_objective(
    model: nn.Module,
    architecture_id: str,
    dataset: PolicyDataset,
    *,
    chunk_size: int = 65_536,
) -> float:
    model.eval()
    numerator = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(dataset.features), chunk_size):
            stop = min(start + chunk_size, len(dataset.features))
            loss = _loss(
                model,
                architecture_id,
                torch.from_numpy(dataset.features[start:stop]),
                torch.from_numpy(dataset.targets[start:stop]),
                torch.from_numpy(dataset.masks[start:stop]),
                torch.from_numpy(dataset.weights[start:stop]),
                torch.from_numpy(dataset.groups[start:stop]),
            )
            size = stop - start
            numerator += float(loss) * size
            count += size
    return numerator / float(count)


def model_table(model: nn.Module, architecture_id: str, catalog: Catalog) -> dict:
    keys = list(catalog.rows)
    features = np.asarray([catalog.rows[key]["tensor"] for key in keys], dtype=np.float32)
    groups = np.asarray([group_id(key[0], key[1]) for key in keys], dtype=np.int64)
    masks = np.zeros((len(keys), 3), dtype=np.float32)
    for index, key in enumerate(keys):
        masks[index, list(catalog.rows[key]["legal_actions"])] = 1.0
    model.eval()
    with torch.no_grad():
        logits = _logits(
            model,
            architecture_id,
            torch.from_numpy(features),
            torch.from_numpy(groups),
        ).masked_fill(torch.from_numpy(masks) != 1, -1e20)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    return {key: probabilities[index] for index, key in enumerate(keys)}


def ensemble_table(tables: Sequence[Mapping]) -> dict:
    if not tables:
        raise ValueError("At least one policy table is required")
    keys = set(tables[0])
    if any(set(table) != keys for table in tables[1:]):
        raise ValueError("Ensemble policy tables contain different information sets")
    return {
        key: np.mean([np.asarray(table[key], dtype=np.float64) for table in tables], axis=0)
        for key in keys
    }


def _gradient_cosines(
    model: nn.Module,
    architecture_id: str,
    dataset: PolicyDataset,
) -> list[dict]:
    model.train()
    vectors = {}
    parameters = list(model.parameters())
    for value in range(PLAYER_ROUND_GROUPS):
        indices = np.flatnonzero(dataset.groups == value)
        if not len(indices):
            continue
        model.zero_grad(set_to_none=True)
        loss = _loss(
            model,
            architecture_id,
            torch.from_numpy(dataset.features[indices]),
            torch.from_numpy(dataset.targets[indices]),
            torch.from_numpy(dataset.masks[indices]),
            torch.from_numpy(dataset.weights[indices]),
            torch.from_numpy(dataset.groups[indices]),
        )
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        vectors[value] = torch.cat(
            [
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
                for parameter, gradient in zip(parameters, gradients)
            ]
        )
    rows = []
    for left, right in itertools.combinations(sorted(vectors), 2):
        a, b = vectors[left], vectors[right]
        denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
        cosine = float(torch.dot(a, b) / denominator) if denominator > 0.0 else 0.0
        rows.append(
            {
                "left_group": left,
                "right_group": right,
                "cosine_similarity": cosine,
                "conflicting": int(cosine < 0.0),
            }
        )
    model.zero_grad(set_to_none=True)
    return rows


def _group_metrics(
    model: nn.Module,
    architecture_id: str,
    dataset: PolicyDataset,
) -> list[dict]:
    rows = []
    for value in range(PLAYER_ROUND_GROUPS):
        indices = np.flatnonzero(dataset.groups == value)
        if not len(indices):
            continue
        subset = PolicyDataset(
            features=dataset.features[indices],
            targets=dataset.targets[indices],
            masks=dataset.masks[indices],
            weights=dataset.weights[indices],
            groups=dataset.groups[indices],
            target_id=dataset.target_id,
        )
        rows.append(
            {
                "group_id": value,
                "player": value // 2,
                "betting_round": value % 2 + 1,
                "num_examples": len(indices),
                "cross_entropy": full_objective(model, architecture_id, subset),
            }
        )
    return rows


def _sample_batch(
    dataset: PolicyDataset, batch_size: int
) -> np.ndarray:
    if len(dataset.features) <= batch_size:
        return np.arange(len(dataset.features), dtype=np.int64)
    # Match the production average-policy trainer's without-replacement sampler.
    return np.asarray(
        random.sample(range(len(dataset.features)), int(batch_size)),
        dtype=np.int64,
    )


def fit_schedule(
    source: SourceData,
    game,
    catalog: Catalog,
    *,
    architecture_id: str,
    dataset: PolicyDataset,
    fit_seed: int,
    learning_rate: float,
    training_budgets: Sequence[int],
    validation_samples: int,
    raw_diagnostics: PolicyDataset | None = None,
    grouped_diagnostics: PolicyDataset | None = None,
) -> list[dict]:
    """Train once to the largest budget and evaluate every requested snapshot."""
    budgets = tuple(sorted({int(value) for value in training_budgets}))
    if not budgets or budgets[0] <= 0:
        raise ValueError("Training budgets must be positive")
    torch.manual_seed(int(fit_seed))
    np.random.seed(int(fit_seed) % (2**32))
    random.seed(int(fit_seed))
    model = build_model(
        architecture_id, source.infostates.shape[1], source.policies.shape[1]
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    batch_size = min(int(source.batch_size), len(dataset.features))
    raw_validation = raw_diagnostics or row_dataset(source, catalog)
    validation_count = min(int(validation_samples), len(raw_validation.features))
    validation_index = np.random.default_rng(34_000 + int(source.seed)).choice(
        len(raw_validation.features), size=validation_count, replace=False
    )
    validation_data = PolicyDataset(
        features=raw_validation.features[validation_index],
        targets=raw_validation.targets[validation_index],
        masks=raw_validation.masks[validation_index],
        weights=raw_validation.weights[validation_index],
        groups=raw_validation.groups[validation_index],
        target_id=ROW_TARGETS,
    )
    grouped_diagnostics = grouped_diagnostics or grouped_dataset(source, catalog)
    started = time.perf_counter()
    results = []
    last_loss = float("nan")
    for step in range(1, budgets[-1] + 1):
        model.train()
        index = _sample_batch(dataset, batch_size)
        loss = _loss(
            model,
            architecture_id,
            torch.from_numpy(dataset.features[index]),
            torch.from_numpy(dataset.targets[index]),
            torch.from_numpy(dataset.masks[index]),
            torch.from_numpy(dataset.weights[index]),
            torch.from_numpy(dataset.groups[index]),
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        last_loss = float(loss.detach())
        if step not in budgets:
            continue
        fitted_table = model_table(model, architecture_id, catalog)
        errors = _policy_errors(
            source.exact_table, fitted_table, source.exact_denominators
        )
        results.append(
            {
                "training_budget": step,
                "learning_rate": float(learning_rate),
                "fit_seconds": time.perf_counter() - started,
                "last_training_loss": last_loss,
                "training_objective": full_objective(model, architecture_id, dataset),
                "reservoir_evaluation_loss": full_objective(
                    model, architecture_id, validation_data
                ),
                "exploitability": exact_exploitability(game, fitted_table),
                "parameter_count": parameter_count(model),
                "model_state": deepcopy(model.state_dict()),
                "policy_table": fitted_table,
                "gradient_rows": _gradient_cosines(
                    model, architecture_id, grouped_diagnostics
                ),
                "group_rows": _group_metrics(
                    model, architecture_id, grouped_diagnostics
                ),
                "error_rows": errors["rows"],
                **{key: value for key, value in errors.items() if key != "rows"},
            }
        )
    return results


def objective_equivalence(
    source: SourceData, catalog: Catalog, probabilities: np.ndarray
) -> tuple[float, float]:
    """Return row and grouped CE for supplied probabilities (test/manifest aid)."""
    raw = row_dataset(source, catalog)
    grouped = grouped_dataset(source, catalog)

    def value(dataset: PolicyDataset, supplied: np.ndarray) -> float:
        log_values = np.log(np.clip(supplied, 1e-12, 1.0))
        losses = -(dataset.targets * log_values).sum(axis=1)
        return float(np.mean(losses * dataset.weights))

    by_tensor = {
        _tensor_key(feature): probabilities[index]
        for index, feature in enumerate(raw.features)
    }
    grouped_probabilities = np.asarray(
        [by_tensor[_tensor_key(feature)] for feature in grouped.features]
    )
    return value(raw, probabilities), value(grouped, grouped_probabilities)


__all__ = [
    "Catalog",
    "PolicyDataset",
    "RoutedPolicyNetwork",
    "build_catalog",
    "build_model",
    "ensemble_table",
    "exact_teacher_dataset",
    "fit_schedule",
    "group_id",
    "grouped_dataset",
    "model_table",
    "objective_equivalence",
    "parameter_count",
    "row_dataset",
]
