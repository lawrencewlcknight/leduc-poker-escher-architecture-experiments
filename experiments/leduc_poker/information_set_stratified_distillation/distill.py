"""Importance-corrected information-set minibatching for policy fitting."""

from __future__ import annotations

from copy import deepcopy
import random
import time

import numpy as np
import torch

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    SourceData,
    _loss,
    _model_table,
    _policy_errors,
)
from vr_deep_cfr.solver import MLP


def information_set_sampling_plan(source: SourceData, exponent: float) -> dict:
    """Construct q(I) proportional to p(I)**exponent and exact p/q weights."""
    if not 0.0 <= float(exponent) <= 1.0:
        raise ValueError("information-set sampling exponent must be in [0, 1]")
    _, inverse, counts = np.unique(
        source.infostates,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    counts = counts.astype(np.int64, copy=False)
    empirical = counts.astype(np.float64) / float(counts.sum())
    proposal = np.power(empirical, float(exponent), dtype=np.float64)
    proposal /= proposal.sum()
    importance = empirical / proposal
    order = np.argsort(inverse, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int64)))
    return {
        "inverse": inverse,
        "counts": counts,
        "empirical_probabilities": empirical,
        "proposal_probabilities": proposal,
        "importance_ratios": importance,
        "row_order": order,
        "group_offsets": offsets,
    }


def draw_information_set_batch(
    rng: np.random.Generator,
    plan: dict,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an information set from q, then one reservoir row within it."""
    groups = rng.choice(
        len(plan["counts"]),
        size=int(batch_size),
        replace=True,
        p=plan["proposal_probabilities"],
    )
    positions = (
        plan["group_offsets"][groups]
        + np.floor(rng.random(int(batch_size)) * plan["counts"][groups]).astype(np.int64)
    )
    rows = plan["row_order"][positions]
    return rows, plan["importance_ratios"][groups].astype(np.float32, copy=False)


def _sampling_diagnostics(plan: dict) -> dict:
    q = plan["proposal_probabilities"]
    ratio = plan["importance_ratios"]
    second_moment = float(np.dot(q, np.square(ratio)))
    return {
        "num_information_sets": int(len(q)),
        "proposal_probability_min": float(np.min(q)),
        "proposal_probability_max": float(np.max(q)),
        "importance_ratio_min": float(np.min(ratio)),
        "importance_ratio_max": float(np.max(ratio)),
        "importance_ratio_expected": float(np.dot(q, ratio)),
        "importance_ratio_second_moment": second_moment,
        "importance_effective_sample_fraction": float(1.0 / second_moment),
    }


def fit_information_set_policy(
    source: SourceData,
    game,
    *,
    information_set_exponent: float,
    fit_seed: int,
    train_steps: int,
    validation_samples: int,
) -> tuple[dict, dict, list[dict], dict]:
    """Fit one CE policy using information-set sampling and p(I)/q(I)."""
    torch.manual_seed(int(fit_seed))
    np.random.seed(int(fit_seed) % (2**32))
    random.seed(int(fit_seed))
    model = MLP(
        source.infostates.shape[1],
        list(source.network_layers),
        source.policies.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=source.learning_rate)
    rng = np.random.default_rng(int(fit_seed))
    plan = information_set_sampling_plan(source, information_set_exponent)
    batch_size = min(source.batch_size, len(source.infostates))
    started = time.perf_counter()
    model.train()
    last_loss = float("nan")
    for _ in range(int(train_steps)):
        index, importance = draw_information_set_batch(rng, plan, batch_size)
        logits = model(torch.from_numpy(source.infostates[index])).masked_fill(
            torch.from_numpy(source.legal_masks[index]) != 1,
            -1e20,
        )
        per_sample = -(
            torch.from_numpy(source.policies[index]) * torch.log_softmax(logits, dim=-1)
        ).sum(dim=1)
        iteration_weights = torch.pow(
            torch.from_numpy(source.iterations[index])
            / float(source.iteration)
            * 2.0,
            source.gamma,
        )
        loss = torch.mean(
            per_sample * iteration_weights * torch.from_numpy(importance)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    fit_seconds = time.perf_counter() - started

    count = min(int(validation_samples), len(source.infostates))
    validation_index = np.random.default_rng(28_000 + int(source.seed)).choice(
        len(source.infostates),
        size=count,
        replace=False,
    )
    with torch.no_grad():
        validation_loss = float(
            _loss(
                model,
                torch.from_numpy(source.infostates[validation_index]),
                torch.from_numpy(source.policies[validation_index]),
                torch.from_numpy(source.legal_masks[validation_index]),
                torch.from_numpy(source.iterations[validation_index]),
                total_iteration=source.iteration,
                gamma=source.gamma,
                loss_name="cross_entropy",
            )
        )
    fitted_table = _model_table(game, model)
    errors = _policy_errors(
        source.exact_table,
        fitted_table,
        source.exact_denominators,
    )
    metrics = {
        "exploitability": exact_exploitability(game, fitted_table),
        "last_training_loss": last_loss,
        "reservoir_evaluation_loss": validation_loss,
        "fit_seconds": fit_seconds,
        **{key: value for key, value in errors.items() if key != "rows"},
    }
    state = {
        "model": deepcopy(model.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
    }
    return metrics, state, errors["rows"], _sampling_diagnostics(plan)


__all__ = [
    "draw_information_set_batch",
    "fit_information_set_policy",
    "information_set_sampling_plan",
]
