"""Consequence-weighted average-policy fitting primitives."""

from __future__ import annotations

from copy import deepcopy
import random
import time
from typing import Mapping, Sequence

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
from experiments.leduc_poker.causal_rare_state_audit.audit import _infoset_catalog
from vr_deep_cfr.solver import MLP

from .config import (
    BASELINE,
    CONSEQUENCE_WEIGHTED,
    ERROR_CONSEQUENCE_WEIGHTED,
    ORACLE_WEIGHTED,
    PRIORITISED,
    TARGETED_BATCH_FRACTION,
    TWO_STAGE,
    WEIGHT_EXPONENT,
    WEIGHT_MAX,
    WEIGHT_MIN,
)


def _tensor_key(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _normalised_clipped_weights(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if not np.any(values > 0.0):
        return np.ones_like(values, dtype=np.float32)
    positive = values[values > 0.0]
    floor = max(float(np.min(positive)) * 0.1, 1e-30)
    transformed = np.power(np.maximum(values, floor), WEIGHT_EXPONENT)
    transformed /= max(float(np.mean(transformed)), 1e-30)
    transformed = np.clip(transformed, WEIGHT_MIN, WEIGHT_MAX)
    transformed /= max(float(np.mean(transformed)), 1e-30)
    return transformed.astype(np.float32)


def reservoir_weight_plan(
    source: SourceData,
    game,
    proxy_rows: Sequence[Mapping],
    selected_proxy_id: str,
) -> dict:
    """Map information-set consequence, error and oracle scores to reservoir rows."""
    catalog = _infoset_catalog(game)
    tensor_to_key = {_tensor_key(row["tensor"]): key for key, row in catalog.items()}
    by_key = {
        (int(row["player"]), str(row["information_set"])): row
        for row in proxy_rows
        if row["checkpoint_id"] == source.checkpoint_id
    }
    if set(by_key) != set(catalog):
        raise ValueError("Proxy rows do not match the complete infoset catalog")
    row_keys = [tensor_to_key.get(_tensor_key(row)) for row in source.infostates]
    if any(key is None for key in row_keys):
        raise ValueError("Reservoir contains an unknown information-state tensor")

    consequence = np.asarray(
        [float(by_key[key][selected_proxy_id]) for key in row_keys], dtype=np.float64
    )
    error_consequence = np.asarray(
        [
            float(by_key[key][selected_proxy_id])
            * max(float(by_key[key]["policy_error"]), 0.0)
            for key in row_keys
        ],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [float(by_key[key]["single_repair_gain_positive"]) for key in row_keys],
        dtype=np.float64,
    )
    consequence_weights = _normalised_clipped_weights(consequence)
    error_weights = _normalised_clipped_weights(error_consequence)
    oracle_weights = _normalised_clipped_weights(oracle)
    proposal = consequence_weights.astype(np.float64)
    proposal /= proposal.sum()
    importance = (1.0 / float(len(proposal))) / proposal
    return {
        "row_keys": row_keys,
        "rows_by_key": by_key,
        "consequence_weights": consequence_weights,
        "error_consequence_weights": error_weights,
        "oracle_weights": oracle_weights,
        "priority_probabilities": proposal,
        "priority_importance": importance.astype(np.float32),
    }


def _draw_batch(
    rng: np.random.Generator,
    size: int,
    batch_size: int,
    probabilities: np.ndarray | None,
) -> np.ndarray:
    return rng.choice(
        size,
        size=min(int(batch_size), size),
        replace=int(batch_size) > size or probabilities is not None,
        p=probabilities,
    )


def fit_consequence_policy(
    source: SourceData,
    game,
    *,
    arm_id: str,
    selected_proxy_id: str,
    proxy_rows: Sequence[Mapping],
    fit_seed: int,
    train_steps: int,
    fine_tune_steps: int,
    validation_samples: int,
) -> tuple[dict, dict, list[dict], dict]:
    """Fit one policy under the frozen Experiment 33 arm contract."""
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
    plan = reservoir_weight_plan(source, game, proxy_rows, selected_proxy_id)
    total_steps = int(train_steps) + (int(fine_tune_steps) if arm_id == TWO_STAGE else 0)
    started = time.perf_counter()
    model.train()
    last_loss = float("nan")

    for step in range(total_steps):
        targeted_phase = arm_id == TWO_STAGE and step >= int(train_steps)
        if arm_id == PRIORITISED or targeted_phase:
            if targeted_phase:
                targeted_count = int(round(source.batch_size * TARGETED_BATCH_FRACTION))
                target_index = _draw_batch(
                    rng,
                    len(source.infostates),
                    targeted_count,
                    plan["priority_probabilities"],
                )
                base_index = _draw_batch(
                    rng,
                    len(source.infostates),
                    source.batch_size - targeted_count,
                    None,
                )
                index = np.concatenate((target_index, base_index))
            else:
                index = _draw_batch(
                    rng,
                    len(source.infostates),
                    source.batch_size,
                    plan["priority_probabilities"],
                )
        else:
            index = _draw_batch(
                rng, len(source.infostates), source.batch_size, None
            )

        features = torch.from_numpy(source.infostates[index])
        targets = torch.from_numpy(source.policies[index])
        masks = torch.from_numpy(source.legal_masks[index])
        logits = model(features).masked_fill(masks != 1, -1e20)
        per_sample = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=1)
        iteration_weights = torch.pow(
            torch.from_numpy(source.iterations[index])
            / float(source.iteration)
            * 2.0,
            source.gamma,
        )
        extra = torch.ones_like(iteration_weights)
        if arm_id == PRIORITISED:
            extra = torch.from_numpy(plan["priority_importance"][index])
        elif arm_id == CONSEQUENCE_WEIGHTED:
            extra = torch.from_numpy(plan["consequence_weights"][index])
        elif arm_id == ERROR_CONSEQUENCE_WEIGHTED:
            extra = torch.from_numpy(plan["error_consequence_weights"][index])
        elif arm_id == ORACLE_WEIGHTED:
            extra = torch.from_numpy(plan["oracle_weights"][index])
        elif targeted_phase:
            extra[: len(target_index)] = torch.from_numpy(
                plan["consequence_weights"][target_index]
            )
        loss = torch.mean(per_sample * iteration_weights * extra)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())

    fit_seconds = time.perf_counter() - started
    count = min(int(validation_samples), len(source.infostates))
    validation_index = np.random.default_rng(33_000 + int(source.seed)).choice(
        len(source.infostates), size=count, replace=False
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
    errors = _policy_errors(source.exact_table, fitted_table, source.exact_denominators)
    details = []
    for key, proxy in sorted(plan["rows_by_key"].items()):
        target = np.asarray(source.exact_table.get(key, fitted_table[key]), dtype=np.float64)
        fitted = np.asarray(fitted_table[key], dtype=np.float64)
        legal = np.logical_or(target > 0.0, fitted > 0.0)
        details.append(
            {
                "player": int(key[0]),
                "information_set": str(key[1]),
                "selected_proxy_score": float(proxy[selected_proxy_id]),
                "single_repair_gain_positive": float(
                    proxy["single_repair_gain_positive"]
                ),
                "l1_error": float(np.sum(np.abs(target - fitted))),
                "kl_exact_to_neural": float(
                    np.sum(
                        np.where(
                            target[legal] > 0.0,
                            target[legal]
                            * (
                                np.log(np.clip(target[legal], 1e-12, 1.0))
                                - np.log(np.clip(fitted[legal], 1e-12, 1.0))
                            ),
                            0.0,
                        )
                    )
                ),
            }
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
    sampling = {
        "priority_probability_min": float(np.min(plan["priority_probabilities"])),
        "priority_probability_max": float(np.max(plan["priority_probabilities"])),
        "priority_importance_min": float(np.min(plan["priority_importance"])),
        "priority_importance_max": float(np.max(plan["priority_importance"])),
        "consequence_weight_min": float(np.min(plan["consequence_weights"])),
        "consequence_weight_max": float(np.max(plan["consequence_weights"])),
        "consequence_weight_effective_sample_fraction": float(
            np.square(np.sum(plan["consequence_weights"]))
            / (
                len(plan["consequence_weights"])
                * np.sum(np.square(plan["consequence_weights"]))
            )
        ),
    }
    return metrics, state, details, sampling


__all__ = ["fit_consequence_policy", "reservoir_weight_plan"]
