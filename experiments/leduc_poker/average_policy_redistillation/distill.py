"""Policy-table decomposition and neural redistillation primitives."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table,
    exact_exploitability,
)
from vr_deep_cfr.solver import MLP


def _tensor_key(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


@dataclass
class SourceData:
    checkpoint_id: str
    seed: int
    iteration: int
    gamma: float
    network_layers: tuple[int, ...]
    learning_rate: float
    batch_size: int
    train_steps: int
    infostates: np.ndarray
    policies: np.ndarray
    legal_masks: np.ndarray
    iterations: np.ndarray
    exact_table: dict
    exact_denominators: dict
    source_model_state: dict[str, torch.Tensor]


def extract_source(payload: Mapping[str, Any], *, seed: int, checkpoint_id: str) -> SourceData:
    if payload.get("type") != "experiment_25_full_training_state":
        raise ValueError("Source is not an Experiment 25 continuation state")
    if payload.get("variant_id") != "averaged_critic_target":
        raise ValueError("Experiment 27 requires the averaged-critic-target source arm")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError("Source seed differs from the worker contract")
    if payload.get("checkpoint_id") != checkpoint_id:
        raise ValueError("Source checkpoint differs from the worker contract")
    config = payload["config"]
    buffer = payload["average_policy_trainer"]["buffer"]
    size = int(buffer["size"])
    if size <= 0:
        raise ValueError("Average-policy reservoir is empty")
    exact = payload["exact_average_strategy"]
    exact_table = {}
    for key, numerator in exact["numerators"].items():
        denominator = float(exact["denominators"][key])
        if denominator > 0.0:
            exact_table[key] = np.asarray(numerator, dtype=np.float64) / denominator
    return SourceData(
        checkpoint_id=checkpoint_id,
        seed=int(seed),
        iteration=int(payload["solver"]["num_iteration"]),
        gamma=float(config["gamma"]),
        network_layers=tuple(
            int(config["num_hiddens"]) for _ in range(int(config["num_layers"]))
        ),
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["ave_policy_batch_size"]),
        train_steps=int(config["ave_policy_network_train_steps"]),
        infostates=np.asarray(buffer["infostate"][:size], dtype=np.float32),
        policies=np.asarray(buffer["q_value"][:size], dtype=np.float32),
        legal_masks=np.asarray(buffer["q_value_mask"][:size], dtype=np.float32),
        iterations=np.asarray(buffer["iteration"][:size], dtype=np.float32).reshape(-1),
        exact_table=exact_table,
        exact_denominators={key: float(value) for key, value in exact["denominators"].items()},
        source_model_state={
            name: tensor.detach().cpu().clone()
            for name, tensor in payload["average_policy_trainer"]["model"].items()
        },
    )


def empirical_table(source: SourceData, game) -> tuple[dict, float]:
    unique, inverse = np.unique(source.infostates, axis=0, return_inverse=True)
    sample_weights = np.power(
        np.maximum(source.iterations, 0.0), source.gamma, dtype=np.float64
    )
    numerators = np.zeros((len(unique), source.policies.shape[1]), dtype=np.float64)
    denominators = np.zeros(len(unique), dtype=np.float64)
    np.add.at(numerators, inverse, source.policies * sample_weights[:, None])
    np.add.at(denominators, inverse, sample_weights)
    values = numerators / np.maximum(denominators[:, None], 1e-30)
    lookup = {_tensor_key(state): policy for state, policy in zip(unique, values)}

    def strategy(state):
        value = lookup.get(_tensor_key(state.information_state_tensor()))
        if value is None:
            result = np.zeros(game.num_distinct_actions(), dtype=np.float64)
            legal = state.legal_actions()
            result[legal] = 1.0 / len(legal)
            return result
        return value

    table = build_policy_table(game, strategy)
    return table, exact_exploitability(game, table)


def _model_table(game, model: MLP) -> dict:
    model.eval()

    def strategy(state):
        features = torch.as_tensor(
            state.information_state_tensor(), dtype=torch.float32
        ).unsqueeze(0)
        mask = torch.as_tensor(state.legal_actions_mask(), dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            logits = model(features)
            probabilities = torch.softmax(logits.masked_fill(~mask, -1e20), dim=-1)
        return probabilities.squeeze(0).cpu().numpy()

    return build_policy_table(game, strategy)


def evaluate_model_state(source: SourceData, game, state_dict: Mapping[str, torch.Tensor]) -> tuple[float, dict]:
    model = MLP(
        source.infostates.shape[1],
        list(source.network_layers),
        source.policies.shape[1],
    )
    model.load_state_dict(state_dict)
    fitted_table = _model_table(game, model)
    return exact_exploitability(game, fitted_table), fitted_table


def _policy_errors(exact_table: Mapping, fitted_table: Mapping, denominators: Mapping) -> dict:
    rows = []
    for key, target in exact_table.items():
        fitted = np.asarray(fitted_table[key], dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        legal = np.logical_or(target > 0.0, fitted > 0.0)
        clipped = np.clip(fitted[legal], 1e-12, 1.0)
        target_legal = target[legal]
        rows.append(
            {
                "information_set": str(key),
                "reach_weight": float(denominators.get(key, 0.0)),
                "l1_error": float(np.sum(np.abs(fitted - target))),
                "kl_exact_to_neural": float(
                    np.sum(
                        np.where(
                            target_legal > 0.0,
                            target_legal * (np.log(np.clip(target_legal, 1e-12, 1.0)) - np.log(clipped)),
                            0.0,
                        )
                    )
                ),
            }
        )
    weights = np.asarray([row["reach_weight"] for row in rows], dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-30)
    return {
        "mean_l1": float(np.mean([row["l1_error"] for row in rows])),
        "max_l1": float(np.max([row["l1_error"] for row in rows])),
        "reach_weighted_l1": float(np.dot(weights, [row["l1_error"] for row in rows])),
        "mean_kl": float(np.mean([row["kl_exact_to_neural"] for row in rows])),
        "reach_weighted_kl": float(np.dot(weights, [row["kl_exact_to_neural"] for row in rows])),
        "rows": rows,
    }


def _loss(model, features, targets, masks, iterations, *, total_iteration, gamma, loss_name):
    logits = model(features).masked_fill(masks != 1, -1e20)
    probabilities = torch.softmax(logits, dim=-1)
    weights = torch.pow(iterations / float(total_iteration) * 2.0, gamma)
    if loss_name == "mse":
        scale = torch.sqrt(weights).unsqueeze(1)
        return F.mse_loss(probabilities * scale, targets * scale)
    if loss_name == "cross_entropy":
        per_sample = -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=1)
        return torch.mean(per_sample * weights)
    raise ValueError(f"Unknown loss {loss_name}")


def fit_reservoir_policy(
    source: SourceData,
    game,
    *,
    loss_name: str,
    fit_seed: int,
    train_steps: int,
    validation_samples: int,
    initial_state: Mapping[str, Any] | None = None,
) -> tuple[dict, dict, list[dict]]:
    torch.manual_seed(int(fit_seed))
    np.random.seed(int(fit_seed) % (2**32))
    random.seed(int(fit_seed))
    model = MLP(source.infostates.shape[1], list(source.network_layers), source.policies.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=source.learning_rate)
    if initial_state is not None:
        model.load_state_dict(initial_state["model"])
        optimizer.load_state_dict(initial_state["optimizer"])
    rng = np.random.default_rng(int(fit_seed))
    started = time.perf_counter()
    model.train()
    last_loss = float("nan")
    for _ in range(int(train_steps)):
        replace = source.batch_size > len(source.infostates)
        index = rng.choice(len(source.infostates), size=min(source.batch_size, len(source.infostates)), replace=replace)
        batch = (
            torch.from_numpy(source.infostates[index]),
            torch.from_numpy(source.policies[index]),
            torch.from_numpy(source.legal_masks[index]),
            torch.from_numpy(source.iterations[index]),
        )
        loss = _loss(
            model,
            *batch,
            total_iteration=source.iteration,
            gamma=source.gamma,
            loss_name=loss_name,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    fit_seconds = time.perf_counter() - started
    count = min(int(validation_samples), len(source.infostates))
    validation_index = np.random.default_rng(27_000 + int(source.seed)).choice(
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
                loss_name=loss_name,
            )
        )
    fitted_table = _model_table(game, model)
    errors = _policy_errors(source.exact_table, fitted_table, source.exact_denominators)
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
    return metrics, state, errors["rows"]


def fit_oracle_policy(
    source: SourceData,
    game,
    *,
    fit_seed: int,
    train_steps: int,
) -> tuple[dict, dict, list[dict]]:
    states, targets, masks, weights = [], [], [], []
    state_rows = {}

    def collect(state):
        key = (int(state.current_player()), str(state.information_state_string(state.current_player())))
        state_rows.setdefault(key, state)
        target = source.exact_table.get(key)
        if target is not None:
            return target
        fallback = np.zeros(game.num_distinct_actions(), dtype=np.float64)
        legal = state.legal_actions()
        fallback[legal] = 1.0 / len(legal)
        return fallback

    build_policy_table(game, collect)
    for key, state in state_rows.items():
        states.append(state.information_state_tensor())
        target = source.exact_table.get(key)
        if target is None:
            target = np.zeros(game.num_distinct_actions(), dtype=np.float64)
            legal = state.legal_actions()
            target[legal] = 1.0 / len(legal)
        targets.append(target)
        masks.append(state.legal_actions_mask())
        weights.append(source.exact_denominators.get(key, 0.0))
    features = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    targets_t = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
    masks_t = torch.as_tensor(np.asarray(masks), dtype=torch.bool)
    weights_t = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    positive = weights_t[weights_t > 0]
    if positive.numel():
        # The tabular evaluator defines never-reached information sets as
        # uniform. Give those rows a small diagnostic fitting weight so the
        # capacity bound represents the complete deployed policy.
        weights_t = torch.where(
            weights_t > 0,
            weights_t,
            torch.min(positive) * 0.01,
        )
    else:
        weights_t.fill_(1.0)
    weights_t /= torch.clamp(weights_t.mean(), min=1e-30)
    torch.manual_seed(int(fit_seed))
    model = MLP(features.shape[1], list(source.network_layers), targets_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=source.learning_rate)
    started = time.perf_counter()
    last_loss = float("nan")
    for _ in range(int(train_steps)):
        logits = model(features).masked_fill(~masks_t, -1e20)
        per_row = -(targets_t * torch.log_softmax(logits, dim=-1)).sum(dim=1)
        loss = torch.mean(per_row * weights_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    fitted_table = _model_table(game, model)
    errors = _policy_errors(source.exact_table, fitted_table, source.exact_denominators)
    return (
        {
            "exploitability": exact_exploitability(game, fitted_table),
            "last_training_loss": last_loss,
            "reservoir_evaluation_loss": float("nan"),
            "fit_seconds": time.perf_counter() - started,
            **{key: value for key, value in errors.items() if key != "rows"},
        },
        {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict())},
        errors["rows"],
    )
