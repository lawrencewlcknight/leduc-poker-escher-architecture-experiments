"""Exact Leduc and neural-policy utilities for Experiments 36--39."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from open_spiel.python import policy
from open_spiel.python.algorithms import best_response, expected_game_score

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table,
    exact_exploitability,
    table_action_probabilities,
)
from experiments.leduc_poker.average_policy_redistillation.distill import SourceData
from vr_deep_cfr.solver import MLP


PolicyKey = tuple[int, str]


@dataclass(frozen=True)
class InformationSet:
    key: PolicyKey
    tensor: np.ndarray
    legal_actions: tuple[int, ...]


def infoset_key(state) -> PolicyKey:
    player = int(state.current_player())
    return player, str(state.information_state_string(player))


def information_sets(game) -> tuple[InformationSet, ...]:
    rows = {}

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return
        key = infoset_key(state)
        tensor = np.asarray(state.information_state_tensor(), dtype=np.float32)
        legal = tuple(int(action) for action in state.legal_actions())
        previous = rows.get(key)
        if previous is not None and (
            previous.legal_actions != legal
            or not np.array_equal(previous.tensor, tensor)
        ):
            raise ValueError(f"Inconsistent information set {key!r}")
        rows[key] = InformationSet(key, tensor, legal)
        for action in legal:
            walk(state.child(action))

    walk(game.new_initial_state())
    return tuple(rows[key] for key in sorted(rows))


def make_model(source: SourceData, state=None) -> MLP:
    model = MLP(
        source.infostates.shape[1],
        list(source.network_layers),
        source.policies.shape[1],
    )
    model.load_state_dict(source.source_model_state if state is None else state)
    return model


def model_table(game, model) -> dict:
    model.eval()

    def strategy(state):
        features = torch.as_tensor(
            state.information_state_tensor(), dtype=torch.float32
        ).unsqueeze(0)
        mask = torch.as_tensor(
            state.legal_actions_mask(), dtype=torch.bool
        ).unsqueeze(0)
        with torch.no_grad():
            logits = model(features).masked_fill(~mask, -1e20)
            return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return build_policy_table(game, strategy)


def tabular_policy(game, table):
    return policy.tabular_policy_from_callable(
        game, lambda state: table_action_probabilities(table, state)
    )


def best_response_actions(game, table, responder: int) -> dict[str, int]:
    response = best_response.BestResponsePolicy(
        game, int(responder), tabular_policy(game, table)
    )
    return {
        str(info): int(response.best_response_action(info))
        for info in response.infosets
    }


def best_response_reach(game, table) -> dict[PolicyKey, float]:
    """Actual visitation under each player's opponent best response."""
    result = defaultdict(float)
    for target_player in range(game.num_players()):
        opponent = 1 - target_player
        actions = best_response_actions(game, table, opponent)

        def walk(state, reach):
            if state.is_terminal():
                return
            if state.is_chance_node():
                for action, probability in state.chance_outcomes():
                    walk(state.child(action), reach * float(probability))
                return
            player_id = int(state.current_player())
            if player_id == target_player:
                key = infoset_key(state)
                result[key] += reach
                for action, probability in table_action_probabilities(
                    table, state
                ).items():
                    walk(state.child(action), reach * float(probability))
            else:
                action = actions[state.information_state_string(player_id)]
                walk(state.child(action), reach)

        walk(game.new_initial_state(), 1.0)
    return dict(result)


def policy_probabilities(model, rows):
    features = torch.as_tensor(
        np.asarray([row.tensor for row in rows]), dtype=torch.float32
    )
    raw_logits = model(features)
    masks = torch.zeros_like(raw_logits, dtype=torch.bool)
    for index, row in enumerate(rows):
        masks[index, list(row.legal_actions)] = True
    logits = raw_logits.masked_fill(~masks, -1e20)
    return logits, torch.softmax(logits, dim=-1), masks


def policy_kl(model, rows, blueprint_table) -> torch.Tensor:
    _, probabilities, masks = policy_probabilities(model, rows)
    target = torch.as_tensor(
        np.asarray([blueprint_table[row.key] for row in rows]),
        dtype=torch.float32,
    )
    log_current = torch.log(torch.clamp(probabilities, min=1e-12))
    log_target = torch.log(torch.clamp(target, min=1e-12))
    values = torch.where(masks, probabilities * (log_current - log_target), 0.0)
    return values.sum(dim=1).mean()


def expected_return_against_fixed_response(
    game, model, *, player_id: int, opponent_actions: Mapping[str, int]
):
    action_size = int(game.num_distinct_actions())

    def walk(state):
        if state.is_terminal():
            return torch.tensor(float(state.returns()[player_id]))
        if state.is_chance_node():
            return sum(
                float(probability) * walk(state.child(action))
                for action, probability in state.chance_outcomes()
            )
        acting = int(state.current_player())
        if acting != player_id:
            action = opponent_actions[state.information_state_string(acting)]
            return walk(state.child(action))
        features = torch.as_tensor(
            state.information_state_tensor(), dtype=torch.float32
        ).unsqueeze(0)
        mask = torch.as_tensor(
            state.legal_actions_mask(), dtype=torch.bool
        ).unsqueeze(0)
        probabilities = torch.softmax(
            model(features).masked_fill(~mask, -1e20), dim=-1
        ).squeeze(0)
        value = torch.tensor(0.0)
        for action in state.legal_actions():
            value = value + probabilities[action] * walk(state.child(action))
        return value

    return walk(game.new_initial_state())


def exact_policy_advantages(game, table, rows):
    """Counterfactual action advantages under the current joint policy."""
    tabular = tabular_policy(game, table)
    cache = {}

    def value(state, player_id):
        key = player_id, tuple(int(action) for action in state.history())
        if key in cache:
            return cache[key]
        if state.is_terminal():
            result = float(state.returns()[player_id])
        elif state.is_chance_node():
            result = sum(
                float(probability) * value(state.child(action), player_id)
                for action, probability in state.chance_outcomes()
            )
        else:
            result = sum(
                probability * value(state.child(action), player_id)
                for action, probability in table_action_probabilities(
                    table, state
                ).items()
            )
        cache[key] = result
        return result

    advantages, masses = {}, {}
    for player_id in range(game.num_players()):
        helper = best_response.BestResponsePolicy(game, player_id, tabular)
        for info, histories in helper.infosets.items():
            key = (player_id, str(info))
            legal = histories[0][0].legal_actions()
            q_values = np.zeros(game.num_distinct_actions(), dtype=np.float64)
            mass = sum(float(cf_reach) for _, cf_reach in histories)
            for action in legal:
                q_values[action] = sum(
                    float(cf_reach) * value(state.child(action), player_id)
                    for state, cf_reach in histories
                ) / max(mass, 1e-30)
            baseline = float(np.dot(table[key], q_values))
            advantages[key] = q_values - baseline
            masses[key] = mass
    return advantages, masses


def exact_metrics(game, table, blueprint_table, exact_table) -> dict:
    candidate = tabular_policy(game, table)
    blueprint = tabular_policy(game, blueprint_table)
    values_0 = expected_game_score.policy_value(
        game.new_initial_state(), [candidate, blueprint]
    )
    values_1 = expected_game_score.policy_value(
        game.new_initial_state(), [blueprint, candidate]
    )
    kls = []
    for key, base in blueprint_table.items():
        current = np.asarray(table[key], dtype=np.float64)
        base = np.asarray(base, dtype=np.float64)
        legal = np.logical_or(current > 0.0, base > 0.0)
        kls.append(
            float(
                np.sum(
                    current[legal]
                    * (
                        np.log(np.clip(current[legal], 1e-12, 1.0))
                        - np.log(np.clip(base[legal], 1e-12, 1.0))
                    )
                )
            )
        )
    return {
        "exploitability": exact_exploitability(game, table),
        "exact_average_exploitability": exact_exploitability(game, exact_table),
        "mean_kl_from_blueprint": float(np.mean(kls)),
        "ev_vs_blueprint_player_0": float(values_0[0]),
        "ev_vs_blueprint_player_1": float(values_1[1]),
        "ev_vs_blueprint_seat_averaged": 0.5
        * (float(values_0[0]) + float(values_1[1])),
    }


def clone_state(model) -> dict:
    return deepcopy({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()})


__all__ = [
    "best_response_actions", "best_response_reach", "clone_state",
    "exact_metrics", "exact_policy_advantages",
    "expected_return_against_fixed_response", "information_sets", "make_model",
    "model_table", "policy_kl", "policy_probabilities",
]
