"""Optimisation methods for Experiments 36--39."""

from __future__ import annotations

import random
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from experiments.leduc_poker.average_policy_redistillation.distill import (
    empirical_table,
)
from vr_deep_cfr.solver import MLP

from .core import (
    best_response_actions,
    best_response_reach,
    exact_policy_advantages,
    expected_return_against_fixed_response,
    model_table,
    policy_kl,
    policy_probabilities,
)


Record = Callable[[torch.nn.Module, int, float, Mapping], None]


def _seed(seed: int) -> np.random.Generator:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    return np.random.default_rng(int(seed))


def best_response_guided_repair(
    *, game, source, model, rows, blueprint_table, config, arm, record: Record,
):
    _seed(source.seed)
    target_table, _ = empirical_table(source, game)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(arm["learning_rate"]))
    updates = int(config["updates"])
    refresh = int(config["refresh_interval"])
    weights = None
    target = torch.as_tensor(
        np.asarray([target_table[row.key] for row in rows]), dtype=torch.float32
    )
    record(model, 0, float("nan"), {"br_reach_mass": 0.0})
    for step in range(1, updates + 1):
        if weights is None or (step - 1) % refresh == 0:
            reach = best_response_reach(game, model_table(game, model))
            raw = np.asarray([reach.get(row.key, 0.0) for row in rows])
            if float(raw.mean()) > 0.0:
                raw = raw / float(raw.mean())
            weights = torch.as_tensor(
                1.0 + float(arm["br_weight"]) * raw, dtype=torch.float32
            )
        logits, _, _ = policy_probabilities(model, rows)
        per_row = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=1)
        loss = torch.mean(per_row * weights)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        record(
            model, step, float(loss.detach()),
            {"br_reach_mass": float(torch.sum(weights - 1.0))},
        )


def kl_exploitability_descent(
    *, game, source, model, rows, blueprint_table, config, arm, record: Record,
):
    _seed(source.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(arm["learning_rate"]))
    updates = int(config["updates"])
    refresh = int(config["refresh_interval"])
    responses = None
    record(model, 0, float("nan"), {"kl_penalty": 0.0})
    for step in range(1, updates + 1):
        if responses is None or (step - 1) % refresh == 0:
            current = model_table(game, model)
            responses = {
                opponent: best_response_actions(game, current, opponent)
                for opponent in range(game.num_players())
            }
        robust_values = [
            expected_return_against_fixed_response(
                game,
                model,
                player_id=player_id,
                opponent_actions=responses[1 - player_id],
            )
            for player_id in range(game.num_players())
        ]
        kl = policy_kl(model, rows, blueprint_table)
        loss = -torch.stack(robust_values).mean() + float(
            arm["kl_coefficient"]
        ) * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        record(
            model, step, float(loss.detach()),
            {
                "kl_penalty": float(kl.detach()),
                "mean_robust_value": float(torch.stack(robust_values).mean().detach()),
            },
        )


def neurd_fine_tuning(
    *, game, source, model, rows, blueprint_table, config, arm, record: Record,
):
    _seed(source.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(arm["learning_rate"]))
    record(model, 0, float("nan"), {"counterfactual_advantage_rms": 0.0})
    for step in range(1, int(config["updates"]) + 1):
        table = model_table(game, model)
        advantages, masses = exact_policy_advantages(game, table, rows)
        logits, probabilities, masks = policy_probabilities(model, rows)
        advantage = torch.as_tensor(
            np.asarray([advantages[row.key] for row in rows]), dtype=torch.float32
        )
        weight = torch.as_tensor(
            np.asarray([masses[row.key] for row in rows]), dtype=torch.float32
        )
        weight = weight / torch.clamp(weight.mean(), min=1e-12)
        # NeuRD bypasses the derivative through softmax.  Detached policy
        # probabilities supply the Hedge/replicator weighting of advantages.
        terms = torch.where(
            masks, probabilities.detach() * advantage * logits, 0.0
        ).sum(dim=1)
        loss = -torch.mean(weight * terms)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        legal_advantage = advantage[masks]
        record(
            model, step, float(loss.detach()),
            {
                "counterfactual_advantage_rms": float(
                    torch.sqrt(torch.mean(torch.square(legal_advantage)))
                )
            },
        )


def _collect_ppo_rollouts(game, model, value_model, rng, episodes: int):
    transitions = []
    model.eval()
    value_model.eval()
    for _ in range(int(episodes)):
        state = game.new_initial_state()
        trajectory = []
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probabilities = zip(*state.chance_outcomes())
                state.apply_action(int(rng.choice(actions, p=probabilities)))
                continue
            feature = np.asarray(state.information_state_tensor(), dtype=np.float32)
            mask = np.asarray(state.legal_actions_mask(), dtype=np.bool_)
            with torch.no_grad():
                feature_t = torch.from_numpy(feature).unsqueeze(0)
                mask_t = torch.from_numpy(mask).unsqueeze(0)
                logits = model(feature_t).masked_fill(~mask_t, -1e20)
                probabilities = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                value = float(value_model(feature_t).squeeze())
            action = int(rng.choice(len(probabilities), p=probabilities))
            trajectory.append(
                (feature, mask, action, float(np.log(max(probabilities[action], 1e-12))),
                 int(state.current_player()), value)
            )
            state.apply_action(action)
        returns = state.returns()
        transitions.extend((*row, float(returns[row[4]])) for row in trajectory)
    return transitions


def ppo_self_play(
    *, game, source, model, rows, blueprint_table, config, arm, record: Record,
):
    rng = _seed(source.seed)
    input_size = int(source.infostates.shape[1])
    value_model = MLP(input_size, [64, 64], 1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(value_model.parameters()),
        lr=float(arm["learning_rate"]),
    )
    record(model, 0, float("nan"), {"mean_episode_return": 0.0})
    for step in range(1, int(config["updates"]) + 1):
        transitions = _collect_ppo_rollouts(
            game, model, value_model, rng, int(config["rollout_episodes"])
        )
        features = torch.as_tensor(np.asarray([row[0] for row in transitions]))
        masks = torch.as_tensor(np.asarray([row[1] for row in transitions]))
        actions = torch.as_tensor([row[2] for row in transitions], dtype=torch.long)
        old_log = torch.as_tensor([row[3] for row in transitions], dtype=torch.float32)
        old_values = torch.as_tensor([row[5] for row in transitions], dtype=torch.float32)
        returns = torch.as_tensor([row[6] for row in transitions], dtype=torch.float32)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / torch.clamp(
            advantages.std(unbiased=False), min=1e-6
        )
        last_loss = float("nan")
        for _ in range(int(config["ppo_epochs"])):
            order = torch.as_tensor(rng.permutation(len(transitions)), dtype=torch.long)
            for start in range(0, len(order), 2048):
                index = order[start : start + 2048]
                logits = model(features[index]).masked_fill(~masks[index], -1e20)
                log_probs = torch.log_softmax(logits, dim=-1)
                selected = log_probs.gather(1, actions[index, None]).squeeze(1)
                ratio = torch.exp(selected - old_log[index])
                unclipped = ratio * advantages[index]
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(config["clip_ratio"]),
                    1.0 + float(config["clip_ratio"]),
                ) * advantages[index]
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
                values = value_model(features[index]).squeeze(1)
                value_loss = torch.mean(torch.square(values - returns[index]))
                probabilities = torch.softmax(logits, dim=-1)
                entropy = -torch.mean(
                    torch.sum(probabilities * log_probs, dim=-1)
                )
                loss = (
                    policy_loss
                    + float(config["value_coefficient"]) * value_loss
                    - float(config["entropy_coefficient"]) * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(value_model.parameters()), 10.0
                )
                optimizer.step()
                last_loss = float(loss.detach())
        record(
            model, step, last_loss,
            {"mean_episode_return": float(returns.mean())},
        )


METHOD_RUNNERS = {
    "best_response_guided_repair": best_response_guided_repair,
    "kl_exploitability_descent": kl_exploitability_descent,
    "neurd_fine_tuning": neurd_fine_tuning,
    "ppo_self_play": ppo_self_play,
}


__all__ = ["METHOD_RUNNERS"]

