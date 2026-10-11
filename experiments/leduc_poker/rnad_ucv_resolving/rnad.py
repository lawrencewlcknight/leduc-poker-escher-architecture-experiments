# Adapted portions: Copyright 2019 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0; see LICENSE.openspiel.
"""CPU PyTorch RNaD/NeuRD, with a fixed blueprint regularizer.

The alternating-player V-trace recurrence and NeuRD force follow OpenSpiel's
RNaD implementation at 931e39a (Apache-2.0); see NOTICE.md and LICENSE.openspiel.
This is an independent, small-game adaptation, not the 2026 authors' code.
"""

import copy
import time

import numpy as np
import torch
from torch import nn

from .config import RNaDConfig


def legal_distribution(logits, legal):
    log_policy = logits.masked_fill(~legal, -torch.inf).log_softmax(dim=-1)
    return log_policy.exp(), log_policy.masked_fill(~legal, 0.0)


def nerd_loss(logits, probabilities, q_values, legal, weights, threshold, clip):
    """NeuRD, NOT ordinary policy gradient or a PPO/KL surrogate.

    Center logits across legal actions; stop the force at the logit threshold.
    The continuation weight is NOT renormalized away.
    """
    advantage = q_values - (probabilities * q_values).sum(-1, keepdim=True)
    advantage = advantage.detach().clamp(-clip, clip)
    centered = logits - (logits * legal).sum(-1, keepdim=True) / legal.sum(-1, keepdim=True)
    force = (advantage.clamp(max=0) * (centered.detach() > -threshold)
             + advantage.clamp(min=0) * (centered.detach() < threshold))
    return -(legal * centered * force.detach()).sum(-1).mul(weights).sum() / max(1, len(weights))


@torch.no_grad()
def v_trace(values, probabilities, behavior, log_ratio, actions, rewards,
            players, valid, target_player, eta, trace_lambda):
    """Alternating-player RNaD value and action-value targets, gamma=1.

    Opponent turns are integrated in the reverse recurrence. Simply applying
    single-agent TD to alternating-player value predictions would be incorrect.
    On-policy collection gives unit ratios, but the full correction is retained.
    """
    t_count, batch = valid.shape
    selected_pi = probabilities.gather(-1, actions[..., None]).squeeze(-1)
    selected_mu = behavior.gather(-1, actions[..., None]).squeeze(-1)
    ratio = selected_pi / selected_mu.clamp_min(1e-30)
    sign = torch.where(players == target_player, 1.0, -1.0)
    entropy = -eta * (probabilities * log_ratio).sum(-1) * sign
    log_term = -eta * log_ratio * sign[..., None]
    reward_carry = torch.zeros(batch)
    reward_uncorrected = torch.zeros(batch)
    next_value = torch.zeros(batch)
    next_target = torch.zeros(batch)
    importance = torch.ones(batch)
    targets = torch.zeros_like(values)
    q_targets = torch.zeros_like(probabilities)
    for t in range(t_count - 1, -1, -1):
        r_uncorrected = rewards[t] + reward_uncorrected + entropy[t]
        r_discounted = rewards[t] + reward_carry
        joint_ratio = ratio[t] * importance
        vt = (values[t] + joint_ratio * (r_uncorrected + next_value - values[t])
              + trace_lambda * joint_ratio.clamp(max=1) * (next_target - next_value))
        q = values[t, :, None] + log_term[t]
        correction = (r_discounted + importance * next_target - values[t]) / selected_mu[t].clamp_min(1e-30)
        q = q.scatter_add(-1, actions[t, :, None], correction[:, None])
        ours = valid[t] & (players[t] == target_player)
        theirs = valid[t] & ~ours
        targets[t] = torch.where(ours, vt, 0.0)
        q_targets[t] = torch.where(ours[:, None], q, 0.0)
        reward_carry = torch.where(theirs, entropy[t] + ratio[t] * r_discounted, 0.0)
        reward_uncorrected = torch.where(theirs, r_uncorrected, 0.0)
        next_value = torch.where(ours, values[t], torch.where(theirs, next_value, 0.0))
        next_target = torch.where(ours, vt, torch.where(theirs, next_target, 0.0))
        importance = torch.where(theirs, ratio[t] * importance, 1.0)
    return targets, q_targets


def new_value_model(actor):
    """Transfer the actor's features, not the privileged UCV training critic."""
    value = copy.deepcopy(actor)
    last = value.layers[-1]
    value.layers[-1] = nn.Linear(last.weight.shape[1], 1)
    nn.init.zeros_(value.layers[-1].weight)
    nn.init.zeros_(value.layers[-1].bias)
    return value


def prepare_value_model(blueprint, actor, seed, config=RNaDConfig()):
    """Offline on-policy MC calibration; whole hands held out, no BR labels.

    The preparation cost is recorded separately, like loading the blueprint.
    All public-root solves start from exactly this frozen critic.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    train_rows, train_y, test_rows, test_y = [], [], [], []
    for hand in range(config.critic_pretrain_hands + config.critic_validation_hands):
        state = blueprint.game.new_initial_state()
        visited = []
        while not state.is_terminal():
            if state.is_chance_node():
                actions, probabilities = zip(*state.chance_outcomes())
                action = rng.choice(actions, p=probabilities)
            else:
                row = blueprint.index(state)
                visited.append((row, state.current_player()))
                probabilities = blueprint.probs[row]
                action = rng.choice(len(probabilities), p=probabilities / probabilities.sum())
            state.apply_action(int(action))
        dest_rows, dest_y = ((train_rows, train_y) if hand < config.critic_pretrain_hands
                             else (test_rows, test_y))
        for row, player in visited:
            dest_rows.append(row)
            dest_y.append(state.returns()[player])
    x = torch.from_numpy(blueprint.features)
    rows = np.asarray(train_rows)
    targets = torch.tensor(train_y, dtype=torch.float32)
    value = new_value_model(actor)
    optimizer = torch.optim.Adam(value.parameters(), lr=config.critic_learning_rate,
                                 betas=(config.adam_beta1, config.adam_beta2))
    for _ in range(config.critic_pretrain_updates):
        choice = rng.integers(len(rows), size=config.batch_size)
        predicted = value(x[rows[choice]]).squeeze(-1)
        loss = (predicted - targets[choice]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    value.eval()
    with torch.no_grad():
        validation = value(x[test_rows]).squeeze(-1).numpy()
    return value, {
        "preparation_seconds": time.perf_counter() - started,
        "training_hands": config.critic_pretrain_hands,
        "validation_hands": config.critic_validation_hands,
        "updates": config.critic_pretrain_updates,
        "training_information_sets_observed": len(set(train_rows)),
        "validation_mc_mse": float(np.mean((validation - test_y) ** 2)),
        "validation_zero_value_mse": float(np.mean(np.asarray(test_y) ** 2)),
        "labels": "on-policy Monte Carlo returns, separate validation hands",
    }


class NeuralResolver:
    def __init__(self, problem, actor, value, seed, config=RNaDConfig()):
        self.problem, self.config = problem, config
        self.rng = np.random.default_rng(seed)
        # No dropout and no randomized reinitialization: each solve resets exactly.
        self.actor = copy.deepcopy(actor).train()
        self.value = copy.deepcopy(value).train()
        self.target = copy.deepcopy(value).eval()
        self.features = torch.from_numpy(problem.features)
        self.legal = torch.from_numpy(problem.legal)
        with torch.no_grad():
            _, self.magnet_log = legal_distribution(actor(self.features), self.legal)
        betas = (config.adam_beta1, config.adam_beta2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate,
                                               betas=betas)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.critic_learning_rate,
                                               betas=betas)
        self.gadget_logits = nn.Parameter(torch.zeros((len(problem.group_keys), 2)))
        self.gadget_optimizer = torch.optim.Adam([self.gadget_logits],
                                                lr=config.gadget_learning_rate, betas=betas)
        self.iterations = 0
        self.samples = 0
        self.decisions = 0

    def sample(self, probabilities):
        p = self.problem
        selected = self.rng.choice(len(p.roots), self.config.batch_size, p=p.weights)
        nodes = p.roots[selected].copy()
        rows, players, actions, valid, rewards = [], [], [], [], []
        while not p.terminal[nodes].all():
            live = ~p.terminal[nodes]
            row = np.maximum(p.rows[nodes], 0)
            # Uniforms only for live trajectories; padded entries never enter losses.
            chosen = np.zeros(len(nodes), dtype=int)
            cdf = np.cumsum(probabilities[row[live]], axis=1)
            cdf[:, -1] = 1.0
            chosen[live] = (self.rng.random((live.sum(), 1)) > cdf).sum(axis=1)
            rows.append(row.copy())
            players.append(p.actors[nodes].copy())
            actions.append(chosen)
            valid.append(live.copy())
            nodes[live] = p.children[nodes[live], chosen[live]]
            if np.min(nodes) < 0:
                raise ValueError("Sampled an illegal action")
            rewards.append(np.where(live & p.terminal[nodes], p.payoff[nodes], 0.0))
            if len(rows) > 16:
                raise ValueError("Unexpected Leduc continuation length")
        self.samples += len(selected)
        self.decisions += int(np.asarray(valid).sum())
        return (torch.tensor(np.asarray(rows)), torch.tensor(np.asarray(players)),
                torch.tensor(np.asarray(actions)), torch.tensor(np.asarray(valid)),
                torch.tensor(np.asarray(rewards), dtype=torch.float32),
                torch.tensor(p.groups[selected]))

    def step(self):
        c, p = self.config, self.problem
        with torch.no_grad():
            behavior, _ = legal_distribution(self.actor(self.features), self.legal)
        rows, players, actions, valid, rewards0, groups = self.sample(behavior.numpy())
        logits_all = self.actor(self.features)
        pi_all, logs_all = legal_distribution(logits_all, self.legal)
        pi = pi_all[rows]
        log_ratio = logs_all[rows] - self.magnet_log[rows]
        with torch.no_grad():
            v = self.target(self.features).squeeze(-1)[rows]
            continuation = (self.gadget_logits.softmax(-1)[groups, 1] if p.gadget
                            else torch.ones(len(groups)))
        value_predictions = self.value(self.features).squeeze(-1)[rows]
        actor_loss = logits_all.sum() * 0.0
        critic_loss = value_predictions.sum() * 0.0
        for player in (0, 1):
            targets, q = v_trace(v, pi.detach(), behavior[rows], log_ratio.detach(), actions,
                                rewards0 * (1 if player == 0 else -1), players, valid,
                                player, c.eta, c.trace_lambda)
            mask = valid & (players == player)
            weights = continuation.expand_as(valid)[mask]
            actor_loss = actor_loss + nerd_loss(
                logits_all[rows][mask], pi[mask], q[mask], self.legal[rows][mask],
                weights, c.logit_threshold, c.advantage_clip)
            critic_loss = critic_loss + ((value_predictions[mask] - targets[mask]).square()
                                         * weights).sum() / max(1, int(mask.sum()))
        if not torch.isfinite(actor_loss + critic_loss):
            raise FloatingPointError("Non-finite RNaD objective")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.value_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.value_optimizer.step()
        if p.gadget:
            # Integrate the termination branch exactly, sample the continuation.
            # The opt-out decision depends on opponent information only.
            with torch.no_grad():
                sign = torch.where(players == p.opponent, 1.0, -1.0)
                selected_log = log_ratio.detach().gather(-1, actions[..., None]).squeeze(-1)
                shaped = (rewards0 * (1 if p.opponent == 0 else -1)
                          - c.eta * sign * selected_log) * valid
                q_gadget = torch.stack((torch.tensor(p.bounds, dtype=torch.float32)[groups],
                                        shaped.sum(0)), -1)
            logits = self.gadget_logits[groups]
            loss = nerd_loss(logits, logits.softmax(-1), q_gadget, torch.ones_like(logits).bool(),
                             torch.ones(len(groups)), c.logit_threshold, c.advantage_clip)
            self.gadget_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.gadget_optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), self.value.parameters()):
                target.lerp_(source, c.target_rate)
        self.iterations += 1

    @torch.no_grad()
    def probabilities(self):
        result, _ = legal_distribution(self.actor(self.features), self.legal)
        result = result.double().numpy()
        return result / result.sum(axis=1, keepdims=True)
