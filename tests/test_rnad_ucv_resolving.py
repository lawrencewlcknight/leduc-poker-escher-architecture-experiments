"""Numerical, information-boundary and deployment invariants for Experiment 49."""

from dataclasses import replace

import numpy as np
import pyspiel
import pytest
import torch
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from experiments.leduc_poker.rnad_ucv_resolving.config import RNaDConfig
from experiments.leduc_poker.rnad_ucv_resolving.evaluate import resolve, save_policy, load_policy
from experiments.leduc_poker.rnad_ucv_resolving.game import Blueprint, PublicProblem, CFRPlus
from experiments.leduc_poker.rnad_ucv_resolving.rnad import (
    NeuralResolver, legal_distribution, nerd_loss, new_value_model, v_trace,
)
from vr_deep_cfr.solver import MLP


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(49)
    game = pyspiel.load_game("leduc_poker")
    blueprint = Blueprint(game, policy.TabularPolicy(game))
    actor = MLP(30, [16, 16], 3)
    value = new_value_model(actor)
    return blueprint, actor, value


def test_legal_distribution_and_nerd_shift_invariance():
    logits = torch.tensor([[.2, -.2, 90.]], requires_grad=True)
    legal = torch.tensor([[True, True, False]])
    probs, log_probs = legal_distribution(logits, legal)
    assert probs[0, 2] == log_probs[0, 2] == 0
    q = torch.tensor([[2., -2., 100.]])
    loss = nerd_loss(logits, probs, q, legal, torch.ones(1), 2., 5.)
    loss.backward()
    assert logits.grad[0, 0] < 0 < logits.grad[0, 1]
    assert logits.grad[0, 2] == 0
    shifted = logits.detach().add(13).requires_grad_()
    p2, _ = legal_distribution(shifted, legal)
    nerd_loss(shifted, p2, q, legal, torch.ones(1), 2., 5.).backward()
    torch.testing.assert_close(shifted.grad, logits.grad)


def test_nerd_threshold_and_continuation_weight():
    logits = torch.tensor([[3., -3.]], requires_grad=True)
    pi = logits.softmax(-1)
    loss = nerd_loss(logits, pi, torch.tensor([[5., -5.]]), torch.ones_like(logits).bool(),
                     torch.ones(1), 2., 5.)
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    a = torch.zeros((1, 2), requires_grad=True)
    q = torch.tensor([[1., -1.]])
    base = nerd_loss(a, a.softmax(-1), q, torch.ones_like(a).bool(), torch.ones(1), 2., 5.)
    scaled = nerd_loss(a, a.softmax(-1), q, torch.ones_like(a).bool(), torch.tensor([.25]), 2., 5.)
    g1 = torch.autograd.grad(base, a, retain_graph=True)[0]
    g2 = torch.autograd.grad(scaled, a)[0]
    torch.testing.assert_close(g2, g1 * .25)


def test_vtrace_one_step_exact_expectation():
    pi = torch.tensor([[[.3, .7], [.3, .7]]])
    log_ratio = torch.log(pi / .5)
    actions = torch.tensor([[0, 1]])
    rewards = torch.tensor([[-2., 3.]])
    targets, q = v_trace(torch.full((1, 2), .7), pi, pi, log_ratio, actions, rewards,
                         torch.zeros((1, 2), dtype=torch.int64), torch.ones((1, 2)).bool(),
                         0, .2, .95)
    expected_q = torch.tensor([-2., 3.]) - .2 * log_ratio[0, 0]
    torch.testing.assert_close(.3 * q[0, 0] + .7 * q[0, 1], expected_q)
    torch.testing.assert_close(.3 * targets[0, 0] + .7 * targets[0, 1],
                               (.3 * expected_q[0] + .7 * expected_q[1]))


def test_vtrace_alternating_player_returns_and_padding():
    probs = torch.ones((3, 1, 1))
    players = torch.tensor([[0], [1], [-1]])
    valid = torch.tensor([[True], [True], [False]])
    rewards = torch.tensor([[0.], [3.], [0.]])
    values = torch.tensor([[.25], [-.9], [0.]])
    vt, q = v_trace(values, probs, probs, torch.zeros_like(probs), torch.zeros((3, 1)).long(),
                    rewards, players, valid, 0, .2, 1.)
    assert vt[:, 0].tolist() == [3., 0., 0.]
    assert q[:, 0, 0].tolist() == [3., 0., 0.]
    vt1, _ = v_trace(values, probs, probs, torch.zeros_like(probs), torch.zeros((3, 1)).long(),
                     -rewards, players, valid, 1, .2, 1.)
    assert vt1[:, 0].tolist() == [0., -3., 0.]


def test_all_roots_and_no_opponent_reach_in_gadget(fixture):
    bp, _, _ = fixture
    assert len(bp.roots) == 30
    altered = bp.as_policy()
    for key, row in altered.state_lookup.items():
        state = altered.states[row]
        if state.current_player() == 1 and "[Round 1]" in key:
            legal = state.legal_actions()
            altered.action_probability_array[row] = 0
            altered.action_probability_array[row, legal[0]] = 1
    other = Blueprint(bp.game, altered)
    assert other.roots.keys() == bp.roots.keys()
    bayesian_changed = False
    for key in bp.roots:
        first, second = PublicProblem(bp, key, 0), PublicProblem(other, key, 0)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.bounds, second.bounds)
        a, b = PublicProblem(bp, key, 0, False), PublicProblem(other, key, 0, False)
        bayesian_changed |= not np.allclose(a.weights, b.weights)
    assert bayesian_changed


def test_information_set_features_ignore_hidden_opponent_cards(fixture):
    bp, _, _ = fixture
    count = 0
    for key in bp.roots:
        problem = PublicProblem(bp, key, 0)
        for history in bp.roots[key]:
            def walk(state):
                nonlocal count
                if state.is_terminal():
                    return
                row = bp.index(state)
                np.testing.assert_array_equal(bp.features[row],
                                               np.asarray(state.information_state_tensor(state.current_player()), dtype=np.float32))
                for action in state.legal_actions():
                    count += 1
                    walk(state.child(action))
            walk(history.state)
        # Repeated hidden histories are represented by one actor/critic input row.
        assert len(problem.nonterminal) > len(problem.global_rows)
    assert count > 1000


@pytest.mark.parametrize("hero", [0, 1])
def test_exact_bound_and_cfr_reduction(fixture, hero):
    bp, _, _ = fixture
    p = PublicProblem(bp, list(bp.roots)[0], hero)
    assert p.bound_violation(p.blueprint) == pytest.approx(0, abs=1e-12)
    solver = CFRPlus(p)
    np.testing.assert_allclose(solver.matching(solver.regrets, p.legal), p.blueprint)
    for _ in range(100):
        solver.step()
    candidate = solver.probabilities()
    np.testing.assert_allclose(candidate.sum(-1), 1)
    assert p.bound_violation(candidate) <= 1e-6


def test_fixed_update_reproducibility_and_frozen_blueprint(fixture):
    bp, actor, value = fixture
    problem = PublicProblem(bp, list(bp.roots)[0], 0)
    frozen = {k: v.clone() for k, v in actor.state_dict().items()}
    config = replace(RNaDConfig(), batch_size=8)
    first = NeuralResolver(problem, actor, value, 13, config)
    second = NeuralResolver(problem, actor, value, 13, config)
    for _ in range(4):
        first.step()
        second.step()
    np.testing.assert_array_equal(first.probabilities(), second.probabilities())
    assert np.max(np.abs(first.probabilities() - problem.blueprint)) > 0
    for name, parameter in actor.state_dict().items():
        torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["cfr_plus_gadget", "rnad_gadget", "rnad_bayesian"])
def test_zero_budget_and_reload(fixture, tmp_path, arm):
    bp, actor, value = fixture
    rows, metadata = resolve(bp, next(iter(bp.roots)), 0, arm, actor, value, 17,
                             RNaDConfig(), seconds=0)
    assert rows == {} and metadata["accepted_updates"] == 0
    path = tmp_path / "policy.npz"
    save_policy(path, bp, bp.probs)
    loaded = load_policy(path, bp)
    np.testing.assert_array_equal(loaded.action_probability_array, bp.probs)
    assert exploitability.nash_conv(bp.game, loaded) == pytest.approx(
        exploitability.nash_conv(bp.game, bp.table), abs=1e-10)


def test_only_hero_is_deployed(fixture):
    bp, actor, value = fixture
    rows, metadata = resolve(bp, next(iter(bp.roots)), 1, "rnad_gadget", actor, value, 17,
                             replace(RNaDConfig(), batch_size=8), updates=2)
    assert rows and all(bp.players[row] == 1 for row in rows)
    assert metadata["accepted_updates"] == 2
    assert metadata["sampled_continuations"] == 16


def test_over_budget_update_is_discarded(fixture, monkeypatch):
    import time
    bp, actor, value = fixture
    original_step = CFRPlus.step

    def slow_step(self):
        original_step(self)
        time.sleep(.03)

    monkeypatch.setattr(CFRPlus, "step", slow_step)
    rows, metadata = resolve(bp, next(iter(bp.roots)), 0, "cfr_plus_gadget", actor, value,
                             0, RNaDConfig(), seconds=.02)
    assert metadata["accepted_updates"] == 0
    assert metadata["fallback"]
    assert metadata["budget_overrun_seconds"] > 0
    for row, probabilities in rows.items():
        np.testing.assert_array_equal(probabilities, bp.probs[row])


def test_bound_is_not_a_clairvoyant_best_response(fixture):
    bp, _, _ = fixture
    strict_gap = False
    for key in bp.roots:
        p = PublicProblem(bp, key, 0)
        v = -p.payoff.copy()  # opponent is player 1
        for node in p.nonterminal[::-1]:
            actions = np.flatnonzero(p.children[node] >= 0)
            q = v[p.children[node, actions]]
            v[node] = (max(q) if p.actors[node] == p.opponent else
                       np.dot(p.blueprint[p.rows[node], actions], q))
        for group, mass in enumerate(p.group_mass):
            if mass > 0:
                chosen = p.groups == group
                clairvoyant = np.dot(p.weights[chosen] / mass, v[p.roots[chosen]])
                assert clairvoyant >= p.bounds[group] - 1e-10
                strict_gap |= clairvoyant > p.bounds[group] + 1e-5
    assert strict_gap
