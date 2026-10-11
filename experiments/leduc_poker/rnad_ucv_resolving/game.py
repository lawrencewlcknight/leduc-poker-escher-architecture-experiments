"""Public-range construction and an explicit opt-out gadget for tiny Leduc.

No solve is given the actual private deal. Each problem contains *all* compatible
deals. Gadget chance weights exclude the opponent's blueprint action reach.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from open_spiel.python import policy

from escher_poker.live_resolving import is_second_round_root, public_root_key


def check_probabilities(values, legal):
    values = np.asarray(values, dtype=np.float64)
    if (not np.isfinite(values).all() or np.min(values) < -1e-10
            or np.max(np.abs(values[~legal])) > 1e-10
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-7)):
        raise ValueError("Invalid, non-finite or illegal policy probabilities")


@dataclass
class History:
    state: object
    chance: float
    reach: np.ndarray


class Blueprint:
    """Immutable tabulation plus exhaustive public roots (including zero reach)."""

    def __init__(self, game, source):
        self.game = game
        self.table = policy.tabular_policy_from_callable(game, source.action_probabilities)
        self.keys = list(self.table.state_lookup)
        if [self.table.state_lookup[k] for k in self.keys] != list(range(len(self.keys))):
            raise ValueError("Unexpected OpenSpiel tabular ordering")
        self.probs = self.table.action_probability_array.copy()
        self.legal = self.table.legal_actions_mask.astype(bool)
        check_probabilities(self.probs, self.legal)
        self.features = np.asarray([
            s.information_state_tensor(s.current_player()) for s in self.table.states
        ], dtype=np.float32)
        self.players = np.asarray([s.current_player() for s in self.table.states])
        roots = defaultdict(list)

        def walk(state, chance, reach):
            if state.is_terminal():
                return
            if is_second_round_root(state):
                roots[public_root_key(state)].append(History(state.clone(), chance, reach.copy()))
                return
            if state.is_chance_node():
                for action, probability in state.chance_outcomes():
                    walk(state.child(action), chance * probability, reach)
                return
            player = state.current_player()
            row = self.probs[self.index(state)]
            for action in state.legal_actions():
                child_reach = reach.copy()
                child_reach[player] *= row[action]
                # Enumerate even zero-probability actions: a best response can take them.
                walk(state.child(action), chance, child_reach)

        walk(game.new_initial_state(), 1.0, np.ones(2))
        self.roots = dict(sorted(roots.items()))
        self.probs.setflags(write=False)

    def index(self, state):
        return self.table.state_lookup[state.information_state_string(state.current_player())]

    def as_policy(self, probabilities=None):
        result = policy.TabularPolicy(self.game)
        values = self.probs if probabilities is None else probabilities
        check_probabilities(values, self.legal)
        result.action_probability_array[:] = values
        return result


class PublicProblem:
    """Flat, preorder game forest; row indices refer only to information sets."""

    def __init__(self, blueprint, root_key, hero, gadget=True):
        self.hero, self.opponent = hero, 1 - hero
        self.gadget = gadget
        histories = blueprint.roots[root_key]
        weights = np.asarray([
            h.chance * (h.reach[hero] if gadget else np.prod(h.reach)) for h in histories
        ])
        self.mass = float(weights.sum())
        # Zero counterfactual reach: return blueprint, never invent a posterior.
        self.weights = weights / self.mass if self.mass > 0 else weights
        group_keys = [h.state.information_state_string(self.opponent) for h in histories]
        self.group_keys = sorted(set(group_keys))
        self.groups = np.asarray([self.group_keys.index(k) for k in group_keys])
        self.group_mass = np.bincount(self.groups, weights=self.weights,
                                      minlength=len(self.group_keys))
        children, actors, rows, payoff, root_history = [], [], [], [], []

        def expand(state, history_index):
            n = len(children)
            children.append([-1, -1, -1])
            root_history.append(history_index)
            if state.is_terminal():
                actors.append(-1)
                rows.append(-1)
                payoff.append(float(state.returns()[0]))
            else:
                if state.is_chance_node():
                    raise ValueError("Unexpected chance event after Leduc public card")
                actors.append(state.current_player())
                rows.append(blueprint.index(state))
                payoff.append(0.0)
                for action in state.legal_actions():
                    children[n][action] = expand(state.child(action), history_index)
            return n

        self.roots = np.asarray([expand(h.state, i) for i, h in enumerate(histories)])
        self.global_rows = np.asarray(sorted(set(rows) - {-1}))
        local = {int(v): i for i, v in enumerate(self.global_rows)}
        self.rows = np.asarray([local.get(r, -1) for r in rows])
        self.actors = np.asarray(actors)
        self.children = np.asarray(children)
        self.payoff = np.asarray(payoff)
        self.root_history = np.asarray(root_history)
        self.nonterminal = np.flatnonzero(self.actors >= 0)
        self.terminal = self.actors < 0
        self.features = blueprint.features[self.global_rows].copy()
        self.legal = blueprint.legal[self.global_rows].copy()
        self.row_players = blueprint.players[self.global_rows].copy()
        self.blueprint = blueprint.probs[self.global_rows].copy()
        self.bounds = np.zeros(len(self.group_keys))
        if gadget and self.mass > 0:
            for group, mass in enumerate(self.group_mass):
                selected = self.groups == group
                if mass > 0:
                    self.bounds[group] = self.best_response(
                        self.roots[selected], self.weights[selected] / mass, self.blueprint
                    )

    def best_response(self, nodes, weights, probabilities):
        """Opponent BR aggregated BEFORE maximisation; never perfect-information BR.

        All nodes share public betting history and the opponent's private information.
        Hero's differing private cards therefore cannot change the opponent's action.
        """
        if self.terminal[nodes].all():
            sign = 1 if self.opponent == 0 else -1
            return float(np.dot(weights, self.payoff[nodes]) * sign)
        player = self.actors[nodes[0]]
        if player < 0 or not np.all(self.actors[nodes] == player):
            raise ValueError("Inconsistent public frontier")
        actions = np.flatnonzero(self.children[nodes[0]] >= 0)
        if player == self.opponent:
            if len(set(self.rows[nodes])) != 1:
                raise ValueError("BR frontier splits opponent information sets")
            return max(self.best_response(self.children[nodes, a], weights, probabilities)
                       for a in actions)
        return sum(self.best_response(self.children[nodes, a],
                                      weights * probabilities[self.rows[nodes], a], probabilities)
                   for a in actions)

    def values(self, probabilities):
        """Unregularized player-zero continuation values at every history."""
        result = self.payoff.copy()
        for n in self.nonterminal[::-1]:
            actions = np.flatnonzero(self.children[n] >= 0)
            result[n] = np.dot(probabilities[self.rows[n], actions],
                               result[self.children[n, actions]])
        return result

    def bound_violation(self, candidate):
        """Diagnostic only. Never reject/select a deployed policy using this oracle."""
        excess = []
        for g, mass in enumerate(self.group_mass):
            if mass > 0:
                chosen = self.groups == g
                response = self.best_response(self.roots[chosen], self.weights[chosen] / mass,
                                              candidate)
                excess.append(response - self.bounds[g])
        return max(excess, default=0.0)


class CFRPlus:
    """Alternating full-tree CFR+ with linear own-reach-weighted averaging."""

    def __init__(self, problem):
        self.problem = problem
        # A fixed unit-mass positive pseudo-regret makes the initial strategy
        # exactly the blueprint, rather than throwing it away for uniform play.
        # This is an initialization only, not a continuing regularization term.
        self.regrets = problem.blueprint.copy()
        self.average = np.zeros_like(problem.blueprint)
        self.gadget_regrets = np.zeros((len(problem.group_keys), 2))
        self.iterations = 0

    @staticmethod
    def matching(regrets, legal):
        positive = np.maximum(regrets, 0) * legal
        total = positive.sum(axis=-1, keepdims=True)
        return np.divide(positive, total, out=legal / legal.sum(axis=-1, keepdims=True),
                         where=total > 0)

    def step(self):
        p = self.problem
        self.iterations += 1
        for player in (0, 1):
            strategy = self.matching(self.regrets, p.legal)
            gadget = self.matching(self.gadget_regrets, np.ones_like(self.gadget_regrets))
            values = p.values(strategy)
            reaches = np.ones((len(p.rows), 2))
            if p.gadget:
                reaches[p.roots, p.opponent] = gadget[p.groups, 1]
            for n in p.nonterminal:
                for action in np.flatnonzero(p.children[n] >= 0):
                    child = p.children[n, action]
                    reaches[child] = reaches[n]
                    reaches[child, p.actors[n]] *= strategy[p.rows[n], action]
            delta = np.zeros_like(strategy)
            sign = 1 if player == 0 else -1
            for n in p.nonterminal[p.actors[p.nonterminal] == player]:
                row = p.rows[n]
                actions = np.flatnonzero(p.children[n] >= 0)
                chance = p.weights[p.root_history[n]]
                delta[row, actions] += chance * reaches[n, 1 - player] * sign * (
                    values[p.children[n, actions]] - values[n])
                self.average[row] += (self.iterations * chance * reaches[n, player]
                                      * strategy[row])
            self.regrets = np.maximum(self.regrets + delta, 0)
            if p.gadget and player == p.opponent:
                cont = np.bincount(p.groups, weights=p.weights * sign * values[p.roots],
                                   minlength=len(p.group_keys))
                q = np.stack((p.group_mass * p.bounds, cont), axis=1)
                v = np.sum(q * gadget, axis=1, keepdims=True)
                self.gadget_regrets = np.maximum(self.gadget_regrets + q - v, 0)

    def probabilities(self):
        total = self.average.sum(axis=1, keepdims=True)
        return np.divide(self.average, total, out=self.problem.blueprint.copy(), where=total > 0)
