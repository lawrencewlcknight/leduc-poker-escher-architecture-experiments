"""Separate predictable TD and collection snapshots, with isolated RNG streams."""
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import random
import time

import numpy as np
import torch

from adaptive_escher.frozen_target_cache import transition_batch, frozen_targets
from unbiased_escher.solver import CrossFittedQMember, CrossFittedQEnsemble
from unbiased_escher.stability import StableUnbiasedControlVariateEscher
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import (
    DiagnosticGroupedWideUCVSolver, _make_solver)
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table, exact_exploitability)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state().clone()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])


def seeded_rng(seed, name):
    outside = capture_rng()
    value = int.from_bytes(hashlib.sha256(f"{seed}/{name}".encode()).digest()[:4], "big")
    try:
        random.seed(value); np.random.seed(value); torch.manual_seed(value)
        return capture_rng()
    finally:
        restore_rng(outside)


@contextmanager
def stream(states, name):
    outside = capture_rng()
    restore_rng(states[name])
    try:
        yield
    finally:
        states[name] = capture_rng()
        restore_rng(outside)


def cpu_weights(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class SplitCritic(CrossFittedQMember):
    def __init__(self, *args, td_window, collection_window, **kwargs):
        self.td_window, self.collection_window = td_window, collection_window
        self.target_history = deque(maxlen=4)
        self.last_target_update_l2 = 0.0
        super().__init__(*args, **kwargs)
        self.collection_model = deepcopy(self.target_model)
        self.cache_frozen_targets = True
        self.fit_seconds = 0.
        self.fit_diagnostics = []

    def weights_for(self, window):
        rows = list(self.target_history)[-window:]
        if not rows:
            return cpu_weights(self.model)
        return {k: torch.stack([r[k] for r in rows]).mean(0) if rows[0][k].is_floating_point()
                else rows[-1][k] for k in rows[0]}

    def install_snapshots(self):
        before = cpu_weights(self.target_model)
        self.target_model.load_state_dict(self.weights_for(self.td_window))
        self.collection_model.load_state_dict(self.weights_for(self.collection_window))
        self.target_model.eval(); self.collection_model.eval()
        self.last_target_update_l2 = sum(float((before[k]-v.cpu()).square().sum())
                                         for k,v in self.target_model.state_dict().items()) ** .5

    def train_model(self, iteration):
        # Existing cached trainer uses only target_model for labels; its final
        # hard copy is replaced before any subsequent collection can observe it.
        started = time.perf_counter()
        before_weights = cpu_weights(self.target_model)
        probe = None
        if len(self.buffer) and (self.batch_size < 0 or len(self.buffer) >= self.batch_size):
            # Deterministic replay rows; identical inputs and next-state policy
            # on both sides of this fit isolate movement of the TD snapshot.
            indices = np.linspace(0, len(self.buffer)-1, min(4096,len(self.buffer)), dtype=int)
            probe = transition_batch(self.buffer, indices, self.device)
            before_labels = frozen_targets(self, probe, iteration)
        loss = super().train_model(iteration)
        if loss is not None:
            self.target_history.append(cpu_weights(self.model))
            self.install_snapshots()
            self.last_target_update_l2 = sum(float((before_weights[k]-v.cpu()).square().sum())
                for k,v in self.target_model.state_dict().items()) ** .5
            after_labels = frozen_targets(self, probe, iteration)
            self.fit_diagnostics.append({"iteration":iteration, "replay_rows":len(self.buffer),
                "probe_rows":len(before_labels), "td_label_update_rms":float((after_labels-before_labels).square().mean().sqrt()),
                "td_parameter_update_l2":self.last_target_update_l2, "final_minibatch_loss":float(loss),
                "cache_build_seconds":self.last_target_cache_stats["cache_build_seconds"],
                "cache_bytes":self.last_target_cache_stats["cache_bytes"]})
        self.fit_seconds += time.perf_counter()-started
        return loss

    def get_baseline(self, state, player):
        x = torch.as_tensor(self.get_history_tensor(state), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.collection_model(x).cpu().numpy()
        return value * (1.0 if player == 0 else -1.0) * np.asarray(state.legal_actions_mask())


class SplitEnsemble(CrossFittedQEnsemble):
    def train_model(self, iteration):
        losses = []
        for i, member in enumerate(self.members):
            with stream(self.streams, f"critic/{i}"):
                losses.append(member.train_model(iteration))
        valid = [x for x in losses if x is not None]
        return float(np.mean(valid)) if valid else None


class MechanismSolver(DiagnosticGroupedWideUCVSolver):
    def __init__(self, *args, critic_td_target_window, critic_collection_control_window,
                 proposal_uniform_mass=.2, **kwargs):
        if critic_td_target_window not in (1,4) or critic_collection_control_window not in (1,4):
            raise ValueError("Only prespecified windows 1 and 4 are supported")
        self.td_window = critic_td_target_window
        self.collection_window = critic_collection_control_window
        self.proposal_uniform_mass = proposal_uniform_mass
        super().__init__(*args, **kwargs)
        self.streams = {name: seeded_rng(kwargs["seed"], name) for name in
                        ("collect/0", "collect/1", "regret/0", "regret/1", "critic/0", "critic/1", "policy")}
        self.q_value_trainer.streams = self.streams
        self.component_seconds = {}

    def init_q_value_trainer(self):
        root = self.game.new_initial_state()
        size = len(np.append(root.information_state_tensor(0), root.information_state_tensor(1)))
        e = SplitEnsemble.__new__(SplitEnsemble)
        e.ensemble_size = self.q_ensemble_size
        e.active_fold = e._active_add_count = 0
        e.members = [SplitCritic(size, self.infostate_size, self.action_size, self.network_layers,
            self.learning_rate, self.baseline_buffer_size // self.q_ensemble_size,
            self.baseline_batch_size, self.baseline_network_train_steps, self.logger,
            self.regret_trainers, self.device, gradient_clip_norm=self.q_gradient_clip_norm,
            td_window=self.td_window, collection_window=self.collection_window)
            for _ in range(self.q_ensemble_size)]
        self.q_value_trainer = e

    def _traverser_sampling_policy(self, *, policy, legal_mask, **kwargs):
        uniform = legal_mask / np.sum(legal_mask)
        p = self.proposal_uniform_mass*uniform + (1-self.proposal_uniform_mass)*policy
        p = np.where(legal_mask > 0, p, 0.)
        return p / p.sum()

    def _record_detailed_estimate_diagnostics(self, **kwargs):
        pass  # Avoid the historical per-visit forensic record allocation.

    def collect_training_data(self, player):
        start = time.perf_counter()
        with stream(self.streams, f"collect/{player}"):
            super().collect_training_data(player)
        self.component_seconds["collection"] = self.component_seconds.get("collection",0.) + time.perf_counter()-start

    def train_regret(self, player):
        start = time.perf_counter()
        with stream(self.streams, f"regret/{player}"):
            super().train_regret(player)
        self.component_seconds["regret_fit"] = self.component_seconds.get("regret_fit",0.) + time.perf_counter()-start

    def iteration(self):
        # Preserve historical start-of-iteration, own-reach-weighted averaging.
        # Cache by information set within the frozen strategy to avoid repeated
        # network calls for histories with the same observable information.
        table = {}
        def policy(state):
            key = (state.current_player(), state.information_state_string())
            if key not in table:
                table[key] = self.regret_trainers[key[0]].get_policy(state, self.num_iteration+1)
            return table[key]
        start = time.perf_counter()
        self.exact_average_strategy.observe_iteration(self.num_iteration+1, policy)
        self.component_seconds["exact_average_accumulation"] = self.component_seconds.get("exact_average_accumulation",0.) + time.perf_counter()-start
        start = time.perf_counter()
        previous_critic_seconds = sum(m.fit_seconds for m in self.q_value_trainer.members)
        StableUnbiasedControlVariateEscher.iteration(self)
        self.component_seconds["critic_fit"] = self.component_seconds.get("critic_fit",0.) + sum(
            m.fit_seconds for m in self.q_value_trainer.members)-previous_critic_seconds
        self.component_seconds["learning_inclusive"] = self.component_seconds.get("learning_inclusive",0.) + time.perf_counter()-start
        self.logger.dump(step=self.num_iteration)

    def fit_policy(self):
        with stream(self.streams, "policy"):
            self.ave_policy_trainer.reset()
            return self.ave_policy_trainer.train_model(self.num_iteration)

    def evaluate_mechanism(self, fit=False):
        start = time.perf_counter()
        row = {"iteration": self.num_iteration, "nodes": self.nodes_touched,
               "trajectories": self.episode,
               "exact_average_exploitability": self.exact_average_strategy.exploitability()}
        if fit:
            fit_start = time.perf_counter()
            self.fit_policy()
            row["policy_fit_seconds"] = time.perf_counter()-fit_start
            table = build_policy_table(self.game, lambda s: self.ave_policy_trainer.action_probabilities(s, probs_as_dict=False))
            row["deployed_exploitability"] = exact_exploitability(self.game, table)
        row["evaluation_seconds"] = time.perf_counter()-start
        return row


def make_solver(seed, config):
    return _make_solver(seed, config, solver_class=MechanismSolver)
