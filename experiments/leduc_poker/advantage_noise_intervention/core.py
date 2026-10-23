"""Frozen collection, exact population references and paired native regret fits."""
from collections import defaultdict
from copy import copy, deepcopy
import hashlib
import math
import random
import time

import numpy as np
import torch

from unbiased_escher.estimator import control_variate_advantage
from experiments.leduc_poker.ucv_exact_tabular_validation.diagnostics import (
    ExactUCVOracle, policy_table_for_mode, _history_key, _model_digest)
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table, exact_exploitability)
from experiments.leduc_poker.critic_averaging_mechanism.solver import capture_rng, restore_rng


class FrozenOracle(ExactUCVOracle):
    def __init__(self, solver, table, fold):
        super().__init__(solver, policy_table=table, variant_id="fixed_beta_one", fold=fold)

    def controls(self, state, traverser):
        previous = self.solver.q_value_trainer.active_fold
        self.solver.q_value_trainer.active_fold = self.fold
        try:
            return super().controls(state, traverser)
        finally:
            self.solver.q_value_trainer.active_fold = previous

    def mean_advantage(self, state, player):
        q = self.exact_q(state, player)
        mask = np.asarray(state.legal_actions_mask(), dtype=float)
        return (q - self.policy(state) @ q) * mask

    def vector_moments(self, state, player):
        """Full conditional covariance, including correlations between actions."""
        controls = self.controls(state, player)
        mean = np.zeros(self.action_size)
        second = np.zeros((self.action_size, self.action_size))
        for action in state.legal_actions():
            probability = controls.sampling_policy[action]
            if probability <= 0:
                continue
            child = self.estimator_return_moments(state.child(action), player)
            kwargs = dict(beta=controls.beta, sampled_action=action,
                          sample_probability=probability, policy=controls.policy,
                          legal_actions_mask=state.legal_actions_mask())
            zero = control_variate_advantage(controls.q_hat, sampled_return=0., **kwargs).advantages
            one = control_variate_advantage(controls.q_hat, sampled_return=1., **kwargs).advantages
            slope = one - zero
            conditional = zero + slope * child.mean
            mean += probability * conditional
            second += probability * (np.outer(conditional, conditional) + child.variance * np.outer(slope, slope))
        return mean, (second - np.outer(mean, mean))


def inference_digest(solver):
    states = []
    for player, trainer in enumerate(solver.regret_trainers):
        states.extend([(f"regret/{player}", trainer.model.state_dict()),
                       (f"previous/{player}", trainer.target_model.state_dict())])
    for fold, member in enumerate(solver.q_value_trainer.members):
        states.append((f"critic/{fold}", member.target_model.state_dict()))
    states.append(("calibration", solver.calibration_trainer.target_model.state_dict()))
    return _model_digest(states)


class FrozenProblem:
    def __init__(self, solver):
        self.solver = solver
        self.iteration = solver.num_iteration  # Already advanced to t+1 by source loader.
        self.table = policy_table_for_mode(solver, "current")
        self.oracles = [FrozenOracle(solver, self.table, f) for f in range(2)]
        self.states = {}
        self.references = {}
        self.audit = {}
        self.initial_digest = inference_digest(solver)
        self.initial_exploitability = exact_exploitability(solver.game, self.table)
        self._build_reference()

    def _build_reference(self):
        # Each player collects equal even numbers of trajectories from each fold.
        # Expected row frequency is sampling reach, not counterfactual reach or
        # uniform hidden-history weighting. Retain only positive-reach support.
        for player in range(2):
            grouped = defaultdict(lambda: {"mass": 0., "sum": np.zeros(self.solver.action_size)})
            max_bias = 0.; max_centering = 0.; min_eigenvalue = 0.
            total_cov = np.zeros((self.solver.action_size, self.solver.action_size))
            total_mass = 0.; history_rows = 0
            for oracle in self.oracles:
                oracle.estimator_return_moments(self.solver.game.new_initial_state(), player)

                def walk(state, reach):
                    nonlocal max_bias, max_centering, min_eigenvalue, total_cov, total_mass, history_rows
                    if state.is_terminal():
                        return
                    if state.is_chance_node():
                        for action, probability in state.chance_outcomes():
                            walk(state.child(action), reach * probability)
                        return
                    key = (state.current_player(), state.information_state_string())
                    self.states.setdefault(key, state.clone())
                    controls = oracle.controls(state, player)
                    if state.current_player() == player:
                        truth = oracle.mean_advantage(state, player)
                        mean, covariance = oracle.vector_moments(state, player)
                        max_bias = max(max_bias, float(np.max(np.abs(mean-truth))))
                        max_centering = max(max_centering, abs(float(controls.policy @ truth)),
                                            float(np.max(np.abs(covariance @ controls.policy))))
                        min_eigenvalue = min(min_eigenvalue, float(np.linalg.eigvalsh(covariance).min()))
                        grouped[key]["mass"] += reach
                        grouped[key]["sum"] += reach * truth
                        total_cov += reach * covariance
                        total_mass += reach
                        history_rows += 1
                    for action in state.legal_actions():
                        probability = controls.sampling_policy[action]
                        if probability > 0:
                            walk(state.child(action), reach * probability)

                walk(self.solver.game.new_initial_state(), .5)
            if max_bias > 1e-9 or max_centering > 1e-9 or min_eigenvalue < -1e-9:
                raise ArithmeticError("Frozen estimator mean/covariance audit failed")
            covariance = total_cov / total_mass
            self.audit[player] = {"history_fold_rows": history_rows, "max_mean_error": max_bias,
                "max_centering_error": max_centering, "minimum_covariance_eigenvalue": min_eigenvalue,
                "mean_within_history_covariance": covariance.tolist(),
                "expected_covariance_by_gamma": {str(g): (g*g*covariance).tolist() for g in (1., .5, 0.)}}
            trainer = self.solver.regret_trainers[player]
            reference = []
            for key, entry in sorted(grouped.items()):
                state = self.states[key]
                mask = np.asarray(state.legal_actions_mask(), dtype=float)
                features = np.asarray(self.solver.get_infostate_tensor(state))
                # Match the native float32 operation order for the fixed offset.
                # Only the oracle advantage expectation is evaluated in float64.
                with torch.no_grad():
                    previous = trainer.predict(trainer.target_model,
                        torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(mask, dtype=torch.float32))
                    age = math.pow(self.iteration-1, trainer.alpha)
                    offset = (torch.clamp(previous, min=0.) * age / (age+1.5)).cpu().numpy().astype(np.float64)
                mu = entry["sum"] / entry["mass"]
                target = offset + mu
                policy = trainer.regret_matching(target, state.legal_actions())
                reference.append({"key": key, "features": features, "mask": mask, "mu": mu,
                    "target": target, "policy": policy, "weight": entry["mass"] / total_mass})
            rms = math.sqrt(sum(r["weight"] * np.sum(r["mu"]**2) for r in reference) /
                            sum(r["weight"] * np.sum(r["mask"]) for r in reference))
            eta = .01 * rms
            for row in reference:
                actions = np.flatnonzero(row["mask"]).tolist()
                changes = []
                for action in actions:
                    for sign in (-1, 1):
                        perturbed = row["target"].copy(); perturbed[action] += sign * eta
                        policy = trainer.regret_matching(perturbed, actions)
                        changes.append(.5 * np.abs(policy-row["policy"]).sum())
                row["sensitivity"] = float(np.mean(changes))
                row["sensitive"] = False
            # Stable, outcome-independent tie breaking by information-state key.
            ordered = sorted(reference, key=lambda r: (-r["sensitivity"], r["key"]))
            for row in ordered[:math.ceil(len(ordered)/4)]:
                row["sensitive"] = True
            self.audit[player].update(eta=eta, positive_sensitivity_count=sum(r["sensitivity"] > 0 for r in reference),
                                      supported_infosets=len(reference))
            self.references[player] = reference
        self.assert_frozen()

    def assert_frozen(self):
        if inference_digest(self.solver) != self.initial_digest:
            raise RuntimeError("Frozen inference inputs changed")

    def collect(self, player, trajectories, seed):
        """Native sampling/return recursion, omitting passive replay side effects."""
        if trajectories % 2:
            raise ValueError("An even trajectory count is required for the exact fold mixture")
        rng = capture_rng(); np.random.seed(seed)
        rows = []

        def visit(state, oracle, trajectory):
            if state.is_terminal():
                return state.returns()[player] / self.solver.max_utility
            if state.is_chance_node():
                actions, probs = zip(*state.chance_outcomes())
                index = int(np.random.choice(range(len(actions)), p=probs))
                return visit(state.child(actions[index]), oracle, trajectory)
            controls = oracle.controls(state, player)
            action = int(np.random.choice(range(state.num_distinct_actions()), p=controls.sampling_policy))
            value = visit(state.child(action), oracle, trajectory)
            estimate = control_variate_advantage(controls.q_hat, beta=controls.beta,
                sampled_action=action, sample_probability=float(controls.sampling_policy[action]),
                sampled_return=value, policy=controls.policy, legal_actions_mask=state.legal_actions_mask())
            if state.current_player() == player:
                # Labels are transformed later; recursive returns always use gamma=1.
                rows.append((np.asarray(self.solver.get_infostate_tensor(state)),
                    estimate.advantages.copy(), oracle.mean_advantage(state, player),
                    np.asarray(state.legal_actions_mask(), dtype=float), controls.policy.copy(),
                    trajectory, oracle.fold, ",".join(map(str, _history_key(state)))))
            return estimate.policy_value
        try:
            for trajectory in range(1, trajectories+1):
                fold = (self.solver.episode + trajectory) % 2
                visit(self.solver.game.new_initial_state(), self.oracles[fold], trajectory)
        finally:
            restore_rng(rng)
        if not rows:
            raise RuntimeError("No regret observations collected")
        names = ("features", "sampled", "mean", "mask", "policy", "trajectory", "fold", "history")
        panel = {name: np.asarray([r[i] for r in rows]) for i, name in enumerate(names)}
        self.assert_frozen()
        return panel


def transform(panel, gamma):
    if gamma not in (0., .5, 1.):
        raise ValueError("Unknown intervention")
    # Identity must preserve the ordinary labels bit for bit.
    result = panel["sampled"].copy() if gamma == 1. else panel["mean"] + gamma * (panel["sampled"]-panel["mean"])
    if not np.isfinite(result).all() or np.max(np.abs(result*(1-panel["mask"]))) > 1e-12:
        raise ArithmeticError("Invalid advantage labels")
    if np.max(np.abs(np.sum(result*panel["policy"], axis=1))) > 1e-9:
        raise ArithmeticError("Uncentered advantage labels")
    return result


class PanelBuffer:
    """Native without-replacement minibatching from an immutable collected panel."""
    def __init__(self, panel, labels, iteration):
        self.panel = panel
        self.labels = labels
        self.iteration = iteration
        self.index_digest = hashlib.sha256()
        self.calls = 0

    def sample(self, count):
        n = len(self.labels)
        indices = list(range(n)) if count == -1 or count > n else random.sample(range(n), count)
        self.index_digest.update(np.asarray(indices, dtype=np.int64).tobytes())
        self.calls += 1
        return tuple(torch.as_tensor(a, dtype=torch.float32) for a in (
            self.panel["features"][indices], self.labels[indices], self.panel["mask"][indices],
            np.full((len(indices), 1), self.iteration)))


def fit(problem, player, panel, gamma, steps, batch_size, seed):
    original = problem.solver.regret_trainers[player]
    trainer = copy(original)
    trainer.model = deepcopy(original.model)
    trainer.target_model = deepcopy(original.target_model)
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=original.learning_rate)
    trainer.optimizer.load_state_dict(deepcopy(original.optimizer.state_dict()))
    trainer.buffer = PanelBuffer(panel, transform(panel, gamma), problem.iteration)
    trainer.train_steps = steps; trainer.batch_size = batch_size
    rng = capture_rng()
    before = _model_digest([("target", trainer.target_model.state_dict())])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    started = time.perf_counter()
    try:
        loss = trainer.train_model(problem.iteration)
    finally:
        restore_rng(rng)
    if trainer.buffer.calls != steps or not np.isfinite(loss):
        raise RuntimeError("Invalid native fit")
    problem.assert_frozen()
    return trainer, {"fit_seconds": time.perf_counter()-started, "optimizer_steps": steps,
                     "minibatch_sha256": trainer.buffer.index_digest.hexdigest(),
                     "previous_target_sha256": before, "final_training_loss": float(loss)}


def evaluate_fit(problem, player, trainer):
    reference = problem.references[player]
    x = np.stack([r["features"] for r in reference]); mask = np.stack([r["mask"] for r in reference])
    output = trainer.forward(trainer.model, x, mask)
    if not np.isfinite(output).all():
        raise ArithmeticError("Nonfinite regret prediction")
    details = []
    for row, prediction in zip(reference, output):
        policy = trainer.regret_matching(prediction, np.flatnonzero(row["mask"]).tolist())
        details.append({"information_state": row["key"][1], "player": player,
            "weight": row["weight"], "sensitive": row["sensitive"], "sensitivity": row["sensitivity"],
            "regret_squared_error": float(np.sum((prediction-row["target"])**2)),
            "legal_actions": int(row["mask"].sum()), "policy_tv": float(.5*np.abs(policy-row["policy"]).sum())})
    # Same fixed-output-dimension normalization as native masked nn.MSELoss.
    mse = sum(r["weight"]*r["regret_squared_error"] for r in details)/problem.solver.action_size
    tv = sum(r["weight"]*r["policy_tv"] for r in details)
    sensitive = [r for r in details if r["sensitive"]]
    other = [r for r in details if not r["sensitive"]]
    group_tv = lambda group: sum(r["weight"]*r["policy_tv"] for r in group)/sum(r["weight"] for r in group)
    # Includes zero-collection-reach infosets too, using the fitted network there.
    table = build_policy_table(problem.solver.game, lambda state:
        trainer.get_policy(state, problem.iteration) if state.current_player() == player
        else problem.table[(state.current_player(), state.information_state_string())])
    return {"exact_regret_mse": float(mse), "policy_tv": float(tv),
            "sensitive_policy_tv": float(group_tv(sensitive)), "other_policy_tv": float(group_tv(other)),
            "sensitivity_tv_gap": float(group_tv(sensitive)-group_tv(other)),
            "hybrid_exploitability": exact_exploitability(problem.solver.game, table)}, details, table
