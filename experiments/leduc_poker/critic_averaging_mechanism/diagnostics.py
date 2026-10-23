"""Exact frozen-policy diagnostics; no diagnostic samples enter learning."""
import numpy as np

from experiments.leduc_poker.ucv_exact_tabular_validation.diagnostics import (
    ExactUCVOracle, StateControls, policy_table_for_mode)


class MechanismOracle(ExactUCVOracle):
    def __init__(self, solver, table, fold, window):
        super().__init__(solver, policy_table=table,
                         variant_id="residual_calibration_disabled", fold=fold)
        self.window = window
        self.models = []
        from copy import deepcopy
        for member in solver.q_value_trainer.members:
            model = deepcopy(member.collection_model)
            model.load_state_dict(member.weights_for(window))
            model.eval()
            self.models.append(model)

    def controls(self, state, traverser):
        import torch
        key = traverser, tuple(state.history())
        if key not in self._controls:
            member_index = 1-self.fold
            member = self.solver.q_value_trainer.members[member_index]
            x = torch.as_tensor(member.get_history_tensor(state), dtype=torch.float32, device=member.device)
            mask = np.asarray(state.legal_actions_mask(), dtype=float)
            with torch.no_grad():
                q = self.models[member_index](x).cpu().numpy() * (1 if traverser == 0 else -1) * mask
            policy = self.policy(state)
            proposal = (self.solver._traverser_sampling_policy(policy=policy, legal_mask=mask)
                        if state.current_player() == traverser else policy.copy())
            self._controls[key] = StateControls(q, np.zeros_like(q), np.zeros_like(q),
                np.ones_like(q), np.ones_like(q), proposal, policy)
        return self._controls[key]


def exact_diagnostics(solver):
    """Compare latest/averaged controls on the same policy and sampling reach.

    The existing exact recursion centres the whole action estimator before
    computing variances, so cross-action covariance is accounted for. Reported
    variances mix hidden histories within each information set using sampling
    reach; they are not claimed to be globally counterfactual-weighted regret
    variance. The within-checkpoint control contrast holds that measure fixed.
    """
    table = policy_table_for_mode(solver, "current")
    detail, summaries = [], []
    for fold in range(2):
        for window in (1,4):
            oracle = MechanismOracle(solver, table, fold, window)
            rows = oracle.rows()
            weights = np.array([r["sampling_reach_mass"] for r in rows])
            weights /= weights.sum()
            summary = {"iteration": solver.num_iteration, "fold": fold, "control_window": window,
                "information_action_rows": len(rows),
                "advantage_variance": float(weights @ np.array([r["advantage_variance"] for r in rows])),
                "max_absolute_advantage_bias": max(abs(r["advantage_bias"]) for r in rows),
                "max_absolute_value_bias": max(abs(r["action_value_bias"]) for r in rows),
                "infoset_critic_rmse": float(np.sqrt(weights @ np.array([
                    (r["q_hat"]-r["exact_action_value"])**2 for r in rows])))}
            if not all(np.isfinite(v) for v in summary.values()):
                raise ArithmeticError("Nonfinite exact diagnostic")
            if max(summary["max_absolute_advantage_bias"], summary["max_absolute_value_bias"]) > 1e-9:
                raise ArithmeticError("Conditional estimator mean audit failed")
            summaries.append(summary)
            detail.extend(dict(r, iteration=solver.num_iteration, control_window=window) for r in rows)
    return summaries, detail
