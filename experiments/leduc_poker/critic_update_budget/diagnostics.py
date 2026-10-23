"""Exact frozen-policy estimator audit under the actual adaptive proposal."""
from dataclasses import replace
import numpy as np
from experiments.leduc_poker.ucv_exact_tabular_validation.diagnostics import ExactUCVOracle, policy_table_for_mode


class BudgetOracle(ExactUCVOracle):
    def __init__(self, solver, table, fold, control):
        super().__init__(solver,policy_table=table,variant_id="fixed_beta_one",fold=fold)
        self.control=control

    def controls(self, state, traverser):
        native=super().controls(state,traverser)
        if self.control=="zero_same_proposal":
            # Keep the learned calibration/proposal fixed while removing Q
            # from the estimator. This is not a counterfactual retrained sampler.
            return replace(native,q_hat=np.zeros_like(native.q_hat))
        return native


def exact_diagnostics(solver):
    table=policy_table_for_mode(solver,"current")
    summaries=[]; detail=[]
    for fold in range(solver.q_ensemble_size):
        for control in ("deployed","zero_same_proposal"):
            oracle=BudgetOracle(solver,table,fold,control)
            rows=oracle.rows()
            weights=np.array([r["sampling_reach_mass"] for r in rows])
            weights/=weights.sum()
            variances=np.array([r["advantage_variance"] for r in rows])
            values={"information_action_rows":len(rows),
                "advantage_variance":float(weights@variances),
                "advantage_variance_p95":float(np.quantile(variances,.95)),
                "advantage_variance_max":float(variances.max()),
                "max_absolute_advantage_bias":max(abs(r["advantage_bias"]) for r in rows),
                "max_absolute_value_bias":max(abs(r["action_value_bias"]) for r in rows),
                "infoset_critic_rmse":float(np.sqrt(weights@np.array([(r["q_hat"]-r["exact_action_value"])**2 for r in rows])))}
            if not all(np.isfinite(v) for v in values.values()):
                raise ArithmeticError("Nonfinite exact diagnostic")
            if max(values["max_absolute_advantage_bias"],values["max_absolute_value_bias"])>1e-9:
                raise ArithmeticError("Conditional estimator mean audit failed")
            summaries.append(dict(values,iteration=solver.num_iteration,fold=fold,control=control))
            detail.extend(dict(r,iteration=solver.num_iteration,control=control) for r in rows)
    return summaries,detail
