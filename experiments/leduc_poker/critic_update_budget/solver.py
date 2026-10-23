"""Native adaptive UCV learner with explicit update budgets and zero Q control."""
import time
import numpy as np
import torch

from unbiased_escher.factorial import TemporallyAveragedCrossFittedQMember
from unbiased_escher.solver import CrossFittedQEnsemble
from unbiased_escher.stability import StableUnbiasedControlVariateEscher
from vr_deep_cfr.solver import DeepCumuAdv
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import DiagnosticGroupedWideUCVSolver, _make_solver
from experiments.leduc_poker.critic_averaging_mechanism.solver import (
    capture_rng, restore_rng, seeded_rng, stream, cpu_weights)
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import build_policy_table, exact_exploitability


class BudgetMember(TemporallyAveragedCrossFittedQMember):
    def __init__(self, *args, zero_control=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_control = zero_control
        self.cache_frozen_targets = True
        self.fit_diagnostics = []
        self.total_updates = 0
        if zero_control:
            self.install_zero_control()

    def install_zero_control(self):
        # Initialize normally first so other components retain common initialization.
        # Also zero stored networks: snapshots must not contain a random critic.
        with torch.no_grad():
            for model in (self.model, self.target_model):
                for parameter in model.parameters():
                    parameter.zero_()
        self.optimizer.state.clear()
        self.target_history.clear()
        self.target_version = 0
        self.last_target_update_l2 = 0.

    def get_baseline(self, state, player):
        if self.zero_control:
            return np.zeros(self.output_size, dtype=np.float64)
        return super().get_baseline(state, player)

    def train_model(self, iteration):
        started = time.perf_counter()
        if self.zero_control:
            if self.train_steps != 0:
                raise ValueError("Zero control cannot fit a critic")
            loss = None
        else:
            loss = super().train_model(iteration)
        updates = self.train_steps if loss is not None else 0
        self.total_updates += updates
        cache = getattr(self, "last_target_cache_stats", {}) if updates else {}
        self.fit_diagnostics.append({"iteration":iteration, "configured_updates":self.train_steps,
            "actual_updates":updates, "cumulative_updates":self.total_updates,
            "replay_rows":len(self.buffer), "target_version":self.target_version,
            "history_size":len(self.target_history), "fit_seconds":time.perf_counter()-started,
            "cache_build_seconds":cache.get("cache_build_seconds",0.),
            "cache_bytes":cache.get("cache_bytes",0),
            "td_parameter_update_l2":self.last_target_update_l2 if updates else 0.,
            "final_minibatch_loss":float(loss) if loss is not None else None})
        return loss


class BudgetEnsemble(CrossFittedQEnsemble):
    def train_model(self, iteration):
        losses=[]
        for index,member in enumerate(self.members):
            with stream(self.streams,f"critic/{index}"):
                losses.append(member.train_model(iteration))
        valid=[v for v in losses if v is not None]
        return float(np.mean(valid)) if valid else None


class CalibrationStream:
    """Forward native calibration state, isolating only its training RNG/timer."""
    def __init__(self, trainer, solver):
        object.__setattr__(self,"trainer",trainer)
        object.__setattr__(self,"solver",solver)

    def __getattr__(self, name):
        return getattr(self.trainer,name)

    def __setattr__(self,name,value):
        setattr(self.trainer,name,value)

    def train_model(self):
        started=time.perf_counter()
        with stream(self.solver.streams,"calibration"):
            result=self.trainer.train_model()
        self.solver.add_seconds("calibration_fit",time.perf_counter()-started)
        return result


class BudgetSolver(DiagnosticGroupedWideUCVSolver):
    def __init__(self, *args, zero_critic_control=False, cache_frozen_critic_targets=True, **kwargs):
        if not cache_frozen_critic_targets:
            raise ValueError("Frozen-target caching is required in this experiment")
        self.zero_critic_control=bool(zero_critic_control)
        budget=int(kwargs["baseline_network_train_steps"])
        if (budget==0) != self.zero_critic_control or budget<0:
            raise ValueError("Zero updates must explicitly select the zero-valued control")
        super().__init__(*args,**kwargs)
        if self.calibration_trainer is None:
            raise ValueError("This experiment requires native residual calibration")
        self.streams={name:seeded_rng(kwargs["seed"],name) for name in
            ("collect/0","collect/1","regret/0","regret/1","critic/0","critic/1","calibration","policy")}
        self.component_seconds={}
        self.q_value_trainer.streams=self.streams
        self.calibration_trainer=CalibrationStream(self.calibration_trainer,self)
        self.policy_fit_count=0
        self.last_policy_fit_iteration=0

    def add_seconds(self, key, seconds):
        self.component_seconds[key]=self.component_seconds.get(key,0.)+seconds

    def init_q_value_trainer(self):
        root=self.game.new_initial_state()
        size=len(np.append(root.information_state_tensor(0),root.information_state_tensor(1)))
        ensemble=BudgetEnsemble.__new__(BudgetEnsemble)
        ensemble.ensemble_size=self.q_ensemble_size
        ensemble.active_fold=ensemble._active_add_count=0
        ensemble.members=[BudgetMember(size,self.infostate_size,self.action_size,self.network_layers,
            self.learning_rate,self.baseline_buffer_size//self.q_ensemble_size,self.baseline_batch_size,
            self.baseline_network_train_steps,self.logger,self.regret_trainers,self.device,
            gradient_clip_norm=self.q_gradient_clip_norm,
            target_average_window=self.critic_target_average_window,zero_control=self.zero_critic_control)
            for _ in range(self.q_ensemble_size)]
        self.q_value_trainer=ensemble

    # Deliberately inherit _traverser_sampling_policy and dfs unchanged.
    def _record_detailed_estimate_diagnostics(self, **kwargs):
        pass  # Passive historical per-visit dumps are replaced by scheduled exact audits.

    def collect_training_data(self, player):
        started=time.perf_counter()
        with stream(self.streams,f"collect/{player}"):
            super().collect_training_data(player)
        self.add_seconds("collection",time.perf_counter()-started)

    def train_regret(self, player):
        started=time.perf_counter()
        with stream(self.streams,f"regret/{player}"):
            super().train_regret(player)
        self.add_seconds("regret_fit",time.perf_counter()-started)

    def train_average_policy(self):
        started=time.perf_counter()
        with stream(self.streams,"policy"):
            # Native candidate reset + 20k fit; omit only the obsolete paired
            # legacy comparison head and expensive historical diagnostic bundle.
            DeepCumuAdv.train_average_policy(self)
        self.policy_fit_count+=1
        self.last_policy_fit_iteration=self.num_iteration
        self.add_seconds("policy_fit",time.perf_counter()-started)

    def _run_checkpoint(self, **kwargs):
        # Native iteration calls this every iteration. Evaluation is scheduled
        # independently by the worker, but candidate fitting is never skipped.
        self.train_average_policy()

    def iteration(self):
        table={}
        def policy(state):
            key=(state.current_player(),state.information_state_string())
            if key not in table:
                table[key]=self.regret_trainers[key[0]].get_policy(state,self.num_iteration+1)
            return table[key]
        started=time.perf_counter()
        self.exact_average_strategy.observe_iteration(self.num_iteration+1,policy)
        self.add_seconds("exact_average_accumulation",time.perf_counter()-started)
        started=time.perf_counter()
        StableUnbiasedControlVariateEscher.iteration(self)
        self.add_seconds("learning_inclusive",time.perf_counter()-started)
        self.add_seconds("critic_fit",sum(m.fit_diagnostics[-1]["fit_seconds"] for m in self.q_value_trainer.members))
        self.logger.dump(step=self.num_iteration)

    def evaluate_budget(self):
        if self.last_policy_fit_iteration!=self.num_iteration:
            raise RuntimeError("Missing native per-iteration average-policy fit")
        started=time.perf_counter()
        table=build_policy_table(self.game,lambda s:self.ave_policy_trainer.action_probabilities(s,probs_as_dict=False))
        return {"iteration":self.num_iteration,"nodes":self.nodes_touched,"trajectories":self.episode,
            "exact_average_exploitability":self.exact_average_strategy.exploitability(),
            "deployed_exploitability":exact_exploitability(self.game,table),
            "policy_fit_count":self.policy_fit_count,"evaluation_seconds":time.perf_counter()-started}


def make_solver(seed, config):
    return _make_solver(seed,config,solver_class=BudgetSolver)
