"""Real-Leduc regression checks for the native-sampler critic-budget study."""
from collections import deque
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from unbiased_escher.factorial import TemporallyAveragedCrossFittedQMember
from unbiased_escher.solver import UnbiasedControlVariateEscher
from experiments.leduc_poker.critic_update_budget.config import configuration, SEEDS, ARMS, tasks, schedule, PILOT_SEED
from experiments.leduc_poker.critic_update_budget.solver import make_solver, capture_rng, restore_rng, cpu_weights, BudgetSolver
from experiments.leduc_poker.critic_update_budget.state import pack, unpack
from experiments.leduc_poker.critic_update_budget.analyse import aggregate
from experiments.leduc_poker.critic_update_budget.diagnostics import exact_diagnostics

ROOT=Path(__file__).resolve().parents[1]


def equal_tree(test,a,b):
    if isinstance(a,torch.Tensor):test.assertTrue(torch.equal(a,b))
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:equal_tree(test,a[k],b[k])
    elif isinstance(a,(list,tuple,deque)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):equal_tree(test,x,y)
    elif isinstance(a,float) and np.isnan(a):test.assertTrue(np.isnan(b))
    else:test.assertEqual(a,b)


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def solver(self,arm="A",seed=751):
        return make_solver(seed,configuration(arm,True))

    def test_frozen_contract_and_only_budget_differences(self):
        from experiments.leduc_poker.cached_grouped_ucv_36h.config import CANDIDATE_CONFIG
        from experiments.leduc_poker.critic_averaging_mechanism.config import SEEDS as averaging_seeds
        self.assertEqual(ARMS,{"A":10000,"B":2500,"C":500,"D":0})
        self.assertEqual(len(tasks()),12);self.assertEqual(len(set(SEEDS)),3)
        self.assertFalse(set(SEEDS)&set(averaging_seeds))
        self.assertNotIn(PILOT_SEED,SEEDS)
        a=configuration("A")
        for arm in "BC":
            b=configuration(arm)
            self.assertEqual({k for k in a if a[k]!=b[k]},{"baseline_network_train_steps"})
        changed={k for k in CANDIDATE_CONFIG if CANDIDATE_CONFIG[k]!=a[k]}
        self.assertEqual(changed,{"max_num_iterations","evaluate_initial_policy","early_evaluation_node_thresholds"})
        self.assertTrue(a["use_residual_calibration"]);self.assertTrue(a["cache_frozen_critic_targets"])
        self.assertEqual(a["sampling_uniform_floor_mass"],.2)
        self.assertEqual(a["critic_target_average_window"],4)
        self.assertEqual(a["average_policy_train_steps"],20000)
        self.assertEqual(a["evaluation_frequency"],1)
        self.assertEqual(schedule()["end"],600)
        self.assertEqual(schedule()["late"],[450,500,550,600])

    def test_common_initialisation(self):
        a=self.solver("A")
        for arm in "BCD":
            b=self.solver(arm)
            for x,y in zip(a.regret_trainers,b.regret_trainers):equal_tree(self,cpu_weights(x.model),cpu_weights(y.model))
            equal_tree(self,cpu_weights(a.calibration_trainer.model),cpu_weights(b.calibration_trainer.model))
            equal_tree(self,cpu_weights(a.ave_policy_trainer.model),cpu_weights(b.ave_policy_trainer.model))
            equal_tree(self,a.streams,b.streams)
            if arm!="D":
                for x,y in zip(a.q_value_trainer.members,b.q_value_trainer.members):equal_tree(self,cpu_weights(x.model),cpu_weights(y.model))

    def test_native_sampler_and_dfs_are_inherited(self):
        self.assertIs(BudgetSolver._traverser_sampling_policy,UnbiasedControlVariateEscher._traverser_sampling_policy)
        self.assertIs(BudgetSolver.dfs,UnbiasedControlVariateEscher.dfs)
        s=self.solver()
        kwargs=dict(q_values=np.zeros(3),beta=np.ones(3),residual_means=np.zeros(3),
                    predicted_variances=np.array([100,1,9]),policy=np.array([0,.1,.9]),legal_mask=np.array([0,1,1]))
        np.testing.assert_allclose(s._traverser_sampling_policy(**kwargs),[0,.3,.7])
        kwargs['policy']=np.array([0,.9,.1])
        np.testing.assert_allclose(s._traverser_sampling_policy(**kwargs),[0,.3,.7])

    def test_update_counts_and_native_policy_fit_cadence(self):
        for arm in "ABCD":
            s=self.solver(arm)
            for _ in range(4):s.iteration()
            self.assertEqual(s.policy_fit_count,4)
            self.assertEqual(s.last_policy_fit_iteration,4)
            self.assertEqual(s.calibration_trainer.target_version,4)
            for member in s.q_value_trainer.members:
                self.assertEqual(member.total_updates,4*configuration(arm,True)["baseline_network_train_steps"])
                if arm!="D":
                    steps=[int(v['step']) for v in member.optimizer.state.values()]
                    self.assertTrue(steps);self.assertEqual(set(steps),{member.total_updates})
                    self.assertEqual(len(member.target_history),4)

    def test_zero_control_never_fits_or_leaves_random_predictions(self):
        s=self.solver("D")
        for _ in range(4):s.iteration()
        root=s.skip_chance_state(s.game.new_initial_state())
        for member in s.q_value_trainer.members:
            self.assertEqual(member.total_updates,0);self.assertEqual(member.target_version,0)
            self.assertEqual(len(member.target_history),0);self.assertFalse(member.optimizer.state)
            self.assertFalse(hasattr(member,"last_target_cache_stats"))
            for model in (member.model,member.target_model):
                self.assertTrue(all(torch.count_nonzero(p)==0 for p in model.parameters()))
            np.testing.assert_array_equal(member.get_baseline(root,0),np.zeros(s.action_size))
            np.testing.assert_array_equal(member.get_baseline(root,1),np.zeros(s.action_size))
        self.assertGreater(len(s.calibration_trainer.buffer),0)
        self.assertGreater(s.calibration_trainer.target_version,0)

    def test_positive_fit_is_exact_native_cached_averaging(self):
        s=self.solver("B")
        for _ in range(3):s.iteration()
        m=s.q_value_trainer.members[0]
        native=TemporallyAveragedCrossFittedQMember(m.input_size,s.infostate_size,s.action_size,s.network_layers,
            s.learning_rate,m.buffer.buffer_size,m.batch_size,m.train_steps,s.logger,s.regret_trainers,s.device,
            gradient_clip_norm=m.gradient_clip_norm,target_average_window=4)
        native.model.load_state_dict(m.model.state_dict());native.target_model.load_state_dict(m.target_model.state_dict())
        native.optimizer.load_state_dict(deepcopy(m.optimizer.state_dict()))
        native.buffer=deepcopy(m.buffer);native.target_history.extend(deepcopy(list(m.target_history)))
        native.target_version=m.target_version;native.cache_frozen_targets=True
        rng=capture_rng();m.train_model(4);restore_rng(rng);native.train_model(4)
        equal_tree(self,cpu_weights(m.model),cpu_weights(native.model))
        equal_tree(self,cpu_weights(m.target_model),cpu_weights(native.target_model))
        equal_tree(self,m.optimizer.state_dict(),native.optimizer.state_dict())
        equal_tree(self,m.target_history,native.target_history)

    def test_resume_preserves_calibration_replay_rng_and_policy(self):
        for arm in "ABCD":
            a=self.solver(arm);config=configuration(arm,True)
            for _ in range(3):a.iteration()
            payload=pack(a,arm,751,config,"identity",{})
            b=self.solver(arm);unpack(b,payload,arm,751,config,"identity")
            for _ in range(2):a.iteration();b.iteration()
            x=pack(a,arm,751,config,"identity",{});y=pack(b,arm,751,config,"identity",{})
            for key in ("regret_trainers","average_policy_trainer","q_ensemble","calibration","exact_average_strategy"):
                equal_tree(self,x[key],y[key])
            equal_tree(self,a.streams,b.streams)
            self.assertEqual(a.policy_fit_count,b.policy_fit_count)
            self.assertEqual(a.nodes_touched,b.nodes_touched)
            self.assertEqual([m.total_updates for m in a.q_value_trainer.members],[m.total_updates for m in b.q_value_trainer.members])
            with self.assertRaises(ValueError):unpack(b,payload,arm,751,config,"wrong")

    def test_evaluation_does_not_change_learning(self):
        a=self.solver();b=self.solver()
        for _ in range(2):a.iteration();b.iteration()
        a.evaluate_budget()
        a.iteration();b.iteration()
        for x,y in zip(a.regret_trainers,b.regret_trainers):equal_tree(self,cpu_weights(x.model),cpu_weights(y.model))
        equal_tree(self,cpu_weights(a.calibration_trainer.model),cpu_weights(b.calibration_trainer.model))
        equal_tree(self,a.streams,b.streams)
        self.assertEqual(a.policy_fit_count,3)

    def test_zero_exact_audit_and_counterfactual_identity(self):
        s=self.solver("D")
        for _ in range(2):s.iteration()
        before=deepcopy(s.streams);fold=s.q_value_trainer.active_fold
        summary,_=exact_diagnostics(s)
        self.assertEqual(len(summary),4)
        for i in (0,2):
            self.assertAlmostEqual(summary[i]['advantage_variance'],summary[i+1]['advantage_variance'],places=13)
        self.assertLess(max(r['max_absolute_advantage_bias'] for r in summary),1e-9)
        equal_tree(self,before,s.streams);self.assertEqual(fold,s.q_value_trainer.active_fold)

    def test_disk_worker_resume_and_checksum(self):
        from experiments.leduc_poker.critic_update_budget.worker import run_worker
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b'
            run_worker(a,'B',PILOT_SEED,True,stop_after=4)
            run_worker(b,'B',PILOT_SEED,True,stop_after=2)
            run_worker(b,'B',PILOT_SEED,True,stop_after=4)
            def metrics(path):
                with (path/'metrics.csv').open() as handle:
                    return [{k:v for k,v in row.items() if not k.endswith('seconds')} for row in csv.DictReader(handle)]
            self.assertEqual(metrics(a),metrics(b))
            marker=json.loads((b/'latest.json').read_text());marker['sha256']='0'*64
            (b/'latest.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'checksum'):run_worker(b,'B',PILOT_SEED,True)

    def test_cloud_scripts_and_missing_run_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):aggregate(tmp)
        spec=importlib.util.spec_from_file_location('builder51',ROOT/'gcp/critic_update_budget_batch.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for kind in ('controller','smoke','pilot','train','aggregate'):
            args=SimpleNamespace(kind=kind,project='example',region='europe-west1',bucket='gs://example',
                run_id='exp51-test',service_account='test@example',repo_ref='a'*40,parallelism=8,
                source_uri='gs://example/source.pt',source_sha256='b'*64,attempt='initial')
            job=mod.build_job(args)
            self.assertEqual(job['taskGroups'][0]['taskCount'],len(tasks()) if kind=='train' else 1)
            with tempfile.NamedTemporaryFile(mode='w',suffix='.sh') as handle:
                handle.write(mod.script(args));handle.flush();subprocess.run(['bash','-n',handle.name],check=True)

    def test_mature_pilot_rejects_changed_native_learning_configuration(self):
        from experiments.leduc_poker.grouped_wide_policy_confirmation.config import CANDIDATE_CONFIG
        from experiments.leduc_poker.critic_update_budget.run import validate_pilot_source
        payload={"type":"experiment_35_full_training_state", "solver":{"num_iteration":300},
                 "config":deepcopy(CANDIDATE_CONFIG)}
        self.assertEqual(validate_pilot_source(payload),300)
        for key,value in (("use_residual_calibration",False), ("sampling_uniform_floor_mass",.3),
                          ("average_policy_train_steps",5000), ("average_policy_reset_each_fit",False)):
            changed=deepcopy(payload);changed["config"][key]=value
            with self.assertRaisesRegex(ValueError,key):validate_pilot_source(changed)
        payload["solver"]["num_iteration"]=299
        with self.assertRaisesRegex(ValueError,"300"):validate_pilot_source(payload)


if __name__=='__main__':unittest.main()
