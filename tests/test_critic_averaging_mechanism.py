"""Real Leduc tests; runnable with unittest without additional dependencies."""
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

from adaptive_escher.frozen_target_cache import build_target_cache
from unbiased_escher.factorial import TemporallyAveragedCrossFittedQMember
from experiments.leduc_poker.critic_averaging_mechanism.config import configuration, SEEDS, tasks, schedule
from experiments.leduc_poker.critic_averaging_mechanism.solver import make_solver, capture_rng, restore_rng, cpu_weights
from experiments.leduc_poker.critic_averaging_mechanism.state import pack,unpack
from experiments.leduc_poker.critic_averaging_mechanism.analyse import effect,aggregate

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


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def solver(self,arm="D",seed=721):
        return make_solver(seed,configuration(arm,True))

    def test_contract(self):
        self.assertEqual(len(tasks()),12);self.assertEqual(len(set(SEEDS)),3)
        for arm in "ABCD":
            self.assertEqual([seed for a,seed in tasks() if a==arm],list(SEEDS))
        self.assertEqual(schedule()["late"],[450,500,550,600])
        self.assertEqual(configuration("D")["baseline_network_train_steps"],10000)
        self.assertEqual(configuration("D")["num_traversals"],10000)
        self.assertEqual(schedule()["fit"],[300,600])

    def test_mature_pilot_configuration_validation(self):
        from experiments.leduc_poker.critic_averaging_mechanism.run import validate_pilot_source
        from experiments.leduc_poker.grouped_wide_policy_confirmation.config import CANDIDATE_CONFIG
        payload={"type":"experiment_35_full_training_state", "solver":{"num_iteration":500},
                 "config":deepcopy(CANDIDATE_CONFIG)}
        self.assertEqual(validate_pilot_source(payload),500)
        payload["config"]["num_hiddens"]=32
        with self.assertRaisesRegex(ValueError,"num_hiddens"):validate_pilot_source(payload)
        payload["solver"]["num_iteration"]=5
        with self.assertRaisesRegex(ValueError,"300"):validate_pilot_source(payload)

    def test_common_initialisation(self):
        a=self.solver("A");d=self.solver("D")
        for x,y in zip(a.regret_trainers,d.regret_trainers):equal_tree(self,cpu_weights(x.model),cpu_weights(y.model))
        equal_tree(self,a.streams,d.streams)

    def test_role_independence_and_cache(self):
        s=self.solver("B")
        for _ in range(3):s.iteration()
        m=s.q_value_trainer.members[0]
        before=build_target_cache(m,s.num_iteration).clone()
        with torch.no_grad():
            for p in m.collection_model.parameters():p.add_(2)
        self.assertTrue(torch.equal(before,build_target_cache(m,s.num_iteration)))
        collection=cpu_weights(m.collection_model)
        with torch.no_grad():
            for p in m.target_model.parameters():p.add_(.5)
        self.assertFalse(torch.equal(before,build_target_cache(m,s.num_iteration)))
        equal_tree(self,collection,cpu_weights(m.collection_model))

    def test_equal_roles_match_existing_native_averaging(self):
        for window,arm in [(1,"A"),(4,"D")]:
            s=self.solver(arm)
            for _ in range(3):s.iteration()
            m=s.q_value_trainer.members[0]
            native=TemporallyAveragedCrossFittedQMember(
                m.input_size,s.infostate_size,s.action_size,s.network_layers,s.learning_rate,
                m.buffer.buffer_size,m.batch_size,m.train_steps,s.logger,s.regret_trainers,s.device,
                gradient_clip_norm=m.gradient_clip_norm,target_average_window=window)
            native.model.load_state_dict(m.model.state_dict());native.target_model.load_state_dict(m.target_model.state_dict())
            native.optimizer.load_state_dict(deepcopy(m.optimizer.state_dict()))
            native.buffer=deepcopy(m.buffer);native.target_history.extend(list(m.target_history)[-window:])
            native.cache_frozen_targets=True
            rng=capture_rng();m.train_model(s.num_iteration);restore_rng(rng);native.train_model(s.num_iteration)
            equal_tree(self,cpu_weights(m.model),cpu_weights(native.model))
            equal_tree(self,cpu_weights(m.target_model),cpu_weights(native.target_model))
            equal_tree(self,cpu_weights(m.collection_model),cpu_weights(native.target_model))

    def test_cached_and_uncached_updates_match(self):
        s=self.solver("C")
        for _ in range(3):s.iteration()
        a=s.q_value_trainer.members[0];b=deepcopy(a);b.cache_frozen_targets=False
        rng=capture_rng();a.train_model(4);restore_rng(rng);b.train_model(4)
        equal_tree(self,cpu_weights(a.model),cpu_weights(b.model))
        equal_tree(self,a.optimizer.state_dict(),b.optimizer.state_dict())

    def test_policy_fit_does_not_change_learning(self):
        a=self.solver();b=self.solver()
        for _ in range(3):a.iteration();b.iteration()
        a.fit_policy()
        for _ in range(2):a.iteration();b.iteration()
        for x,y in zip(a.regret_trainers,b.regret_trainers):equal_tree(self,cpu_weights(x.model),cpu_weights(y.model))
        for x,y in zip(a.q_value_trainer.members,b.q_value_trainer.members):
            equal_tree(self,cpu_weights(x.model),cpu_weights(y.model))
            equal_tree(self,x.buffer.history_buf,y.buffer.history_buf)
        self.assertEqual(a.nodes_touched,b.nodes_touched)

    def test_exact_resume_all_learning_state(self):
        for arm in "ABCD":
            c=configuration(arm,True);a=self.solver(arm)
            for _ in range(3):a.iteration()
            payload=pack(a,arm,721,c,"identity",{"metrics":[]})
            b=self.solver(arm);unpack(b,payload,arm,721,c,"identity")
            for _ in range(2):a.iteration();b.iteration()
            x=pack(a,arm,721,c,"identity",{});y=pack(b,arm,721,c,"identity",{})
            for key in ("regret_trainers","average_policy_trainer","q_ensemble","exact_average_strategy"):
                equal_tree(self,x[key],y[key])
            equal_tree(self,a.streams,b.streams)
            self.assertEqual(a.nodes_touched,b.nodes_touched)
            with self.assertRaises(ValueError):unpack(b,payload,arm,721,c,"wrong-code")

    def test_proposal_support_and_critic_independence(self):
        s=self.solver()
        policy=np.array([0.,.1,.9]);mask=np.array([0.,1.,1.])
        p=s._traverser_sampling_policy(policy=policy,legal_mask=mask,q_values=np.ones(3))
        np.testing.assert_allclose(p,[0.,.18,.82])

    def test_analysis_missing_workers_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):aggregate(d)
        self.assertEqual(effect([1.])["ci_low"],None)
        self.assertAlmostEqual(effect([-.01]*8)["ci_high"],-.01)

    def test_worker_disk_resume_and_corruption(self):
        from experiments.leduc_poker.critic_averaging_mechanism.worker import run_worker
        from experiments.leduc_poker.critic_averaging_mechanism.config import PILOT_SEED
        with tempfile.TemporaryDirectory() as tmp:
            uninterrupted=Path(tmp)/"continuous"; resumed=Path(tmp)/"resumed"
            run_worker(uninterrupted,"B",PILOT_SEED,True,stop_after=4)
            run_worker(resumed,"B",PILOT_SEED,True,stop_after=2)
            run_worker(resumed,"B",PILOT_SEED,True,stop_after=4)
            def outcomes(path):
                with (path/"metrics.csv").open() as f:
                    return [{k:v for k,v in r.items() if not k.endswith("seconds")} for r in csv.DictReader(f)]
            self.assertEqual(outcomes(uninterrupted),outcomes(resumed))
            marker=json.loads((resumed/"latest.json").read_text())
            marker['sha256']='0'*64
            (resumed/"latest.json").write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,"checksum"):
                run_worker(resumed,"B",PILOT_SEED,True,stop_after=4)

    def test_cloud_scripts_and_array(self):
        spec=importlib.util.spec_from_file_location("builder50",ROOT/"gcp/critic_averaging_mechanism_batch.py")
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for kind in ("controller","smoke","pilot","train","aggregate"):
            args=SimpleNamespace(kind=kind,project="example",region="europe-west1",bucket="gs://example",
                run_id="exp50-test",service_account="test@example",repo_ref="a"*40,parallelism=8,
                source_uri="gs://example/source.pt",source_sha256="b"*64,attempt="initial")
            job=mod.build_job(args);group=job["taskGroups"][0]
            self.assertEqual(group["taskCount"],len(tasks()) if kind=="train" else 1)
            if kind=="controller":
                self.assertEqual(group["taskSpec"]["maxRunDuration"],"561600s")
                args.parallelism=3
                self.assertEqual(mod.build_job(args)["taskGroups"][0]["taskSpec"]["maxRunDuration"],"1080000s")
            with tempfile.NamedTemporaryFile(mode="w",suffix=".sh") as f:
                f.write(mod.script(args));f.flush()
                subprocess.run(["bash","-n",f.name],check=True)


if __name__=="__main__":unittest.main()
