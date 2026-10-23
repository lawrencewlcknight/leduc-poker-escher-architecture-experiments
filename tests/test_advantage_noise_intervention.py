"""Real Leduc checks for the oracle intervention and native fitting control."""
from copy import deepcopy, copy
import json
import importlib.util
import os
from pathlib import Path
import random
import tempfile
import subprocess
import types
import unittest

import numpy as np
import torch

from unbiased_escher.estimator import control_variate_advantage
from vr_deep_cfr.solver import ReservoirBuffer
from experiments.leduc_poker.critic_averaging_mechanism.solver import capture_rng, restore_rng
from experiments.leduc_poker.grouped_wide_policy_confirmation.config import CANDIDATE_CONFIG, PRODUCTION_SEEDS
from experiments.leduc_poker.advantage_noise_intervention.config import contract, tasks, SEEDS, SMOKE_SEED, derived_seed
from experiments.leduc_poker.advantage_noise_intervention.source import load_source, validate_source, reconstruct
from experiments.leduc_poker.advantage_noise_intervention.core import FrozenProblem, transform, fit, evaluate_fit, inference_digest
from experiments.leduc_poker.advantage_noise_intervention.worker import run_worker, worker_name
from experiments.leduc_poker.advantage_noise_intervention.analyse import aggregate


def equal(test, a, b):
    if isinstance(a, torch.Tensor): test.assertTrue(torch.equal(a,b))
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a, dict):
        test.assertEqual(set(a),set(b))
        for key in a: equal(test,a[key],b[key])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): equal(test,x,y)
    else: test.assertEqual(a,b)


class NoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.solver, _ = load_source(None, SMOKE_SEED, 'smoke', True)
        cls.problem = FrozenProblem(cls.solver)

    def test_contract_and_source_selection(self):
        self.assertEqual(SEEDS, tuple(sorted(PRODUCTION_SEEDS)[:3]))
        c=contract()
        self.assertEqual(len(tasks())*c['repetitions']*3*2,288)
        self.assertEqual((c['fit_steps'],c['batch_size'],c['traversals_per_player']),(750,2048,10000))
        self.assertNotEqual(derived_seed(1,'x',0,0,'collection'),derived_seed(1,'x',0,0,'minibatches'))
        with self.assertRaises(ValueError): contract(True,True)

    def test_source_validation_and_memory_light_restore(self):
        seed, checkpoint = tasks()[0]
        payload={'type':'experiment_35_full_training_state','schema_version':1,
            'experiment_name':'grouped_wide_policy_confirmation','variant_id':'grouped_wide_ucv',
            'seed':seed,'checkpoint_id':checkpoint,'solver':{'num_iteration':200},
            'config':deepcopy(CANDIDATE_CONFIG),'regret_trainers':[{},{}],
            'q_ensemble':{'members':[{},{}]},'calibration':{}}
        validate_source(payload,seed,checkpoint)
        for key,value in [('use_residual_calibration',False),('sampling_uniform_floor_mass',.3),('advantage_network_train_steps',100)]:
            changed=deepcopy(payload);changed['config'][key]=value
            with self.assertRaisesRegex(ValueError,key):validate_source(changed,seed,checkpoint)
        # Reconstruct the actual tiny solver components without any replay payload.
        from experiments.leduc_poker.ucv_residual_target_factorial.training_state import build_training_state
        from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _smoke_overrides
        config=deepcopy(CANDIDATE_CONFIG);_smoke_overrides(config)
        actual=build_training_state(self.solver,variant_id='grouped_wide_ucv',seed=SMOKE_SEED,
            checkpoint_id='smoke',active_seconds=0,repository_commit='test',config=config,captured_snapshots=[])
        restored=reconstruct(actual)
        self.assertEqual(inference_digest(restored),inference_digest(self.solver))
        equal(self,restored.regret_trainers[0].optimizer.state_dict(),self.solver.regret_trainers[0].optimizer.state_dict())
        self.assertEqual(restored.calibration_trainer.target_version,self.solver.calibration_trainer.target_version)
        self.assertEqual(restored.num_iteration,self.solver.num_iteration+1)
        self.assertEqual(len(restored.q_value_trainer.members[0].buffer),0)

    def test_collector_matches_native_trajectories_and_labels(self):
        s,_=load_source(None,SMOKE_SEED,'smoke',True)
        s.num_traversals=8
        p=FrozenProblem(s)
        # Both the collector and exact oracle use one float64-normalized frozen
        # profile. Feed that identical profile to native DFS; do not compare two
        # different float32/float64 normalization paths at 1e-12 tolerance.
        for trainer in s.regret_trainers:
            trainer.get_policy=lambda state,T: p.table[(state.current_player(),state.information_state_string())]
        for player in (0,1):
            seed=153+player
            panel=p.collect(player,8,seed)
            captured=[]
            def record(_self, **kw):
                captured.append((tuple(kw['state'].history()),kw['estimate'].advantages.copy()))
            s._record_detailed_estimate_diagnostics=types.MethodType(record,s)
            rng=capture_rng(); episode=s.episode
            np.random.seed(seed)
            s.collect_training_data(player)
            restore_rng(rng);s.episode=episode
            self.assertEqual([','.join(map(str,h)) for h,_ in captured],panel['history'].tolist())
            np.testing.assert_allclose(np.stack([a for _,a in captured]),panel['sampled'],atol=1e-12,rtol=1e-12)
            p.assert_frozen()

    def test_exact_full_covariance_scaling_on_enumerated_suffix(self):
        p=self.problem; oracle=p.oracles[0]
        candidates=[s for (player,_),s in p.states.items() if player==0]
        # Last betting decisions have terminal children: enumerate every possible
        # sampled action, rather than asserting the algebra on arbitrary arrays.
        state=next(s for s in candidates if len(s.legal_actions())>=2 and
                   all(s.child(a).is_terminal() for a in s.legal_actions()) and
                   np.linalg.norm(oracle.mean_advantage(s,0))>1e-5)
        c=oracle.controls(state,0); mu=oracle.mean_advantage(state,0)
        samples=[]; probabilities=[]
        for action in state.legal_actions():
            value=state.child(action).returns()[0]/p.solver.max_utility
            estimate=control_variate_advantage(c.q_hat,beta=c.beta,sampled_action=action,
                sample_probability=c.sampling_policy[action],sampled_return=value,policy=c.policy,
                legal_actions_mask=state.legal_actions_mask())
            samples.append(estimate.advantages);probabilities.append(c.sampling_policy[action])
        a=np.stack(samples);w=np.asarray(probabilities)
        panel={'sampled':a,'mean':np.tile(mu,(len(a),1)),
            'mask':np.tile(state.legal_actions_mask(),(len(a),1)),'policy':np.tile(c.policy,(len(a),1))}
        mean,cov=oracle.vector_moments(state,0)
        self.assertGreater(np.linalg.norm(cov),1e-8)
        for gamma in (1.,.5,0.):
            transformed=transform(panel,gamma)
            np.testing.assert_allclose(w@transformed,mu,atol=1e-12)
            residual=transformed-mu
            np.testing.assert_allclose((residual.T*w)@residual,gamma**2*cov,atol=1e-12)
        np.testing.assert_array_equal(transform(panel,1.),a)

    def test_reference_mean_uses_native_collection_reach(self):
        p=self.problem
        for player in (0,1):
            combined={}
            for oracle in p.oracles:
                for row in oracle.rows():
                    if row['player']!=player:continue
                    key=(row['information_state'],row['action'])
                    mass,value=combined.get(key,(0.,0.))
                    m=.5*row['sampling_reach_mass']
                    combined[key]=(mass+m,value+m*row['exact_advantage'])
            for row in p.references[player]:
                for action in np.flatnonzero(row['mask']):
                    mass,value=combined[row['key'][1],action]
                    self.assertAlmostEqual(row['mu'][action],value/mass,places=12)
            self.assertAlmostEqual(sum(r['weight'] for r in p.references[player]),1.,places=12)

    def test_native_identity_fit_optimizer_pairing_and_frozen_inputs(self):
        p=self.problem;panel=p.collect(0,16,555);rng=capture_rng()
        baseline,stat=fit(p,0,panel,1.,3,8,777)
        equal(self,rng,capture_rng())
        original=p.solver.regret_trainers[0]
        native=copy(original);native.model=deepcopy(original.model);native.target_model=deepcopy(original.target_model)
        native.optimizer=torch.optim.Adam(native.model.parameters(),lr=original.learning_rate)
        native.optimizer.load_state_dict(deepcopy(original.optimizer.state_dict()))
        native.buffer=ReservoirBuffer(len(panel['sampled']),original.input_size,original.output_size)
        for x,y,mask in zip(panel['features'],panel['sampled'],panel['mask']):
            native.add_data(x,y,mask,p.iteration)
        native.train_steps=3;native.batch_size=8
        random.seed(777);np.random.seed(777);torch.manual_seed(777)
        native.train_model(p.iteration);restore_rng(rng)
        equal(self,native.model.state_dict(),baseline.model.state_dict())
        equal(self,native.optimizer.state_dict(),baseline.optimizer.state_dict())
        start_steps=[int(s['step']) for s in original.optimizer.state.values()]
        end_steps=[int(s['step']) for s in baseline.optimizer.state.values()]
        self.assertEqual(end_steps,[n+3 for n in start_steps])
        for gamma in (.5,0.):
            other,otherstat=fit(p,0,panel,gamma,3,8,777)
            self.assertEqual(stat['minibatch_sha256'],otherstat['minibatch_sha256'])
            self.assertEqual(stat['previous_target_sha256'],otherstat['previous_target_sha256'])
        evaluation,_,_=evaluate_fit(p,0,baseline)
        self.assertTrue(all(np.isfinite(x) for x in evaluation.values()))
        equal(self,rng,capture_rng());p.assert_frozen()

    def test_resume_complete_analysis_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            run_worker(root,smoke=True,stop_after=1)
            d=root/'workers'/worker_name(SMOKE_SEED,'smoke')
            before=(d/'rep_00/metrics.json').read_bytes()
            result=run_worker(root,smoke=True)
            self.assertEqual(result['fits'],12)
            self.assertEqual((d/'rep_00/metrics.json').read_bytes(),before)
            summary=aggregate(root,True)
            self.assertEqual(summary['source_seed_count'],1)
            self.assertTrue(all(r['n']==1 for r in summary['effects']))
            with self.assertRaises(FileNotFoundError):aggregate(root,False)
            (d/'rep_00/panel_0.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrupt'):aggregate(root,True)
            with self.assertRaisesRegex(ValueError,'corrupt'):run_worker(root,smoke=True)

    def test_generated_cloud_scripts_and_historical_array_mapping(self):
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('builder52',root/'gcp/advantage_noise_intervention_batch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for kind in ('controller','smoke','pilot','worker','aggregate'):
            args=types.SimpleNamespace(kind=kind,project='example',region='europe-west1',bucket='gs://dest',
                source_bucket='gs://source',source_run_id='exp35-confirm-20260916-011231',run_id='exp52-test',
                service_account='test@example',repo_ref='a'*40,parallelism=3,attempt='initial')
            job=module.build_job(args)
            self.assertEqual(job['taskGroups'][0]['taskCount'],6 if kind=='worker' else 1)
            self.assertEqual(job['taskGroups'][0]['taskSpec']['maxRetryCount'],0)
            with tempfile.NamedTemporaryFile(mode='w',suffix='.sh') as script:
                script.write(module.script(args));script.flush()
                subprocess.run(['bash','-n',script.name],check=True)
        from unittest.mock import patch
        from experiments.leduc_poker.advantage_noise_intervention.source import fetch_source
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            (d/'SUCCESS.json').write_text(json.dumps({'training_states':[{'filename':'state_time_24h.pt',
                'relative_path':'states/state_time_24h.pt','sha256':'b'*64}]}))
            with patch('experiments.leduc_poker.advantage_noise_intervention.source.subprocess.run') as call:
                fetch_source('gs://source','exp35-example',0,d)
                remote=call.call_args_list[0].args[0][3]
                self.assertIn('task_002_grouped_wide_ucv_seed_145871',remote)

    def test_remote_receipts_follow_data_and_failed_upload_publishes_nothing(self):
        from unittest.mock import patch
        from experiments.leduc_poker.advantage_noise_intervention.worker import upload
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'rep_00').mkdir()
            (d/'rep_00/SUCCESS.json').write_text('{}');(d/'SUCCESS.json').write_text('{}')
            with patch.dict(os.environ,{'EXP52_REMOTE_WORKER':'gs://example/worker'}):
                with patch('experiments.leduc_poker.advantage_noise_intervention.worker.subprocess.run') as call:
                    upload(d)
                    commands=[r.args[0] for r in call.call_args_list]
                    self.assertEqual([c[2] for c in commands],['rsync','cp','cp'])
                    self.assertEqual(commands[-1][-1],'gs://example/worker/SUCCESS.json')
                with patch('experiments.leduc_poker.advantage_noise_intervention.worker.subprocess.run',
                           side_effect=RuntimeError('network failed')) as call:
                    with self.assertRaises(RuntimeError):upload(d)
                    self.assertEqual(call.call_count,1)


if __name__=='__main__':unittest.main()
