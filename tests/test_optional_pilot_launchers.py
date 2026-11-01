"""Exercise the real launchers with a fake gcloud; never submit cloud jobs."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from experiments.cloud_source_identity import source_identity


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {50: "critic_averaging_mechanism", 51: "critic_update_budget"}
FAKE_GCLOUD = '''
import json, os, shutil, sys
from pathlib import Path
root = Path(os.environ['FAKE_CLOUD_ROOT'])
args = sys.argv[1:]
with (root/'calls.jsonl').open('a') as handle:
    handle.write(json.dumps(args)+'\\n')
# Any pilot object read fails, including one accidentally introduced in a gate.
if args[:2] == ['storage', 'cp']:
    if not args[2].endswith('/smoke/SMOKE_SUCCESS.json'):
        sys.exit('Unexpected storage access: '+str(args))
    shutil.copyfile(root/'smoke_receipt.json', args[3])
elif args[:3] == ['batch', 'jobs', 'submit']:
    config = json.loads(Path(args[args.index('--config')+1]).read_text())
    stage = config['labels']['stage']
    (root/('job-'+args[3])).write_text(os.environ.get('FAKE_SMOKE_STATE', 'SUCCEEDED')
                                     if stage == 'smoke' else 'SUCCEEDED')
    (root/(stage+'.json')).write_text(json.dumps(config))
elif args[:3] == ['batch', 'jobs', 'describe']:
    state = root/('job-'+args[3])
    if not state.exists():
        sys.exit(1)
    print(state.read_text())
else:
    sys.exit('Unexpected gcloud command: '+str(args))
'''


class OptionalPilotLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()

    def launch(self, experiment, action, *, smoke_state='SUCCEEDED', smoke_status='complete',
               matching_source=True, attempt='initial', launcher_ref=None):
        name = EXPERIMENTS[experiment]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindir = root/'bin'; bindir.mkdir()
            # Isolate both cloud access and the launcher's generated specs.
            gcloud = bindir/'gcloud'
            gcloud.write_text('#!'+sys.executable+'\n'+FAKE_GCLOUD)
            gcloud.chmod(0o755)
            mktemp = bindir/'mktemp'
            mktemp.write_text('#!'+sys.executable+'\nimport os,tempfile\n'
                              'print(tempfile.mkdtemp(dir=os.environ["FAKE_CLOUD_ROOT"]))\n')
            mktemp.chmod(0o755)
            python = bindir/'python3'
            python.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' "$@"\n')
            python.chmod(0o755)
            (root/'smoke_receipt.json').write_text(json.dumps({'status': smoke_status,
                'source': source_identity(f'tests/test_{name}.py') if matching_source else 'old-code'}))
            env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                FAKE_CLOUD_ROOT=str(root), FAKE_SMOKE_STATE=smoke_state,
                PROJECT_ID='example', REGION='europe-west1', BUCKET='gs://example/results',
                SA_EMAIL='runner@example.com', REPO_REF=self.commit, RUN_ID=f'exp{experiment}-test',
                PARALLELISM='3', ATTEMPT=attempt, **{f'EXP{experiment}_REMOTE_CONTROLLER': '1'})
            # Full runs must not depend on pilot source variables either.
            env.pop('SOURCE_URI', None); env.pop('SOURCE_SHA256', None)
            env.pop('STUDY_REPOSITORY', None)
            env.pop('LAUNCHER_REF', None)
            if launcher_ref is not None:
                env['LAUNCHER_REF'] = launcher_ref
            if action == 'pilot-cloud':
                env.update(SOURCE_URI='gs://example/source.pt', SOURCE_SHA256='b'*64)
            result = subprocess.run(['bash', f'gcp/run_{name}.sh', action], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=30)
            calls = [json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()] \
                if (root/'calls.jsonl').exists() else []
            configs = {stage: json.loads((root/f'{stage}.json').read_text())
                for stage in ('controller', 'smoke', 'train', 'aggregate', 'pilot')
                if (root/f'{stage}.json').exists()}
        return result, calls, configs

    def test_run_and_resume_submit_without_completed_pilot(self):
        for experiment in EXPERIMENTS:
            for action, attempt in (('run', 'initial'), ('resume', 'retry1')):
                with self.subTest(experiment=experiment, action=action):
                    result, calls, configs = self.launch(experiment, action, attempt=attempt)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(set(configs), {'controller'})
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(calls[0][:3], ['batch', 'jobs', 'submit'])
                    self.assertIn(f'exp{experiment}-test-controller-{attempt}', calls[0])
                    script = configs['controller']['taskGroups'][0]['taskSpec']['runnables'][0]['script']['text']
                    self.assertIn('orchestrate', script)
                    self.assertIn(self.commit, script)

    def test_controller_orders_smoke_training_aggregation_without_pilot(self):
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                result, calls, configs = self.launch(experiment, 'orchestrate')
                self.assertEqual(result.returncode, 0, result.stderr)
                submits = [args[3] for args in calls if args[:3] == ['batch', 'jobs', 'submit']]
                self.assertEqual(submits, [f'exp{experiment}-test-{stage}-initial'
                                         for stage in ('smoke', 'train', 'aggregate')])
                self.assertFalse(any('/pilot/' in arg for args in calls for arg in args))
                self.assertEqual(configs['train']['taskGroups'][0]['taskCount'], 12)
                self.assertEqual(configs['train']['taskGroups'][0]['parallelism'], 3)
                smoke_script = configs['smoke']['taskGroups'][0]['taskSpec']['runnables'][0]['script']['text']
                self.assertIn('test_optional_pilot_launchers.py', smoke_script)

    def test_recovery_propagates_launcher_ref_without_changing_scientific_ref(self):
        infrastructure = 'b'*40
        for experiment in EXPERIMENTS:
            for action in ('resume', 'orchestrate'):
                with self.subTest(experiment=experiment, action=action):
                    result, _, configs = self.launch(experiment, action, attempt='retry1',
                                                     launcher_ref=infrastructure)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    for kind, config in configs.items():
                        text = config['taskGroups'][0]['taskSpec']['runnables'][0]['script']['text']
                        expected = infrastructure if kind == 'controller' else self.commit
                        self.assertIn(f'git -C "$REPOSITORY" checkout --detach {expected}', text)
                        if kind == 'controller':
                            self.assertIn(f'export LAUNCHER_REF={infrastructure}', text)
                            self.assertIn(f'REPO_REF={self.commit}', text)

    def test_invalid_launcher_ref_cannot_submit(self):
        for experiment in EXPERIMENTS:
            result, calls, _ = self.launch(experiment, 'run', launcher_ref='main')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('LAUNCHER_REF', result.stderr)
            self.assertEqual(calls, [])

    def test_failed_stale_or_incomplete_smoke_blocks_training(self):
        for experiment in EXPERIMENTS:
            for change in ({'smoke_state': 'FAILED'}, {'matching_source': False},
                           {'smoke_status': 'incomplete'}):
                with self.subTest(experiment=experiment, change=change):
                    result, calls, configs = self.launch(experiment, 'orchestrate', **change)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(set(configs), {'smoke'})

    def test_pilot_remains_available_and_resume_still_requires_new_attempt(self):
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                result, calls, configs = self.launch(experiment, 'pilot-cloud')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(set(configs), {'pilot'})
                result, calls, configs = self.launch(experiment, 'resume')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Resume requires a new ATTEMPT', result.stderr)
                self.assertEqual(calls, [])

    def test_shared_fingerprint_matches_existing_worker_definition(self):
        for name in EXPERIMENTS.values():
            digest = hashlib.sha256()
            for folder in ('adaptive_escher', 'unbiased_escher', 'vr_deep_cfr',
                           'escher_poker', 'experiments', 'gcp'):
                for path in sorted((ROOT/folder).rglob('*')):
                    if path.suffix in ('.py', '.sh'):
                        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
            for filename in ('requirements.txt', 'setup.py', 'pyproject.toml', f'tests/test_{name}.py'):
                path = ROOT/filename
                if path.is_file():
                    digest.update(filename.encode()); digest.update(path.read_bytes())
            expected = digest.hexdigest()
            self.assertEqual(source_identity(f'tests/test_{name}.py'), expected)
            worker = importlib.import_module(f'experiments.leduc_poker.{name}.worker')
            self.assertEqual(worker.source_identity(), expected)

    def test_controller_fingerprint_needs_no_installed_ml_dependencies(self):
        result = subprocess.run([sys.executable, '-S', '-c',
            'from experiments.cloud_source_identity import source_identity; '
            'print(source_identity("tests/test_critic_update_budget.py"))'],
            cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), source_identity('tests/test_critic_update_budget.py'))


if __name__ == '__main__':
    unittest.main()
