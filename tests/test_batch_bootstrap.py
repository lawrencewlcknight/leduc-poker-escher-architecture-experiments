"""Execute generated Batch startup on a fake reused VM; no network/cloud calls."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from experiments.cloud_source_identity import source_identity
from gcp.batch_bootstrap import bootstrap


ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {50: "critic_averaging_mechanism_batch", 51: "critic_update_budget_batch",
            52: "advantage_noise_intervention_batch"}
SCIENCE = "a" * 40
LAUNCHER = "b" * 40
FAKE_COMMAND = r'''
import json, os, sys
from pathlib import Path
root = Path(os.environ['FAKE_VM'])
command = Path(sys.argv[0]).name
args = sys.argv[1:]
with (root/'calls.jsonl').open('a') as handle:
    handle.write(json.dumps([command, *args])+'\n')
if command == 'sudo':
    os.execvp(args[0], args)
elif command == 'sleep':
    pass
elif command == 'apt-get':
    action = 'install' if 'install' in args else 'update'
    counter = root/('apt-'+action)
    count = int(counter.read_text())+1 if counter.exists() else 1
    counter.write_text(str(count))
    if count <= int(os.environ.get('FAIL_'+action.upper(), '0')):
        print('E: Could not get lock /var/lib/dpkg/lock-frontend (unattended-upgr)', file=sys.stderr)
        sys.exit(100)
elif command == 'git':
    if args[0] == 'clone':
        target = Path(args[-1])
        if target.exists() and any(target.iterdir()):
            sys.exit('fatal: destination path already exists and is not empty')
        target.mkdir(parents=True, exist_ok=True)
    elif args[0] == '-C':
        target = Path(args[1])
        if args[2:4] == ['checkout', '--detach']:
            (target/'.revision').write_text(args[4])
        elif args[2:] == ['rev-parse', 'HEAD']:
            print((target/'.revision').read_text())
        elif args[2:] == ['diff', '--exit-code', 'HEAD', '--']:
            if target.name == 'study' and os.environ.get('BROKEN_STUDY') == '1':
                print('deleted: historical scientific source', file=sys.stderr)
                sys.exit(1)
        else:
            sys.exit('Unexpected git arguments: '+str(args))
    else:
        sys.exit('Unexpected git arguments: '+str(args))
else:
    sys.exit('Unexpected command: '+command)
'''


def job(experiment, kind, launcher_ref=LAUNCHER):
    args = SimpleNamespace(kind=kind, repo_ref=SCIENCE, launcher_ref=launcher_ref,
        bucket="gs://example/results", run_id="test-run", source_bucket="gs://example/source",
        source_run_id="source-run", source_uri="gs://example/source.pt", source_sha256="c"*64,
        project="example", region="europe-west1", service_account="runner@example.com",
        parallelism=3, attempt="retry1")
    return importlib.import_module("gcp."+BUILDERS[experiment]).build_job(args)


def script(experiment, kind, launcher_ref=LAUNCHER):
    return job(experiment, kind, launcher_ref)["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]


class BatchBootstrapTests(unittest.TestCase):
    def test_recovery_checkout_can_fetch_historical_blobs_from_partial_clone(self):
        # A shared clone of a blob-filtered clone does not inherit its promisor
        # remote. Use a real file:// Git server with filtering enabled so this
        # catches missing historical objects, unlike our fake-git startup tests.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = root/'upstream'
            upstream.mkdir()
            git = shutil.which('git')
            env = dict(os.environ, GIT_AUTHOR_NAME='Test', GIT_AUTHOR_EMAIL='test@example.com',
                       GIT_COMMITTER_NAME='Test', GIT_COMMITTER_EMAIL='test@example.com')
            def run_git(*args):
                return subprocess.check_output([git, '-C', str(upstream), *args],
                    env=env, text=True, stderr=subprocess.PIPE).strip()
            run_git('init', '--quiet')
            run_git('config', 'uploadpack.allowFilter', 'true')
            (upstream/'gcp').mkdir()
            (upstream/'tests').mkdir()
            (upstream/'gcp'/'scientific.py').write_text('original scientific code\n')
            test_file = 'tests/test_critic_update_budget.py'
            (upstream/test_file).write_text('original scientific test\n')
            run_git('add', '.')
            run_git('commit', '--quiet', '-m', 'scientific revision')
            science = run_git('rev-parse', 'HEAD')
            expected = source_identity(test_file, upstream)
            (upstream/'gcp'/'scientific.py').write_text('later launcher revision\n')
            (upstream/test_file).write_text('later launcher test\n')
            run_git('add', '.')
            run_git('commit', '--quiet', '-m', 'launcher revision')
            launcher = run_git('rev-parse', 'HEAD')
            fake_env = self.fake_environment(root)
            # Only apt/sudo/sleep are faked; cloning, filtering and checkout are real.
            fake_git = root/'bin'/'git'
            fake_git.unlink()
            fake_git.symlink_to(git)
            preamble = bootstrap(50, upstream.as_uri(), science,
                                 controller=True, launcher_ref=launcher)
            preamble += '\nprintf "STUDY_CHECKOUT=%s\\n" "$STUDY_REPOSITORY"\n'
            result = subprocess.run(['bash', '-c', preamble], env=fake_env, text=True,
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            study = Path(next(line.split('=', 1)[1] for line in result.stdout.splitlines()
                              if line.startswith('STUDY_CHECKOUT=')))
            self.assertEqual(source_identity(test_file, study), expected, result.stderr)
            self.assertEqual(subprocess.check_output([git, '-C', str(study), 'status',
                '--porcelain'], text=True), '')
            self.assertEqual(subprocess.check_output([git, '-C', str(study), 'rev-parse',
                'HEAD'], text=True).strip(), science)

    def fake_environment(self, root, **extra):
        bindir = root/'bin'
        bindir.mkdir(exist_ok=True)
        for name in ('sudo', 'apt-get', 'git', 'sleep'):
            path = bindir/name
            path.write_text('#!'+sys.executable+'\n'+FAKE_COMMAND)
            path.chmod(0o755)
        return dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
            FAKE_VM=str(root), BATCH_WORK_ROOT=str(root/'workspace with spaces'),
            TEST_PYTHON=sys.executable, **extra)

    def startup(self, experiment, kind, env):
        # This is the unmodified generated preamble, including apt and git.
        preamble = script(experiment, kind).split('\nBUCKET_ROOT=', 1)[0]
        preamble += '''
export WORK REPOSITORY OUTPUT SOURCE
"$TEST_PYTHON" -c 'import os,json; print(json.dumps({k:os.environ.get(k) for k in ("WORK","REPOSITORY","OUTPUT","SOURCE","STUDY_REPOSITORY")}))'
'''
        return subprocess.run(['bash', '-c', preamble], env=env, text=True,
                              capture_output=True, timeout=30)

    def test_successive_tasks_and_same_task_retry_are_isolated(self):
        for experiment in BUILDERS:
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                env = self.fake_environment(root)
                old = Path(env['BATCH_WORK_ROOT'])/f'exp{experiment}'/'repository'
                old.mkdir(parents=True)
                (old/'preserve.txt').write_text('prior task results')
                workspaces = []
                for index in ('0', '3', '3'):
                    env['BATCH_TASK_INDEX'] = index
                    result = self.startup(experiment, 'worker' if experiment == 52 else 'train', env)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    paths = json.loads(result.stdout.splitlines()[-1])
                    workspace = Path(paths['WORK'])
                    workspaces.append(workspace)
                    self.assertEqual((Path(paths['REPOSITORY'])/'.revision').read_text(), SCIENCE)
                    self.assertTrue(Path(paths['OUTPUT']).is_dir())
                    self.assertFalse(Path(paths['SOURCE']).exists())
                    # A later task must not see this source/output contamination.
                    Path(paths['SOURCE']).mkdir()
                    (Path(paths['SOURCE'])/'source.pt').write_text('previous source')
                    (Path(paths['OUTPUT'])/'result.json').write_text('previous result')
                self.assertEqual(len(set(workspaces)), 3)
                self.assertEqual((old/'preserve.txt').read_text(), 'prior task results')

    def test_package_locks_are_retried_for_update_and_install(self):
        for experiment in BUILDERS:
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = self.startup(experiment, 'controller',
                    self.fake_environment(root, FAIL_UPDATE='2', FAIL_INSTALL='1'))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Package setup attempt 1/5 failed (exit 100)', result.stderr)
                calls = [json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
                apt = [row for row in calls if row[0] == 'apt-get']
                self.assertEqual(len(apt), 5)
                self.assertTrue(all('DPkg::Lock::Timeout=300' in row and 'Acquire::Retries=3' in row for row in apt))
                self.assertEqual(sum(row[0] == 'sleep' for row in calls), 3)
                paths = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual((Path(paths['REPOSITORY'])/'.revision').read_text(), LAUNCHER)
                self.assertEqual((Path(paths['STUDY_REPOSITORY'])/'.revision').read_text(), SCIENCE)

    def test_persistent_package_failure_is_bounded_and_prevents_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.startup(51, 'controller', self.fake_environment(root, FAIL_INSTALL='99'))
            self.assertEqual(result.returncode, 100)
            self.assertEqual((root/'apt-install').read_text(), '5')
            calls = [json.loads(line) for line in (root/'calls.jsonl').read_text().splitlines()]
            self.assertEqual(sum(row[0] == 'sleep' for row in calls), 4)
            self.assertFalse(any(row[0] == 'git' for row in calls))

    def test_successful_git_exit_with_incomplete_study_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.startup(50, 'controller', self.fake_environment(root, BROKEN_STUDY='1'))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Incomplete or modified checkout', result.stderr)
            self.assertNotIn('Verified scientific checkout', result.stdout)

    def test_all_job_stages_parse_and_pin_the_correct_revision(self):
        for experiment in BUILDERS:
            worker = 'worker' if experiment == 52 else 'train'
            for kind in ('controller', 'smoke', 'pilot', worker, 'aggregate'):
                with self.subTest(experiment=experiment, kind=kind):
                    text = script(experiment, kind)
                    result = subprocess.run(['bash', '-n'], input=text, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    expected = LAUNCHER if kind == 'controller' else SCIENCE
                    self.assertIn(f'git -C "$REPOSITORY" checkout --detach {expected}', text)
                    self.assertIn('mktemp -d', text)
                    if kind == 'controller':
                        self.assertIn('export LAUNCHER_REF='+LAUNCHER, text)
                        self.assertIn('REPO_REF='+SCIENCE, text)
                        self.assertIn('git clone --filter=blob:none --no-checkout', text)
                        self.assertIn('verify_checkout "$WORK/study"', text)
                        self.assertNotIn('git clone --shared', text)
                    else:
                        self.assertIn('uv venv --python 3.9 --seed "$WORK/venv"', text)
                        self.assertNotIn('/tmp/exp', text)
            group = job(experiment, worker)['taskGroups'][0]
            self.assertEqual(group['taskCount'], 6 if experiment == 52 else 12)
            self.assertEqual(group['parallelism'], 3)
            # Recovery remains explicit; do not silently retry scientific failures.
            self.assertEqual(group['taskSpec']['maxRetryCount'], 0)
            text = script(experiment, 'controller', launcher_ref=None)
            self.assertIn(f'git -C "$REPOSITORY" checkout --detach {SCIENCE}', text)
            self.assertNotIn('git clone --shared', text)

    def test_scientific_fingerprint_reads_explicit_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'gcp').mkdir()
            (root/'gcp'/'original.py').write_text('original code\n')
            first = source_identity('tests/test_critic_update_budget.py', root)
            self.assertNotEqual(first, source_identity('tests/test_critic_update_budget.py'))
            (root/'gcp'/'original.py').write_text('changed code\n')
            self.assertNotEqual(first, source_identity('tests/test_critic_update_budget.py', root))

    def test_exp52_recovery_controller_keeps_pilot_and_scientific_refs(self):
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        fake_cloud = r'''
import json, os, shutil, sys
from pathlib import Path
root=Path(os.environ['FAKE_CLOUD'])
args=sys.argv[1:]
if args[:2]==['storage','cp']:
    kind='pilot' if '/pilot/' in args[2] else 'smoke'
    shutil.copyfile(root/(kind+'.json'), args[3])
elif args[:3]==['batch','jobs','submit']:
    config=json.loads(Path(args[args.index('--config')+1]).read_text())
    (root/(config['labels']['stage']+'-job.json')).write_text(json.dumps(config))
    (root/args[3]).write_text('SUCCEEDED')
elif args[:3]==['batch','jobs','describe']:
    path=root/args[3]
    if not path.exists(): sys.exit(1)
    print(path.read_text())
else: sys.exit('Unexpected gcloud arguments: '+str(args))
'''
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            bindir=root/'bin'; bindir.mkdir()
            gcloud=bindir/'gcloud'
            gcloud.write_text('#!'+sys.executable+'\n'+fake_cloud); gcloud.chmod(0o755)
            python=bindir/'python3'
            python.symlink_to(sys.executable)
            pilot={'status':'complete','code':'original-science','runtime':
                dict.fromkeys(('python','torch','numpy','open_spiel','platform'), 'same'),
                'result':{'pilot':True,'smoke':False,'fits':6,'fit_steps':750}}
            (root/'pilot.json').write_text(json.dumps(pilot))
            (root/'smoke.json').write_text(json.dumps(pilot))
            env=dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
                FAKE_CLOUD=str(root), PROJECT_ID='example', REGION='europe-west1',
                BUCKET='gs://example/results', RUN_ID='exp52-test', SA_EMAIL='runner@example.com',
                REPO_REF=commit, LAUNCHER_REF=LAUNCHER, ATTEMPT='retry1', PARALLELISM='3',
                EXP52_REMOTE_CONTROLLER='1')
            for action in ('resume','orchestrate'):
                result=subprocess.run(['bash','gcp/run_advantage_noise_intervention.sh',action],
                    cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
            for kind in ('controller','smoke','worker','aggregate'):
                config=json.loads((root/(kind+'-job.json')).read_text())
                text=config['taskGroups'][0]['taskSpec']['runnables'][0]['script']['text']
                expected=LAUNCHER if kind=='controller' else commit
                self.assertIn(f'git -C "$REPOSITORY" checkout --detach {expected}',text)
            # The existing scientific compatibility check must still reject a mismatch.
            pilot['code']='different-science'
            (root/'pilot.json').write_text(json.dumps(pilot))
            result=subprocess.run(['bash','gcp/run_advantage_noise_intervention.sh','orchestrate'],
                cwd=ROOT,env=env,text=True,capture_output=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Pilot and study code differ',result.stderr)


if __name__ == '__main__':
    unittest.main()
