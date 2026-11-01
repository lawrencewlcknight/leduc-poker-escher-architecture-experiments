"""Runtime regression coverage for metadata captured by Batch shell scripts."""
from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from experiments import cloud_worker_identity as identity


ROOT = Path(__file__).resolve().parents[1]
BUILDERS = {
    50: "critic_averaging_mechanism_batch",
    51: "critic_update_budget_batch",
    52: "advantage_noise_intervention_batch",
}


def generated_script(experiment, kind):
    builder = importlib.import_module("gcp." + BUILDERS[experiment])
    args = SimpleNamespace(kind=kind, bucket="gs://example/results", run_id="test-run",
        source_bucket="gs://example/sources", source_run_id="source-run", repo_ref="a"*40,
        project="example", region="europe-west1", service_account="runner@example.com",
        parallelism=3, attempt="initial", source_uri="gs://example/source.pt", source_sha256="b"*64)
    return builder.script(args)


class WorkerIdentityTests(unittest.TestCase):
    def test_noisy_import_and_schedule_diagnostics_go_to_stderr(self):
        warning = "Optional module pokerkit_wrapper was not importable: No module named 'pokerkit'"
        for experiment, row, expected in ((50, ("A", 123), "task_000_A_123"),
                                          (51, ("D", 456), "task_000_D_456"),
                                          (52, (789, "time_24h"), "seed_789_time_24h")):
            def noisy_import(_name):
                print(warning)
                def tasks():
                    print("schedule diagnostic")
                    return [row]
                return SimpleNamespace(tasks=tasks)
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(experiment=experiment), redirect_stdout(stdout), redirect_stderr(stderr), \
                    mock.patch.object(identity.importlib, "import_module", side_effect=noisy_import):
                identity.main(["--experiment", str(experiment), "--task-index", "0",
                               "--remote-root", "gs://example/run/pilot"])
            self.assertEqual(stdout.getvalue(), expected + "\n")
            self.assertEqual(stderr.getvalue(), warning + "\nschedule diagnostic\n")

    def test_invalid_inputs_never_emit_a_worker_name(self):
        for extra in (("--task-index", "-1"), ("--task-index", "1"),
                      ("--remote-root", "gs://example/run\nwarning"), ("--remote-root", "")):
            stdout, stderr = io.StringIO(), io.StringIO()
            with self.subTest(extra=extra), redirect_stdout(stdout), redirect_stderr(stderr), \
                    mock.patch.object(identity.importlib, "import_module",
                                      return_value=SimpleNamespace(tasks=lambda: [("A", 123)])):
                with self.assertRaises(SystemExit) as error:
                    identity.main(["--experiment", "50", "--task-index", "0",
                                   "--remote-root", "gs://example/run", *extra])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(stdout.getvalue(), "")

    def test_failed_noisy_import_emits_no_metadata(self):
        def broken_import(_name):
            print("import diagnostic")
            raise ImportError("required dependency missing")
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), \
                mock.patch.object(identity.importlib, "import_module", side_effect=broken_import):
            with self.assertRaises(ImportError):
                identity.main(["--experiment", "52", "--task-index", "0",
                               "--remote-root", "gs://example/run"])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "import diagnostic\n")

    def test_uri_and_schedule_validation(self):
        for uri in ("", "https://example/run", "gs://example", "gs://example/a b",
                    "gs://example/run\n", "gs://example/run\x00", "gs://example//run",
                    "gs://example/../run", "gs://example/run?x=1"):
            with self.subTest(uri=uri), self.assertRaises(ValueError):
                identity.validate_worker_uri(uri)
        self.assertEqual(identity.validate_worker_uri("gs://example/prefix/run/workers/task_000_A_123"),
                         "gs://example/prefix/run/workers/task_000_A_123")
        for experiment, row in ((50, ("A\nwarning", 1)), (51, ("A", "1\nwarning")),
                                (52, (123, "time_24h\nwarning")), (52, (-1, "time_24h"))):
            with self.subTest(experiment=experiment, row=row), \
                    mock.patch.object(identity.importlib, "import_module",
                                      return_value=SimpleNamespace(tasks=lambda: [row])):
                with self.assertRaises(ValueError):
                    identity.worker_name(experiment, 0)

    def test_every_identity_matches_existing_worker_schedule(self):
        for experiment, module in identity.CONFIG_MODULES.items():
            schedule = importlib.import_module(module).tasks()
            for index, (first, second) in enumerate(schedule):
                expected = (f"seed_{first}_{second}" if experiment == 52
                            else f"task_{index:03d}_{first}_{second}")
                with self.subTest(experiment=experiment, index=index):
                    self.assertEqual(identity.worker_name(experiment, index), expected)
            with self.assertRaises(ValueError):
                identity.worker_name(experiment, len(schedule))

    def run_metadata(self, experiment, kind, index=0, bucket="gs://example/results"):
        script = generated_script(experiment, kind)
        # Execute the real generated metadata commands in a fresh Python process,
        # but stop before mkdir/download/training or any gcloud invocation.
        metadata = script[script.index('\nINDEX='):script.index('\nmkdir -p "$LOCAL_WORKER"')]
        shell = 'set -euo pipefail\npython() { "$TEST_PYTHON" "$@"; }\n' + metadata
        shell += f'\nprintf "%s\\n" "$TASK" "$LOCAL_WORKER" "$EXP{experiment}_REMOTE_WORKER"\n'
        env = dict(os.environ, TEST_PYTHON=sys.executable, BATCH_TASK_INDEX=str(index),
                   BUCKET_ROOT=bucket, RUN_ID="test-run", OUTPUT="/tmp/output with spaces")
        return subprocess.run(["bash", "-c", shell], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=60)

    def test_generated_shell_produces_clean_canonical_paths_in_fresh_process(self):
        for experiment, kind, count in ((50, "train", 12), (51, "train", 12),
                                         (52, "worker", 6), (52, "pilot", 1)):
            for index in sorted({0, count-1}):
                with self.subTest(experiment=experiment, kind=kind, index=index):
                    result = self.run_metadata(experiment, kind, index)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    name = identity.worker_name(experiment, index)
                    remote = "gs://example/results/test-run" + ("/pilot" if kind == "pilot" else "")
                    self.assertEqual(result.stdout.splitlines(), [name,
                        f"/tmp/output with spaces/workers/{name}", f"{remote}/workers/{name}"])

    def test_generated_shell_stops_on_bad_destination_before_work(self):
        for experiment, kind in ((50, "train"), (51, "train"), (52, "pilot"), (52, "worker")):
            with self.subTest(experiment=experiment, kind=kind):
                result = self.run_metadata(experiment, kind, bucket="gs://example/warning\nrun")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertIn("Invalid worker URI", result.stderr)
        for kind in ("pilot", "worker"):
            script = generated_script(52, kind)
            self.assertLess(script.index("experiments.cloud_worker_identity"), script.index(" fetch-source "))
        for experiment in BUILDERS:
            self.assertIn("test_cloud_worker_identity.py", generated_script(experiment, "smoke"))

    def test_workers_reject_bad_destination_before_loading_or_training(self):
        for experiment, config_module in identity.CONFIG_MODULES.items():
            worker = importlib.import_module(config_module.rsplit(".", 1)[0] + ".worker")
            expensive = "load_source" if experiment == 52 else "make_solver"
            for bad_uri in ("", "gs://example/run\noptional-module-warning"):
                with self.subTest(experiment=experiment, uri=bad_uri), tempfile.TemporaryDirectory() as root, \
                        mock.patch.dict(os.environ, {f"EXP{experiment}_REMOTE_WORKER": bad_uri}), \
                        mock.patch.object(worker, expensive) as load:
                    output = Path(root)/"not-created"
                    with self.assertRaisesRegex(ValueError, "Invalid worker URI"):
                        if experiment == 52:
                            worker.run_worker(output, smoke=True)
                        else:
                            arm, seed = importlib.import_module(config_module).tasks(True)[0]
                            worker.run_worker(output, arm, seed, smoke=True)
                    load.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
