from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    ALGORITHM_ORDER,
    HELDOUT_SEEDS,
    SMOKE_SEEDS,
    TARGET_ACTIVE_SECONDS,
    TARGET_NODES,
    task_schedule,
    validate_contract,
)


def _batch_builder_module():
    path = Path(__file__).resolve().parents[1] / "gcp" / "four_algorithm_heldout_batch.py"
    spec = importlib.util.spec_from_file_location("heldout_batch_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frozen_task_schedule_is_stable_and_complete():
    schedule = task_schedule()
    assert len(schedule) == 32
    assert len(set(schedule)) == 32
    assert schedule[:8] == [(ALGORITHM_ORDER[0], seed) for seed in HELDOUT_SEEDS]
    assert schedule[-8:] == [(ALGORITHM_ORDER[-1], seed) for seed in HELDOUT_SEEDS]


def test_contract_separates_smoke_and_heldout_labels():
    validate_contract(
        seeds=HELDOUT_SEEDS,
        target_nodes=TARGET_NODES,
        target_seconds=TARGET_ACTIVE_SECONDS,
        smoke=False,
    )
    validate_contract(seeds=SMOKE_SEEDS, target_nodes=1, target_seconds=0, smoke=True)
    with pytest.raises(ValueError, match="held-out"):
        validate_contract(
            seeds=(HELDOUT_SEEDS[0],), target_nodes=1, target_seconds=0, smoke=True
        )
    with pytest.raises(ValueError, match="frozen eight"):
        validate_contract(
            seeds=tuple(reversed(HELDOUT_SEEDS)),
            target_nodes=TARGET_NODES,
            target_seconds=TARGET_ACTIVE_SECONDS,
            smoke=False,
        )


def test_batch_training_job_is_one_task_per_vm_and_uses_array_index():
    builder = _batch_builder_module()
    args = type(
        "Args",
        (),
        {
            "kind": "train",
            "arch_repo_url": builder.ARCH_REPO_URL,
            "deep_repo_url": builder.DEEP_REPO_URL,
            "arch_repo_ref": "a" * 40,
            "deep_repo_ref": "b" * 40,
            "bucket_root": "gs://example/results",
            "run_id": "test-run",
            "parallelism": 32,
            "max_retries": 0,
            "service_account": "batch@example.iam.gserviceaccount.com",
            "provisioning_model": "STANDARD",
        },
    )()
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == 32
    assert group["parallelism"] == 32
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["computeResource"] == {
        "cpuMilli": 8000,
        "memoryMib": 30000,
    }
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == (
        "n2-standard-8"
    )
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "BATCH_TASK_INDEX" in script
    assert "SUCCESS.json" in script
    assert "--task-index" in script
    assert "gcloud storage rsync" in script


def test_batch_smoke_runs_all_four_implementations_before_production():
    builder = _batch_builder_module()
    args = type(
        "Args",
        (),
        {
            "kind": "smoke",
            "arch_repo_url": builder.ARCH_REPO_URL,
            "deep_repo_url": builder.DEEP_REPO_URL,
            "arch_repo_ref": "a" * 40,
            "deep_repo_ref": "b" * 40,
            "bucket_root": "gs://example/results",
            "run_id": "test-run",
            "parallelism": 32,
            "max_retries": 0,
            "service_account": "batch@example.iam.gserviceaccount.com",
            "provisioning_model": "STANDARD",
        },
    )()
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    assert group["taskCount"] == 1
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "four_algorithm_heldout_benchmark.run smoke" in script
    assert "--no-resume" in script
    # Batch does not guarantee HOME in a runnable's environment. Keep the
    # bootstrap self-contained under /tmp so `set -u` cannot fail after the uv
    # installer succeeds.
    assert "$HOME" not in script
    assert "UV_INSTALL_DIR=/tmp/uv-bin" in script
    assert "UV_NO_MODIFY_PATH=1" in script
    assert 'export PATH="$UV_INSTALL_DIR:$PATH"' in script


def test_remote_controller_owns_the_full_cloud_sequence():
    builder = _batch_builder_module()
    args = type(
        "Args",
        (),
        {
            "kind": "controller",
            "arch_repo_url": builder.ARCH_REPO_URL,
            "deep_repo_url": builder.DEEP_REPO_URL,
            "arch_repo_ref": "a" * 40,
            "deep_repo_ref": "b" * 40,
            "bucket_root": "gs://example/results",
            "run_id": "test-run",
            "parallelism": 32,
            "max_retries": 2,
            "service_account": "batch@example.iam.gserviceaccount.com",
            "provisioning_model": "SPOT",
            "project_id": "example-project",
            "region": "europe-west2",
            "controller_action": "orchestrate",
        },
    )()
    job = builder.build_job(args)
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 1
    assert group["parallelism"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "604800s"
    assert group["taskSpec"]["maxRetryCount"] == 2
    assert policy["machineType"] == "e2-small"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "HELDOUT_REMOTE_CONTROLLER=1" in script
    assert 'CONTROLLER_ACTION=orchestrate' in script
    assert (
        'exec bash gcp/run_four_algorithm_heldout_benchmark.sh "$CONTROLLER_ACTION"'
        in script
    )
    assert "pip install" not in script


def test_local_run_submits_controller_without_waiting_for_children():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "gcp"
        / "run_four_algorithm_heldout_benchmark.sh"
    ).read_text(encoding="utf-8")
    run_case = launcher.split("  run)\n", 1)[1].split("  orchestrate)\n", 1)[0]
    assert 'submit_job "$CONTROLLER_JOB" "$TEMP_DIR/controller.json"' in run_case
    assert "wait_for_job" not in run_case
    assert "The laptop may now be disconnected or switched off." in run_case


def test_run_command_returns_after_one_controller_submission(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "gcloud.log"
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_GCLOUD_LOG"\n',
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_GCLOUD_LOG": str(invocation_log),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west2",
        "BUCKET": "gs://example/results",
        "SA_EMAIL": "batch@example.iam.gserviceaccount.com",
        "ARCH_REPO_REF": "a" * 40,
        "DEEP_CFR_REPO_REF": "b" * 40,
        "RUN_ID": "test-remote-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_four_algorithm_heldout_benchmark.sh", "run"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "batch jobs submit test-remote-run-controller" in invocations[0]
    assert "The laptop may now be disconnected or switched off." in completed.stdout


def test_remote_orchestrator_submits_all_stages_in_order(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    submitted_jobs = tmp_path / "submitted.log"
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        """#!/bin/sh
if [ "$1 $2 $3" = "batch jobs submit" ]; then
  printf "%s\\n" "$4" >> "$FAKE_SUBMITTED_JOBS"
  exit 0
fi
if [ "$1 $2 $3" = "batch jobs describe" ]; then
  if grep -qx "$4" "$FAKE_SUBMITTED_JOBS" 2>/dev/null; then
    printf "SUCCEEDED\\n"
    exit 0
  fi
  exit 1
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SUBMITTED_JOBS": str(submitted_jobs),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west2",
        "BUCKET": "gs://example/results",
        "SA_EMAIL": "batch@example.iam.gserviceaccount.com",
        "ARCH_REPO_REF": "a" * 40,
        "DEEP_CFR_REPO_REF": "b" * 40,
        "RUN_ID": "test-remote-run",
        "HELDOUT_REMOTE_CONTROLLER": "1",
    }
    subprocess.run(
        ["bash", "gcp/run_four_algorithm_heldout_benchmark.sh", "orchestrate"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert submitted_jobs.read_text(encoding="utf-8").splitlines() == [
        "test-remote-run-smoke",
        "test-remote-run-train",
        "test-remote-run-aggregate",
    ]


def test_remote_resume_replaces_a_failed_smoke_before_training(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    submitted_jobs = tmp_path / "submitted.log"
    fake_gcloud = fake_bin / "gcloud"
    fake_gcloud.write_text(
        """#!/bin/sh
if [ "$1 $2 $3" = "batch jobs submit" ]; then
  printf "%s\\n" "$4" >> "$FAKE_SUBMITTED_JOBS"
  exit 0
fi
if [ "$1 $2 $3" = "batch jobs describe" ]; then
  if [ "$4" = "test-remote-run-smoke" ]; then
    printf "FAILED\\n"
    exit 0
  fi
  if grep -qx "$4" "$FAKE_SUBMITTED_JOBS" 2>/dev/null; then
    printf "SUCCEEDED\\n"
    exit 0
  fi
  exit 1
fi
exit 2
""",
        encoding="utf-8",
    )
    fake_gcloud.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SUBMITTED_JOBS": str(submitted_jobs),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west2",
        "BUCKET": "gs://example/results",
        "SA_EMAIL": "batch@example.iam.gserviceaccount.com",
        "ARCH_REPO_REF": "a" * 40,
        "DEEP_CFR_REPO_REF": "b" * 40,
        "RUN_ID": "test-remote-run",
        "RESUME_TAG": "123456",
        "HELDOUT_REMOTE_CONTROLLER": "1",
    }
    subprocess.run(
        ["bash", "gcp/run_four_algorithm_heldout_benchmark.sh", "orchestrate-resume"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    assert submitted_jobs.read_text(encoding="utf-8").splitlines() == [
        "test-remote-run-smoke-retry-123456",
        "test-remote-run-retry-123456",
        "test-remote-run-reaggregate-123456",
    ]
