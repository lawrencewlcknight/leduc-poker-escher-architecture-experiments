from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.leduc_poker.deep_cfr_ucv_36h_plateau.config import (
    ALGORITHM_ORDER,
    CHECKPOINT_TARGET_HOURS,
    MAX_DEEP_CFR_ITERATIONS,
    MAX_UCV_ITERATIONS,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    validate_contract,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    HELDOUT_SEEDS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _batch_builder_module():
    path = REPOSITORY / "gcp" / "deep_cfr_ucv_36h_batch.py"
    spec = importlib.util.spec_from_file_location("exp21_batch_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str, *, parallelism: int = 10):
    return SimpleNamespace(
        kind=kind,
        arch_repo_url=builder.ARCH_REPO_URL,
        deep_repo_url=builder.DEEP_REPO_URL,
        arch_repo_ref="a" * 40,
        deep_repo_ref="b" * 40,
        bucket_root="gs://example/results",
        run_id="exp21-test",
        parallelism=parallelism,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west2",
        controller_action="orchestrate",
    )


def test_frozen_seed_task_and_checkpoint_contract():
    assert PRODUCTION_SEEDS == tuple(HELDOUT_SEEDS[:5])
    assert PRODUCTION_SEEDS == (104729, 130363, 155921, 181081, 205759)
    assert CHECKPOINT_TARGET_HOURS == tuple(range(2, 37, 2))
    schedule = checkpoint_schedule()
    assert len(schedule) == 18
    assert schedule[0]["checkpoint_id"] == "time_02h"
    assert schedule[-1] == {
        "checkpoint_id": "time_36h",
        "target_active_seconds": 129600.0,
        "target_active_hours": 36.0,
    }
    tasks = task_schedule()
    assert len(tasks) == 10
    assert len(set(tasks)) == 10
    assert tasks[:5] == tuple((ALGORITHM_ORDER[0], seed) for seed in PRODUCTION_SEEDS)
    assert tasks[5:] == tuple((ALGORITHM_ORDER[1], seed) for seed in PRODUCTION_SEEDS)
    assert MAX_DEEP_CFR_ITERATIONS == 18_000
    assert MAX_UCV_ITERATIONS == 600


def test_contract_rejects_result_selected_seeds_or_changed_schedule():
    schedule = checkpoint_schedule()
    validate_contract(seeds=PRODUCTION_SEEDS, schedule=schedule, smoke=False)
    validate_contract(
        seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True
    )
    with pytest.raises(ValueError, match="Production seeds"):
        validate_contract(
            seeds=tuple(reversed(PRODUCTION_SEEDS)), schedule=schedule, smoke=False
        )
    changed = [dict(row) for row in schedule]
    changed[-1]["target_active_seconds"] -= 1
    with pytest.raises(ValueError, match="Checkpoint schedule"):
        validate_contract(seeds=PRODUCTION_SEEDS, schedule=changed, smoke=False)


def test_production_batch_is_ten_standard_n2_workers_with_hard_limit():
    builder = _batch_builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 10
    assert group["parallelism"] == 10
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "180000s"
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["computeResource"] == {
        "cpuMilli": 8000,
        "memoryMib": 30000,
    }
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "BATCH_TASK_INDEX" in script
    assert "deep_cfr_ucv_36h_plateau.run worker" in script
    assert "SUCCESS.json" in script
    assert "gcloud storage rsync" in script
    assert "$HOME" not in script


def test_smoke_and_aggregate_jobs_run_the_expected_commands():
    builder = _batch_builder_module()
    smoke = builder.build_job(_builder_args(builder, "smoke"))["taskGroups"][0]
    smoke_script = smoke["taskSpec"]["runnables"][0]["script"]["text"]
    assert smoke["taskCount"] == 1
    assert "deep_cfr_ucv_36h_plateau.run smoke" in smoke_script
    assert "--no-resume" in smoke_script

    aggregate = builder.build_job(_builder_args(builder, "aggregate"))["taskGroups"][0]
    aggregate_script = aggregate["taskSpec"]["runnables"][0]["script"]["text"]
    assert "deep_cfr_ucv_36h_plateau.run aggregate" in aggregate_script
    assert '"$BUCKET_ROOT/$RUN_ID/workers"' in aggregate_script
    assert '"$BUCKET_ROOT/$RUN_ID/analysis"' in aggregate_script


def test_remote_controller_owns_the_cloud_sequence():
    builder = _batch_builder_module()
    job = builder.build_job(_builder_args(builder, "controller"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "604800s"
    assert group["taskSpec"]["maxRetryCount"] == 2
    assert policy["machineType"] == "e2-small"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "EXP21_REMOTE_CONTROLLER=1" in script
    assert 'exec bash gcp/run_deep_cfr_ucv_36h_plateau.sh "$CONTROLLER_ACTION"' in script


def test_local_run_returns_after_controller_submission(tmp_path):
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
        "RUN_ID": "exp21-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_deep_cfr_ucv_36h_plateau.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert "batch jobs submit exp21-test-run-controller" in invocations[0]
    assert "The laptop may now be disconnected or switched off." in completed.stdout


def test_readme_orders_experiment_21_after_experiment_20_and_documents_outputs():
    root_readme = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    assert root_readme.index("## Experiment 20:") < root_readme.index(
        "## Experiment 21:"
    )
    section = root_readme.split("## Experiment 21:", 1)[1]
    assert "./gcp/run_deep_cfr_ucv_36h_plateau.sh run" in section
    assert "./gcp/run_deep_cfr_ucv_36h_plateau.sh smoke-local" in section
    assert "exploitability_by_training_time.png" in section
    assert "exploitability_by_nodes_touched.png" in section
