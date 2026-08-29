from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_batch_training_job_is_one_task_per_vm_and_uses_array_index(tmp_path):
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
