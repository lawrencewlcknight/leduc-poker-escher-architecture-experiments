from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import (
    HELDOUT_SEEDS,
)
from experiments.leduc_poker.ucv_three_arm_15m_simplification.config import (
    FIXED_BETA_ONE,
    FULL_EXPERIMENT_6,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    TARGET_NODES,
    TWO_CROSS_FITTED_CRITICS,
    VARIANT_ORDER,
    task_schedule,
    validate_contract,
    variant_config,
)
from experiments.leduc_poker.ucv_three_arm_15m_simplification.diagnostics import (
    OnlineMoments,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "ucv_three_arm_15m_batch.py"
    spec = importlib.util.spec_from_file_location("exp22_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str, parallelism: int = 18):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        bucket_root="gs://example/results",
        run_id="exp22-test",
        parallelism=parallelism,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_frozen_three_arm_six_seed_contract():
    assert VARIANT_ORDER == (
        FULL_EXPERIMENT_6,
        FIXED_BETA_ONE,
        TWO_CROSS_FITTED_CRITICS,
    )
    assert PRODUCTION_SEEDS == (452106, 864014, 716235, 928759, 809334, 945659)
    assert not set(PRODUCTION_SEEDS).intersection(HELDOUT_SEEDS)
    tasks = task_schedule()
    assert len(tasks) == 18
    assert len(set(tasks)) == 18
    assert tasks[:6] == tuple((FULL_EXPERIMENT_6, seed) for seed in PRODUCTION_SEEDS)
    assert tasks[-6:] == tuple(
        (TWO_CROSS_FITTED_CRITICS, seed) for seed in PRODUCTION_SEEDS
    )
    validate_contract(seeds=PRODUCTION_SEEDS, target_nodes=TARGET_NODES, smoke=False)
    validate_contract(seeds=SMOKE_SEEDS, target_nodes=100, smoke=True)


def test_contract_rejects_changed_seed_order_or_budget():
    with pytest.raises(ValueError, match="Production seeds"):
        validate_contract(
            seeds=tuple(reversed(PRODUCTION_SEEDS)),
            target_nodes=TARGET_NODES,
            smoke=False,
        )
    with pytest.raises(ValueError, match="Node target"):
        validate_contract(
            seeds=PRODUCTION_SEEDS, target_nodes=TARGET_NODES - 1, smoke=False
        )


def test_variant_overrides_are_one_factor_only():
    full = variant_config(FULL_EXPERIMENT_6)
    fixed = variant_config(FIXED_BETA_ONE)
    two = variant_config(TWO_CROSS_FITTED_CRITICS)
    assert fixed["fixed_control_variate_beta"] == 1.0
    assert two["q_ensemble_size"] == 2
    assert full["fixed_control_variate_beta"] is None
    assert full["q_ensemble_size"] == 3
    assert {key for key in full if full[key] != fixed[key]} == {
        "fixed_control_variate_beta"
    }
    assert {key for key in full if full[key] != two[key]} == {"q_ensemble_size"}


def test_online_moments_are_numerically_correct():
    moments = OnlineMoments()
    for value in (1.0, 2.0, 3.0):
        moments.add(value)
    assert moments.count == 3
    assert moments.mean == 2.0
    assert moments.variance == 1.0
    assert moments.mean_absolute == 2.0
    assert moments.minimum == 1.0
    assert moments.maximum == 3.0


def test_production_batch_has_18_standard_n2_workers_and_hard_limit():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 18
    assert group["parallelism"] == 18
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "72000s"
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "BATCH_TASK_INDEX" in script
    assert "ucv_three_arm_15m_simplification.run worker" in script
    assert "SUCCESS.json" in script
    assert "$HOME" not in script


def test_cloud_smoke_and_remote_controller_are_mandatory():
    builder = _builder_module()
    smoke = builder.build_job(_builder_args(builder, "smoke"))["taskGroups"][0]
    smoke_script = smoke["taskSpec"]["runnables"][0]["script"]["text"]
    assert "ucv_three_arm_15m_simplification.run smoke" in smoke_script
    assert "--no-resume" in smoke_script

    controller_job = builder.build_job(_builder_args(builder, "controller"))
    controller = controller_job["taskGroups"][0]
    script = controller["taskSpec"]["runnables"][0]["script"]["text"]
    assert controller["taskSpec"]["maxRetryCount"] == 2
    assert "EXP22_REMOTE_CONTROLLER=1" in script
    assert 'run_ucv_three_arm_15m_simplification.sh "$CONTROLLER_ACTION"' in script


def test_local_run_returns_after_one_controller_submission(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "gcloud.log"
    fake = fake_bin / "gcloud"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$GCLOUD_LOG"\n')
    fake.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GCLOUD_LOG": str(log),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west1",
        "BUCKET": "gs://example/results",
        "SA_EMAIL": "batch@example.iam.gserviceaccount.com",
        "REPO_REF": "a" * 40,
        "RUN_ID": "exp22-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_ucv_three_arm_15m_simplification.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "batch jobs submit exp22-test-run-controller" in calls[0]
    assert "laptop may now be disconnected" in completed.stdout


def test_readme_places_experiment_22_after_experiment_21():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 21:") < readme.index("## Experiment 22:")
    section = readme.split("## Experiment 22:", 1)[1]
    assert "run_ucv_three_arm_15m_simplification.sh smoke-local" in section
    assert "run_ucv_three_arm_15m_simplification.sh run" in section
    assert "155--170 N2 VM-hours" in section
