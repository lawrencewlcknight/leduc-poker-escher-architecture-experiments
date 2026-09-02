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
    PRODUCTION_SEEDS as EXPERIMENT_22_SEEDS,
)
from experiments.leduc_poker.ucv_24h_stability_development.config import (
    CHECKPOINT_TARGET_HOURS,
    FAST_CORE,
    FULL_ADAPTIVE,
    NONPREDICTIVE_CORE,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    STABLE_NONPREDICTIVE_CORE,
    VARIANT_ORDER,
    checkpoint_schedule,
    task_schedule,
    validate_contract,
    variant_config,
)
from unbiased_escher.stability import StableUnbiasedControlVariateEscher


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "ucv_24h_stability_batch.py"
    spec = importlib.util.spec_from_file_location("exp23_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str, parallelism: int = 16):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        bucket_root="gs://example/results",
        run_id="exp23-test",
        parallelism=parallelism,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_frozen_four_arm_four_seed_and_checkpoint_contract():
    assert VARIANT_ORDER == (
        FULL_ADAPTIVE,
        FAST_CORE,
        NONPREDICTIVE_CORE,
        STABLE_NONPREDICTIVE_CORE,
    )
    assert PRODUCTION_SEEDS == (254158, 577260, 961890, 848645)
    assert not set(PRODUCTION_SEEDS).intersection(HELDOUT_SEEDS)
    assert not set(PRODUCTION_SEEDS).intersection(EXPERIMENT_22_SEEDS)
    assert CHECKPOINT_TARGET_HOURS == tuple(range(2, 25, 2))
    schedule = checkpoint_schedule()
    assert len(schedule) == 13
    assert schedule[0]["checkpoint_id"] == "time_02h"
    assert schedule[11]["checkpoint_id"] == "time_24h"
    assert schedule[12]["checkpoint_id"] == "node_15m"
    assert len(task_schedule()) == 16
    validate_contract(seeds=PRODUCTION_SEEDS, schedule=schedule, smoke=False)
    validate_contract(
        seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True
    )


def test_variant_changes_are_cumulative_and_calibration_is_retained():
    original = variant_config(FULL_ADAPTIVE)
    fast = variant_config(FAST_CORE)
    nonpredictive = variant_config(NONPREDICTIVE_CORE)
    stable = variant_config(STABLE_NONPREDICTIVE_CORE)
    assert original["fixed_control_variate_beta"] is None
    assert original["q_ensemble_size"] == 3
    assert fast["fixed_control_variate_beta"] == 1.0
    assert fast["q_ensemble_size"] == 2
    assert fast["use_residual_calibration"] is True
    assert nonpredictive["use_instantaneous_predictor"] is False
    assert nonpredictive["use_residual_calibration"] is True
    assert stable["anneal_start_nodes"] == 15_000_000
    assert stable["anneal_end_nodes"] == 45_000_000
    assert stable["anneal_final_learning_rate"] == 1e-4
    assert stable["regret_policy_gradient_clip_norm"] == 5.0


def test_cosine_schedule_has_frozen_endpoints():
    solver = object.__new__(StableUnbiasedControlVariateEscher)
    solver.nodes_touched = 0
    solver.initial_regret_policy_learning_rate = 1e-3
    solver.anneal_start_nodes = 15_000_000
    solver.anneal_end_nodes = 45_000_000
    solver.anneal_final_learning_rate = 1e-4
    assert solver.scheduled_learning_rate(0) == pytest.approx(1e-3)
    assert solver.scheduled_learning_rate(15_000_000) == pytest.approx(1e-3)
    assert solver.scheduled_learning_rate(30_000_000) == pytest.approx(5.5e-4)
    assert solver.scheduled_learning_rate(45_000_000) == pytest.approx(1e-4)
    assert solver.scheduled_learning_rate(60_000_000) == pytest.approx(1e-4)


def test_batch_has_16_standard_workers_and_hard_cost_ceiling():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 16
    assert group["parallelism"] == 16
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "129600s"
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "ucv_24h_stability_development.run worker" in script
    assert "BATCH_TASK_INDEX" in script
    assert "SUCCESS.json" in script
    assert "$HOME" not in script


def test_remote_controller_requires_cloud_smoke_before_production():
    builder = _builder_module()
    smoke = builder.build_job(_builder_args(builder, "smoke"))["taskGroups"][0]
    assert "ucv_24h_stability_development.run smoke" in smoke["taskSpec"][
        "runnables"
    ][0]["script"]["text"]
    controller = builder.build_job(_builder_args(builder, "controller"))
    group = controller["taskGroups"][0]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskSpec"]["maxRetryCount"] == 2
    assert "EXP23_REMOTE_CONTROLLER=1" in script
    assert 'run_ucv_24h_stability_development.sh "$CONTROLLER_ACTION"' in script


def test_local_run_returns_after_controller_submission(tmp_path):
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
        "RUN_ID": "exp23-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_ucv_24h_stability_development.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "batch jobs submit exp23-test-run-controller" in calls[0]
    assert "laptop may now be disconnected" in completed.stdout


def test_readme_places_experiment_23_after_experiment_22():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 22:") < readme.index("## Experiment 23:")
    section = readme.split("## Experiment 23:", 1)[1]
    assert "run_ucv_24h_stability_development.sh smoke-local" in section
    assert "run_ucv_24h_stability_development.sh run" in section
    assert "400--440 N2 VM-hours" in section
