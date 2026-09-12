from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.leduc_poker.deep_cfr_ucv_36h_plateau.config import (
    PRODUCTION_SEEDS as EXPERIMENT_21_SEEDS,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    PRODUCTION_SEEDS,
    REFERENCE_EXPERIMENT_21_RUN_ID,
    REFERENCE_EXPERIMENT_24_RUN_ID,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    training_state_checkpoint_ids,
    validate_contract,
)
from unbiased_escher.policy_distillation import (
    SOFT_TARGET_CROSS_ENTROPY,
    SoftTargetCrossEntropyAvePolicyTrainer,
)
from vr_deep_cfr.logger import Logger


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "promoted_ucv_cross_entropy_36h_batch.py"
    spec = importlib.util.spec_from_file_location("exp29_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str, parallelism: int = 5):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        deep_repo_url=builder.DEEP_REPO_URL,
        repo_ref="a" * 40,
        deep_repo_ref="b" * 40,
        experiment_21_run_id=REFERENCE_EXPERIMENT_21_RUN_ID,
        experiment_24_run_id=REFERENCE_EXPERIMENT_24_RUN_ID,
        bucket_root="gs://example/results",
        run_id="exp29-test",
        parallelism=parallelism,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_frozen_candidate_seed_and_checkpoint_contract():
    assert PRODUCTION_SEEDS == EXPERIMENT_21_SEEDS
    assert PRODUCTION_SEEDS == (104729, 130363, 155921, 181081, 205759)
    assert task_schedule() == tuple((CANDIDATE_ID, seed) for seed in PRODUCTION_SEEDS)
    schedule = checkpoint_schedule()
    assert len(schedule) == 19
    assert schedule[0]["checkpoint_id"] == "time_02h"
    assert schedule[17]["checkpoint_id"] == "time_36h"
    assert schedule[18]["checkpoint_id"] == "node_15m"
    assert training_state_checkpoint_ids() == ("time_24h", "time_36h")
    validate_contract(seeds=PRODUCTION_SEEDS, schedule=schedule, smoke=False)
    validate_contract(seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True)
    with pytest.raises(ValueError, match="Production seeds"):
        validate_contract(seeds=tuple(reversed(PRODUCTION_SEEDS)), schedule=schedule, smoke=False)


def test_promoted_architecture_contains_only_supported_development_choices():
    assert CANDIDATE_CONFIG["fixed_control_variate_beta"] == 1.0
    assert CANDIDATE_CONFIG["q_ensemble_size"] == 2
    assert CANDIDATE_CONFIG["use_instantaneous_predictor"] is False
    assert CANDIDATE_CONFIG["use_residual_calibration"] is True
    assert CANDIDATE_CONFIG["regret_network_type"] == "mlp"
    assert CANDIDATE_CONFIG["critic_target_average_window"] == 4
    assert CANDIDATE_CONFIG["average_policy_loss"] == SOFT_TARGET_CROSS_ENTROPY
    assert CANDIDATE_CONFIG["average_policy_reset_each_fit"] is True


def test_soft_target_cross_entropy_matches_manual_weighted_objective():
    trainer = SoftTargetCrossEntropyAvePolicyTrainer(
        2, 3, [4], 1e-3, 16, 2, 1, Logger(verbose=False), "cpu", 2.0
    )
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
    samples = (
        torch.zeros((2, 2)),
        torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.4, 0.6]]),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        torch.tensor([[1.0], [2.0]]),
    )
    observed = trainer.compute_loss(samples, 2)
    expected = torch.mean(
        torch.tensor([(1.0 / 2.0 * 2.0) ** 2, (2.0 / 2.0 * 2.0) ** 2])
        * torch.log(torch.tensor(2.0))
    )
    assert torch.allclose(observed, expected)


def test_batch_has_five_on_demand_workers_and_hard_ceiling():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 5
    assert group["parallelism"] == 5
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "194400s"
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "promoted_ucv_cross_entropy_36h.run worker" in script
    assert "EXP29_REMOTE_TASK_URI" in script
    assert "training_states" in script
    assert "$HOME" not in script


def test_smoke_aggregate_and_controller_include_both_references():
    builder = _builder_module()
    smoke = builder.build_job(_builder_args(builder, "smoke"))["taskGroups"][0]
    assert "promoted_ucv_cross_entropy_36h.run smoke" in smoke["taskSpec"]["runnables"][0]["script"]["text"]

    aggregate = builder.build_job(_builder_args(builder, "aggregate"))["taskGroups"][0]
    script = aggregate["taskSpec"]["runnables"][0]["script"]["text"]
    assert '"$BUCKET_ROOT/$EXP21_RUN_ID"' in script
    assert '"$BUCKET_ROOT/$EXP24_RUN_ID"' in script
    assert "--experiment-21-root" in script
    assert "--experiment-24-root" in script

    controller = builder.build_job(_builder_args(builder, "controller"))["taskGroups"][0]
    controller_script = controller["taskSpec"]["runnables"][0]["script"]["text"]
    assert "EXP29_REMOTE_CONTROLLER=1" in controller_script
    assert 'run_promoted_ucv_cross_entropy_36h.sh "$CONTROLLER_ACTION"' in controller_script


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
        "DEEP_CFR_REPO_REF": "b" * 40,
        "RUN_ID": "exp29-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_promoted_ucv_cross_entropy_36h.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "batch jobs submit exp29-test-run-controller" in calls[0]
    assert "laptop may now be disconnected" in completed.stdout


def test_readme_places_experiment_29_after_experiment_28():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 28:") < readme.index("## Experiment 29:")
    section = readme.split("## Experiment 29:", 1)[1]
    assert "run_promoted_ucv_cross_entropy_36h.sh smoke-local" in section
    assert "run_promoted_ucv_cross_entropy_36h.sh run" in section
    assert "post-selection" in section
