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
from experiments.leduc_poker.ucv_advantage_replay_36h.config import (
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    CHECKPOINT_TARGET_HOURS,
    MAX_ITERATIONS,
    PRODUCTION_SEEDS,
    REFERENCE_RUN_ID,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    validate_contract,
)
from unbiased_escher.advantage_replay import PersistentAdvantageReplayTrainer
from vr_deep_cfr.logger import Logger


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "ucv_advantage_replay_36h_batch.py"
    spec = importlib.util.spec_from_file_location("exp26_batch", path)
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
        experiment_21_run_id=REFERENCE_RUN_ID,
        bucket_root="gs://example/results",
        run_id="exp26-test",
        parallelism=parallelism,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_batch_clones_the_canonical_repositories():
    builder = _builder_module()
    assert builder.REPO_URL == (
        "https://github.com/lawrencewlcknight/"
        "leduc-poker-escher-architecture-experiments.git"
    )
    assert builder.DEEP_REPO_URL == (
        "https://github.com/lawrencewlcknight/"
        "leduc-poker-deep-cfr-experiments.git"
    )


def test_frozen_candidate_seed_and_checkpoint_contract():
    assert PRODUCTION_SEEDS == EXPERIMENT_21_SEEDS
    assert PRODUCTION_SEEDS == (104729, 130363, 155921, 181081, 205759)
    assert CHECKPOINT_TARGET_HOURS == tuple(range(2, 37, 2))
    assert MAX_ITERATIONS == 800
    assert task_schedule() == tuple((CANDIDATE_ID, seed) for seed in PRODUCTION_SEEDS)
    schedule = checkpoint_schedule()
    assert len(schedule) == 19
    assert schedule[0]["checkpoint_id"] == "time_02h"
    assert schedule[17]["checkpoint_id"] == "time_36h"
    assert schedule[18] == {
        "checkpoint_id": "node_15m",
        "checkpoint_type": "nodes",
        "target_active_seconds": None,
        "target_active_hours": None,
        "target_nodes": 15_000_000,
    }
    validate_contract(seeds=PRODUCTION_SEEDS, schedule=schedule, smoke=False)
    validate_contract(seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True)
    with pytest.raises(ValueError, match="Production seeds"):
        validate_contract(seeds=tuple(reversed(PRODUCTION_SEEDS)), schedule=schedule, smoke=False)


def test_replay_architecture_starts_from_promoted_nonpredictive_core():
    assert CANDIDATE_CONFIG["fixed_control_variate_beta"] == 1.0
    assert CANDIDATE_CONFIG["q_ensemble_size"] == 2
    assert CANDIDATE_CONFIG["use_instantaneous_predictor"] is False
    assert CANDIDATE_CONFIG["force_prediction_gate_zero"] is True
    assert CANDIDATE_CONFIG["use_residual_calibration"] is True
    assert CANDIDATE_CONFIG["regret_policy_gradient_clip_norm"] is None
    assert CANDIDATE_CONFIG["anneal_start_nodes"] is None
    assert CANDIDATE_CONFIG["anneal_end_nodes"] is None
    assert CANDIDATE_CONFIG["anneal_final_learning_rate"] is None
    assert CANDIDATE_CONFIG["advantage_replay_weight_exponent"] == 1.0


def _replay_trainer() -> PersistentAdvantageReplayTrainer:
    return PersistentAdvantageReplayTrainer(
        2,
        2,
        [4],
        1e-3,
        8,
        2,
        1,
        Logger(verbose=False),
        False,
        "cpu",
        2.3,
        replay_weight_exponent=1.0,
    )


def test_advantage_memory_persists_across_outer_iteration_resets():
    trainer = _replay_trainer()
    trainer.add_data([1.0, 0.0], [0.25, -0.25], [1.0, 1.0], 1)
    trainer.reset_buffer()
    trainer.add_data([0.0, 1.0], [-0.5, 0.5], [1.0, 1.0], 2)
    assert trainer.buffer.cur_id == 2
    assert trainer.replay_is_persistent is True


def test_direct_replay_loss_does_not_read_previous_fitted_target():
    trainer = _replay_trainer()
    samples = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.25, -0.25], [-0.5, 0.5]]),
        torch.ones((2, 2)),
        torch.tensor([[1.0], [2.0]]),
    )
    before = trainer.compute_loss(samples, 2).detach().clone()
    with torch.no_grad():
        for parameter in trainer.target_model.parameters():
            parameter.fill_(10_000.0)
    after = trainer.compute_loss(samples, 2).detach().clone()
    assert torch.equal(before, after)
    assert trainer.uses_recursive_target is False


def test_batch_has_five_standard_workers_and_hard_ceiling():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 5
    assert group["parallelism"] == 5
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "180000s"
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "ucv_advantage_replay_36h.run worker" in script
    assert "BATCH_TASK_INDEX" in script
    assert "SUCCESS.json" in script
    assert "$HOME" not in script


def test_smoke_and_aggregate_exercise_reference_contract():
    builder = _builder_module()
    smoke = builder.build_job(_builder_args(builder, "smoke"))["taskGroups"][0]
    smoke_script = smoke["taskSpec"]["runnables"][0]["script"]["text"]
    assert "ucv_advantage_replay_36h.run smoke" in smoke_script
    assert 'deep-cfr-repo "$DEEP_REPO"' in smoke_script

    aggregate = builder.build_job(_builder_args(builder, "aggregate"))["taskGroups"][0]
    script = aggregate["taskSpec"]["runnables"][0]["script"]["text"]
    assert '"$BUCKET_ROOT/$EXP21_RUN_ID"' in script
    assert 'experiment-21-root "$OUTPUT_ROOT/experiment_21_reference"' in script
    assert "ucv_advantage_replay_36h.run aggregate" in script


def test_remote_controller_runs_entire_workflow():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "controller"))
    group = job["taskGroups"][0]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskSpec"]["maxRetryCount"] == 2
    assert "EXP26_REMOTE_CONTROLLER=1" in script
    assert 'run_ucv_advantage_replay_36h.sh "$CONTROLLER_ACTION"' in script


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
        "RUN_ID": "exp26-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_ucv_advantage_replay_36h.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "batch jobs submit exp26-test-run-controller" in calls[0]
    assert "laptop may now be disconnected" in completed.stdout


def test_readme_places_experiment_26_after_experiment_25():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 25:") < readme.index("## Experiment 26:")
    section = readme.split("## Experiment 26:", 1)[1]
    assert "run_ucv_advantage_replay_36h.sh smoke-local" in section
    assert "run_ucv_advantage_replay_36h.sh run" in section
    assert "190--200 N2 VM-hours" in section
    assert "development" in section
