from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.leduc_poker.four_algorithm_heldout_benchmark.config import HELDOUT_SEEDS
from experiments.leduc_poker.ucv_residual_target_factorial.analyse import (
    _stream_merged_csv,
)
from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    COMBINED,
    CONTROL,
    CRITIC_TARGET_AVERAGE_WINDOW,
    PRODUCTION_SEEDS,
    REGRET_RESIDUAL_BLOCKS,
    REGRET_RESIDUAL_WIDTH,
    RESIDUAL_ONLY,
    SMOKE_SEEDS,
    VARIANT_ORDER,
    checkpoint_schedule,
    task_schedule,
    training_state_checkpoint_ids,
    validate_contract,
    variant_config,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    build_training_state,
    restore_training_state,
)
from experiments.leduc_poker.ucv_residual_target_factorial.worker import (
    _make_solver,
    _smoke_overrides,
)
from unbiased_escher.factorial import (
    ResidualLayerNormRegretMLP,
    TemporallyAveragedCrossFittedQMember,
)
from vr_deep_cfr.logger import Logger


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "ucv_residual_target_factorial_batch.py"
    spec = importlib.util.spec_from_file_location("exp25_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str, parallelism: int = 12):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        bucket_root="gs://example/results",
        run_id="exp25-test",
        parallelism=parallelism,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_batch_clones_the_canonical_repository():
    builder = _builder_module()
    assert builder.REPO_URL == (
        "https://github.com/lawrencewlcknight/"
        "leduc-poker-escher-architecture-experiments.git"
    )


def test_frozen_factorial_seed_and_checkpoint_contract():
    assert VARIANT_ORDER == (
        CONTROL,
        RESIDUAL_ONLY,
        AVERAGED_TARGET_ONLY,
        COMBINED,
    )
    assert len(PRODUCTION_SEEDS) == 3
    assert len(set(PRODUCTION_SEEDS)) == 3
    assert not set(PRODUCTION_SEEDS).intersection(HELDOUT_SEEDS)
    schedule = checkpoint_schedule()
    assert len(schedule) == 19
    assert schedule[0]["checkpoint_id"] == "time_02h"
    assert schedule[17]["checkpoint_id"] == "time_36h"
    assert schedule[18]["checkpoint_id"] == "node_15m"
    assert training_state_checkpoint_ids() == ("time_24h", "time_36h")
    assert len(task_schedule()) == 12
    validate_contract(seeds=PRODUCTION_SEEDS, schedule=schedule, smoke=False)
    validate_contract(
        seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True
    )


def test_factorial_arms_change_only_the_intended_switches():
    control = variant_config(CONTROL)
    residual = variant_config(RESIDUAL_ONLY)
    averaged = variant_config(AVERAGED_TARGET_ONLY)
    combined = variant_config(COMBINED)
    assert control["regret_network_type"] == "mlp"
    assert control["critic_target_average_window"] == 1
    assert residual["regret_network_type"] == "residual_layer_norm"
    assert residual["critic_target_average_window"] == 1
    assert averaged["regret_network_type"] == "mlp"
    assert averaged["critic_target_average_window"] == CRITIC_TARGET_AVERAGE_WINDOW
    assert combined["regret_network_type"] == "residual_layer_norm"
    assert combined["critic_target_average_window"] == CRITIC_TARGET_AVERAGE_WINDOW
    for config in (control, residual, averaged, combined):
        assert config["fixed_control_variate_beta"] == 1.0
        assert config["q_ensemble_size"] == 2
        assert config["use_instantaneous_predictor"] is False
        assert config["use_residual_calibration"] is True


def test_residual_regret_model_shape_zero_head_and_gradients():
    model = ResidualLayerNormRegretMLP(
        30,
        3,
        width=REGRET_RESIDUAL_WIDTH,
        blocks=REGRET_RESIDUAL_BLOCKS,
    )
    inputs = torch.randn(7, 30)
    outputs = model(inputs)
    assert outputs.shape == (7, 3)
    assert torch.equal(outputs, torch.zeros_like(outputs))
    outputs.sum().backward()
    assert model.output_layer.weight.grad is not None
    assert len(model.blocks) == 4


def test_temporal_target_is_parameter_mean_of_completed_fits():
    member = TemporallyAveragedCrossFittedQMember(
        4,
        2,
        2,
        [3],
        1e-3,
        16,
        2,
        1,
        Logger(verbose=False),
        [],
        "cpu",
        gradient_clip_norm=10.0,
        target_average_window=4,
    )
    for value in (1.0, 2.0, 3.0, 4.0, 5.0):
        with torch.no_grad():
            for parameter in member.model.parameters():
                parameter.fill_(value)
        member._install_temporal_average()
    assert len(member.target_history) == 4
    for parameter in member.target_model.parameters():
        assert torch.allclose(parameter, torch.full_like(parameter, 3.5))


def test_large_diagnostic_csvs_are_merged_as_streams(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    output = tmp_path / "merged.csv"
    first.write_text("metric,value\nalpha,1\nbeta,2\n")
    second.write_text("metric,value\ngamma,3\n")

    count = _stream_merged_csv(
        output,
        (
            (first, CONTROL, 11),
            (second, RESIDUAL_ONLY, 13),
        ),
    )

    assert count == 3
    assert output.read_text().splitlines() == [
        "metric,seed,value,variant_id",
        f"alpha,11,1,{CONTROL}",
        f"beta,11,2,{CONTROL}",
        f"gamma,13,3,{RESIDUAL_ONLY}",
    ]


def test_full_state_resume_reproduces_next_iteration():
    config = variant_config(COMBINED)
    _smoke_overrides(config)
    first = _make_solver(7, config)
    first._solve_start_time = 0.0
    first._post_checkpoint_callback = None
    first.iteration()
    first._checkpoint_resume_rng_state = first._capture_rng_state()
    payload = build_training_state(
        first,
        variant_id=COMBINED,
        seed=7,
        checkpoint_id="test",
        active_seconds=0.0,
        repository_commit="test-commit",
        config=config,
        captured_snapshots=[],
    )
    first.iteration()

    resumed = _make_solver(7, config)
    restore_training_state(
        resumed,
        payload,
        variant_id=COMBINED,
        seed=7,
        repository_commit="test-commit",
        config=config,
    )
    resumed._solve_start_time = 0.0
    resumed._post_checkpoint_callback = None
    resumed.iteration()
    assert resumed.num_iteration == first.num_iteration
    assert resumed.episode == first.episode
    assert resumed.nodes_touched == first.nodes_touched
    for expected, observed in zip(first.regret_trainers, resumed.regret_trainers):
        for name, tensor in expected.model.state_dict().items():
            assert torch.equal(tensor, observed.model.state_dict()[name])
    for expected, observed in zip(
        first.q_value_trainer.members, resumed.q_value_trainer.members
    ):
        for name, tensor in expected.target_model.state_dict().items():
            assert torch.equal(tensor, observed.target_model.state_dict()[name])


def test_batch_has_12_on_demand_workers_and_hard_ceiling():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    assert group["taskCount"] == 12
    assert group["parallelism"] == 12
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "194400s"
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert "ucv_residual_target_factorial.run worker" in script
    assert "EXP25_REMOTE_TASK_URI" in script
    assert "gcloud storage rsync" in script
    assert "$HOME" not in script


def test_remote_controller_runs_smoke_before_production():
    builder = _builder_module()
    smoke = builder.build_job(_builder_args(builder, "smoke"))["taskGroups"][0]
    assert "ucv_residual_target_factorial.run smoke" in smoke["taskSpec"][
        "runnables"
    ][0]["script"]["text"]
    controller = builder.build_job(_builder_args(builder, "controller"))
    script = controller["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "EXP25_REMOTE_CONTROLLER=1" in script
    assert 'run_ucv_residual_target_factorial.sh "$CONTROLLER_ACTION"' in script


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
        "RUN_ID": "exp25-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_ucv_residual_target_factorial.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "batch jobs submit exp25-test-run-controller" in calls[0]
    assert "laptop may now be disconnected" in completed.stdout


def test_readme_places_experiment_25_after_experiment_24():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 24:") < readme.index("## Experiment 25:")
    section = readme.split("## Experiment 25:", 1)[1]
    assert "run_ucv_residual_target_factorial.sh smoke-local" in section
    assert "run_ucv_residual_target_factorial.sh run" in section
    assert "450--480 N2 VM-hours" in section
