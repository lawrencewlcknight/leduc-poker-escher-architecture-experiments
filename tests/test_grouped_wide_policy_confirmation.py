from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.leduc_poker.grouped_wide_policy_confirmation.config import (
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    POLICY_LEARNING_RATE,
    POLICY_NETWORK_LAYERS,
    POLICY_TRAINING_STEPS,
    PREVIOUSLY_USED_SEEDS,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    training_state_checkpoint_ids,
    validate_contract,
)
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import (
    _make_solver as make_confirmation_solver,
    _smoke_overrides as confirmation_smoke_overrides,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.worker import (
    _make_solver as make_experiment_29_solver,
    _smoke_overrides as experiment_29_smoke_overrides,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_CONFIG as EXPERIMENT_29_CONFIG,
)
from unbiased_escher.policy_distillation import (
    GROUPED_SOFT_TARGET_CROSS_ENTROPY,
    GroupedSoftTargetCrossEntropyAvePolicyTrainer,
)
from vr_deep_cfr.logger import Logger


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "grouped_wide_policy_confirmation_batch.py"
    spec = importlib.util.spec_from_file_location("exp35_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        experiment_29_run_id="exp29-source",
        bucket_root="gs://example/results",
        run_id="exp35-test",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_uses_five_fresh_seeds_and_36_hour_schedule():
    assert PRODUCTION_SEEDS == (470892, 385626, 145871, 902492, 318362)
    assert len(set(PRODUCTION_SEEDS)) == 5
    assert not set(PRODUCTION_SEEDS).intersection(PREVIOUSLY_USED_SEEDS)
    assert task_schedule() == tuple((CANDIDATE_ID, seed) for seed in PRODUCTION_SEEDS)
    schedule = checkpoint_schedule()
    assert schedule[0]["checkpoint_id"] == "time_02h"
    assert schedule[17]["checkpoint_id"] == "time_36h"
    assert schedule[18]["checkpoint_id"] == "node_15m"
    assert training_state_checkpoint_ids() == ("time_24h", "time_36h")
    validate_contract(seeds=PRODUCTION_SEEDS, schedule=schedule, smoke=False)
    validate_contract(
        seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True
    )


def test_candidate_freezes_experiment_34_selected_policy_fit():
    assert CANDIDATE_CONFIG["average_policy_loss"] == GROUPED_SOFT_TARGET_CROSS_ENTROPY
    assert CANDIDATE_CONFIG["average_policy_network_layers"] == POLICY_NETWORK_LAYERS
    assert CANDIDATE_CONFIG["average_policy_learning_rate"] == POLICY_LEARNING_RATE
    assert CANDIDATE_CONFIG["average_policy_train_steps"] == POLICY_TRAINING_STEPS
    assert CANDIDATE_CONFIG["fixed_control_variate_beta"] == 1.0
    assert CANDIDATE_CONFIG["q_ensemble_size"] == 2
    assert CANDIDATE_CONFIG["critic_target_average_window"] == 4


def test_grouped_trainer_preserves_row_wise_objective():
    trainer = GroupedSoftTargetCrossEntropyAvePolicyTrainer(
        2, 3, [4], 1e-3, 8, 8, 1, Logger(verbose=False), "cpu", 2.0
    )
    rows = [
        ([1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], 2.0),
        ([0.0, 1.0], [0.0, 0.25, 0.75], [0.0, 1.0, 1.0], 2.0),
    ]
    for feature, target, mask, iteration in rows:
        trainer.add_data(feature, target, mask, iteration)
    features, targets, masks, weights = trainer._grouped_training_data(2)
    logits_by_feature = {
        (1.0, 0.0): np.asarray([0.7, 0.3, 0.0]),
        (0.0, 1.0): np.asarray([0.0, 0.4, 0.6]),
    }
    raw = []
    for feature, target, _, iteration in rows:
        probability = logits_by_feature[tuple(feature)]
        raw.append(-np.dot(target, np.log(np.clip(probability, 1e-12, 1.0))) * (iteration / 2.0 * 2.0) ** 2)
    grouped = []
    for feature, target, weight in zip(features.numpy(), targets.numpy(), weights.numpy()):
        probability = logits_by_feature[tuple(float(item) for item in feature)]
        grouped.append(-np.dot(target, np.log(np.clip(probability, 1e-12, 1.0))) * weight)
    assert np.isclose(np.mean(raw), np.mean(grouped), rtol=1e-6, atol=1e-7)
    assert trainer.grouped_num_rows == 3
    assert trainer.grouped_num_information_sets == 2


def test_wide_policy_initialisation_does_not_shift_ucv_core_rng():
    confirmation = deepcopy(CANDIDATE_CONFIG)
    baseline = deepcopy(EXPERIMENT_29_CONFIG)
    confirmation_smoke_overrides(confirmation)
    experiment_29_smoke_overrides(baseline)
    left = make_confirmation_solver(731, confirmation)
    right = make_experiment_29_solver(731, baseline)
    assert left.ave_policy_trainer.network_layers == list(POLICY_NETWORK_LAYERS)
    assert right.ave_policy_trainer.network_layers == [64, 64, 64]
    for left_trainer, right_trainer in zip(left.regret_trainers, right.regret_trainers):
        for key, value in left_trainer.model.state_dict().items():
            assert torch.equal(value, right_trainer.model.state_dict()[key])


def test_cloud_contract_has_five_standard_workers_and_bounded_runtime():
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
    assert "grouped_wide_policy_confirmation.run worker" in script
    assert "EXP35_REMOTE_TASK_URI" in script
    assert "Invalid Experiment 35 task metadata" in script
    assert "$HOME" not in script


def test_aggregate_avoids_large_continuation_and_diagnostic_files():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "aggregate"))
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--exclude=" in script
    assert "training_states|diagnostics" in script
    assert '"$BUCKET_ROOT/$EXP29_RUN_ID/analysis"' in script
    assert "--experiment-29-root" in script
    assert "grouped_wide_policy_confirmation.run aggregate" in script


def test_readme_places_experiment_35_after_experiment_34():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 34:") < readme.index("## Experiment 35:")
    section = readme.split("## Experiment 35:", 1)[1]
    assert "run_grouped_wide_policy_confirmation.sh smoke-local" in section
    assert "run_grouped_wide_policy_confirmation.sh run" in section
