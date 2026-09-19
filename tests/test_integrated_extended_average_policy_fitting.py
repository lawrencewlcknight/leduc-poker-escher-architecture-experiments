from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.leduc_poker.integrated_extended_average_policy_fitting.config import (
    CANDIDATE_CONFIG,
    CANDIDATE_ID,
    FINAL_AUDIT_UPDATES,
    PRODUCTION_SEEDS,
    REFINEMENT_UPDATES,
    checkpoint_schedule,
    contract_manifest,
    final_active_checkpoint_id,
    refinement_schedule,
    task_schedule,
    validate_contract,
)
from experiments.leduc_poker.integrated_extended_average_policy_fitting.worker import (
    _buffer_source,
    _make_solver,
    _restore_rng,
    _rng_state,
    _smoke_overrides,
)
from experiments.leduc_poker.average_policy_optimization_horizon.fitting import (
    grouped_replay_dataset,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "integrated_extended_average_policy_fitting_batch.py"
    spec = importlib.util.spec_from_file_location("exp43_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        experiment_29_run_id="exp29-source",
        bucket_root="gs://example/results",
        run_id="exp43-test",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_integrates_selected_horizon_without_replacing_ordinary_fit():
    assert PRODUCTION_SEEDS == (104729, 130363, 155921, 181081, 205759)
    assert task_schedule() == tuple((CANDIDATE_ID, seed) for seed in PRODUCTION_SEEDS)
    assert CANDIDATE_CONFIG["ave_policy_network_train_steps"] == 5_000
    assert CANDIDATE_CONFIG["integrated_refinement_updates"] == 1_700
    assert REFINEMENT_UPDATES == 1_700
    assert FINAL_AUDIT_UPDATES == (400, 800, 1200, 1600, 1700)
    assert refinement_schedule(final_checkpoint=False) == (0, 1700)
    assert refinement_schedule(final_checkpoint=True) == (0, 400, 800, 1200, 1600, 1700)
    assert final_active_checkpoint_id() == "time_36h"
    validate_contract(
        seeds=PRODUCTION_SEEDS, schedule=checkpoint_schedule(), smoke=False
    )
    manifest = contract_manifest()
    assert manifest["refinement"]["training_requires_game_tree"] is False
    assert manifest["evidence_status"] == "post-selection paired development evidence"


def test_smoke_solver_builds_real_grouped_dataset_and_rng_can_be_restored():
    config = deepcopy(CANDIDATE_CONFIG)
    _smoke_overrides(config)
    solver = _make_solver(0, config)
    state = solver.game.new_initial_state()
    while state.is_chance_node():
        state = state.child(state.chance_outcomes()[0][0])
    policy = np.zeros(solver.action_size, dtype=np.float32)
    legal = state.legal_actions()
    policy[legal] = 1.0 / len(legal)
    solver.ave_policy_trainer.add_data(
        state.information_state_tensor(), policy,
        state.legal_actions_mask(), 1,
    )
    dataset = grouped_replay_dataset(_buffer_source(solver))
    assert dataset.replay_rows == 1
    assert len(dataset.features) == 1

    saved = _rng_state()
    expected = torch.rand(4)
    _restore_rng(saved)
    observed = torch.rand(4)
    assert torch.equal(expected, observed)


def test_cloud_job_uses_five_n2_standard_8_workers_and_bounded_runtime():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == 5
    assert group["parallelism"] == 5
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "194400s"
    assert group["taskSpec"]["maxRetryCount"] == 1
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    assert "EXP43_REMOTE_TASK_URI" in script
    assert "Invalid Experiment 43 task metadata" in script
    assert "$HOME" not in script


def test_aggregate_excludes_large_continuation_and_diagnostic_files():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "aggregate"))
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "training_states|diagnostics" in script
    assert '"$BUCKET_ROOT/$EXP29_RUN_ID/analysis"' in script
    assert "integrated_extended_average_policy_fitting.run aggregate" in script


def test_root_readme_places_experiment_43_after_42():
    text = (REPOSITORY / "README.md").read_text()
    assert text.index("## Experiment 42:") < text.index("## Experiment 43:")
    section = text.split("## Experiment 43:", 1)[1]
    assert "run_integrated_extended_average_policy_fitting.sh smoke-local" in section
    assert "run_integrated_extended_average_policy_fitting.sh run" in section
