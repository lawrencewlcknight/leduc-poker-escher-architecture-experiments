from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.leduc_poker.information_set_stratified_distillation.config import (
    ARM_ORDER,
    FIT_REPLICATES,
    PRODUCTION_SEEDS,
    SOURCE_CHECKPOINTS,
    SOURCE_VARIANT_ID,
    validate_contract,
)
from experiments.leduc_poker.information_set_stratified_distillation.distill import (
    draw_information_set_batch,
    information_set_sampling_plan,
)
from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    PRODUCTION_SEEDS as EXPERIMENT_25_SEEDS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "information_set_stratified_distillation_batch.py"
    spec = importlib.util.spec_from_file_location("exp28_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _args(builder, kind: str):
    return SimpleNamespace(
        kind=kind,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        bucket_root="gs://example/results",
        run_id="exp28-test",
        experiment_25_run_id="exp25-source",
        parallelism=3,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_reuses_three_experiment_25_source_trajectories():
    assert PRODUCTION_SEEDS == EXPERIMENT_25_SEEDS
    assert SOURCE_VARIANT_ID == AVERAGED_TARGET_ONLY
    assert SOURCE_CHECKPOINTS == ("time_24h", "time_36h")
    assert len(ARM_ORDER) == 3
    assert FIT_REPLICATES == 3
    validate_contract(seeds=PRODUCTION_SEEDS, smoke=False)


def test_sampling_plan_has_exact_importance_correction():
    source = SimpleNamespace(
        infostates=np.asarray(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=np.float32,
        )
    )
    empirical = information_set_sampling_plan(source, 1.0)
    np.testing.assert_allclose(
        empirical["proposal_probabilities"],
        empirical["empirical_probabilities"],
    )
    np.testing.assert_allclose(empirical["importance_ratios"], 1.0)

    uniform = information_set_sampling_plan(source, 0.0)
    np.testing.assert_allclose(uniform["proposal_probabilities"], [0.5, 0.5])
    np.testing.assert_allclose(uniform["importance_ratios"], [1.5, 0.5])
    assert np.isclose(
        np.dot(
            uniform["proposal_probabilities"],
            uniform["importance_ratios"],
        ),
        1.0,
    )


def test_information_set_batch_rows_and_weights_follow_plan():
    source = SimpleNamespace(
        infostates=np.asarray(
            [[0.0], [0.0], [0.0], [1.0]],
            dtype=np.float32,
        )
    )
    plan = information_set_sampling_plan(source, 0.0)
    rows, ratios = draw_information_set_batch(
        np.random.default_rng(28),
        plan,
        10_000,
    )
    assert set(rows.tolist()) == {0, 1, 2, 3}
    assert set(ratios.tolist()) == {0.5, 1.5}
    rare_fraction = float(np.mean(rows == 3))
    assert 0.47 < rare_fraction < 0.53


def test_cloud_job_downloads_only_two_source_states_per_worker():
    builder = _builder_module()
    job = builder.build_job(_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == 3
    assert group["parallelism"] == 3
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == "43200s"
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    assert "for CHECKPOINT in time_24h time_36h" in script
    assert "ucv_residual_target_factorial.config" in script
    assert "information_set_stratified_distillation.run worker" in script
    assert "$HOME" not in script


def test_remote_controller_runs_smoke_before_workers():
    builder = _builder_module()
    smoke = builder.build_job(_args(builder, "smoke"))
    smoke_script = smoke["taskGroups"][0]["taskSpec"]["runnables"][0]["script"][
        "text"
    ]
    assert "information_set_stratified_distillation.run smoke" in smoke_script
    controller = builder.build_job(_args(builder, "controller"))
    controller_script = controller["taskGroups"][0]["taskSpec"]["runnables"][0][
        "script"
    ]["text"]
    assert "EXP28_REMOTE_CONTROLLER=1" in controller_script
    assert (
        'run_information_set_stratified_distillation.sh "$CONTROLLER_ACTION"'
        in controller_script
    )


def test_root_readme_places_experiment_28_after_experiment_27():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 27:") < readme.index("## Experiment 28:")
    section = readme.split("## Experiment 28:", 1)[1]
    assert "run_information_set_stratified_distillation.sh smoke-local" in section
    assert "run_information_set_stratified_distillation.sh run" in section
