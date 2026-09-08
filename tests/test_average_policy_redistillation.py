from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from experiments.leduc_poker.average_policy_redistillation.config import (
    ARM_ORDER,
    FACTORIAL_ARMS,
    FIT_REPLICATES,
    PRODUCTION_SEEDS,
    SOURCE_CHECKPOINTS,
    SOURCE_VARIANT_ID,
    validate_contract,
)
from experiments.leduc_poker.ucv_residual_target_factorial.config import (
    AVERAGED_TARGET_ONLY,
    PRODUCTION_SEEDS as EXPERIMENT_25_SEEDS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "average_policy_redistillation_batch.py"
    spec = importlib.util.spec_from_file_location("exp27_batch", path)
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
        run_id="exp27-test",
        experiment_25_run_id="exp25-source",
        parallelism=3,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_reuses_only_the_three_experiment_25_source_seeds():
    assert PRODUCTION_SEEDS == EXPERIMENT_25_SEEDS
    assert SOURCE_VARIANT_ID == AVERAGED_TARGET_ONLY
    assert SOURCE_CHECKPOINTS == ("time_24h", "time_36h")
    assert len(FACTORIAL_ARMS) == 4
    assert len(ARM_ORDER) == 5
    assert FIT_REPLICATES == 3
    validate_contract(seeds=PRODUCTION_SEEDS, smoke=False)


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
    assert "average_policy_redistillation.run worker" in script
    assert "$HOME" not in script


def test_remote_controller_runs_smoke_before_workers():
    builder = _builder_module()
    smoke = builder.build_job(_args(builder, "smoke"))
    smoke_script = smoke["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "average_policy_redistillation.run smoke" in smoke_script
    controller = builder.build_job(_args(builder, "controller"))
    controller_script = controller["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "EXP27_REMOTE_CONTROLLER=1" in controller_script
    assert 'run_average_policy_redistillation.sh "$CONTROLLER_ACTION"' in controller_script


def test_root_readme_places_experiment_27_after_experiment_26():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 26:") < readme.index("## Experiment 27:")
    section = readme.split("## Experiment 27:", 1)[1]
    assert "run_average_policy_redistillation.sh smoke-local" in section
    assert "run_average_policy_redistillation.sh run" in section
