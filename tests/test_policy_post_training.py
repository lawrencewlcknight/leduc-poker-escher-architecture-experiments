from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel

from experiments.leduc_poker.average_policy_redistillation.distill import SourceData
from experiments.leduc_poker.policy_post_training.config import (
    METHODS,
    PRODUCTION_SEEDS,
    evaluation_steps,
    method_config,
)
from experiments.leduc_poker.policy_post_training.core import (
    information_sets,
    make_model,
    model_table,
)
from experiments.leduc_poker.policy_post_training.methods import METHOD_RUNNERS
from vr_deep_cfr.solver import MLP


REPOSITORY = Path(__file__).resolve().parents[1]


def _source():
    game = pyspiel.load_game("leduc_poker")
    rows = information_sets(game)
    action_size = game.num_distinct_actions()
    policies, masks, table = [], [], {}
    for row in rows:
        policy = np.zeros(action_size, dtype=np.float32)
        policy[list(row.legal_actions)] = 1.0 / len(row.legal_actions)
        mask = np.zeros(action_size, dtype=np.float32)
        mask[list(row.legal_actions)] = 1.0
        policies.append(policy)
        masks.append(mask)
        table[row.key] = policy.astype(np.float64)
    model = MLP(len(rows[0].tensor), [8], action_size)
    return game, rows, SourceData(
        checkpoint_id="time_36h",
        seed=7,
        iteration=2,
        gamma=2.0,
        network_layers=(8,),
        learning_rate=1e-3,
        batch_size=16,
        train_steps=2,
        infostates=np.asarray([row.tensor for row in rows], dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        legal_masks=np.asarray(masks, dtype=np.float32),
        iterations=np.ones(len(rows), dtype=np.float32),
        exact_table=table,
        exact_denominators={key: 1.0 for key in table},
        source_model_state=model.state_dict(),
    )


def _builder_module():
    path = REPOSITORY / "gcp" / "policy_post_training_batch.py"
    spec = importlib.util.spec_from_file_location("post_training_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, method, kind):
    return SimpleNamespace(
        kind=kind,
        method=method,
        repo_url=builder.REPO_URL,
        repo_ref="a" * 40,
        experiment_29_run_id="exp29-source",
        bucket_root="gs://example/results",
        run_id=f"exp{METHODS[method]['experiment_id']}-test",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_assigns_four_separate_experiments_to_same_sources():
    assert [METHODS[key]["experiment_id"] for key in METHODS] == [36, 37, 38, 39]
    assert len(PRODUCTION_SEEDS) == 5
    for method in METHODS:
        assert len(method_config(method)["arms"]) == 3
        assert evaluation_steps(method)[0] == 0
        assert evaluation_steps(method)[-1] == method_config(method)["updates"]


def test_each_method_executes_smoke_updates_on_real_leduc():
    game, rows, source = _source()
    blueprint = model_table(game, make_model(source))
    for method_id, runner in METHOD_RUNNERS.items():
        config = method_config(method_id, smoke=True)
        arm = next(iter(config["arms"].values()))
        observed = []
        runner(
            game=game,
            source=source,
            model=make_model(source),
            rows=rows,
            blueprint_table=blueprint,
            config=config,
            arm=arm,
            record=lambda model, update, loss, diagnostics: observed.append(update),
        )
        assert observed == [0, 1, 2]


def test_cloud_jobs_are_isolated_bounded_and_source_validated():
    builder = _builder_module()
    for method in METHODS:
        job = builder.build_job(_builder_args(builder, method, "train"))
        group = job["taskGroups"][0]
        policy = job["allocationPolicy"]["instances"][0]["policy"]
        script = group["taskSpec"]["runnables"][0]["script"]["text"]
        assert group["taskCount"] == 5
        assert group["parallelism"] == 5
        assert group["taskSpec"]["maxRetryCount"] == 1
        assert group["taskSpec"]["maxRunDuration"] == "43200s"
        assert policy["machineType"] == "n2-standard-4"
        assert policy["provisioningModel"] == "STANDARD"
        assert "POST_TRAINING_TASK" in script
        assert "Invalid post-training task metadata" in script
        assert "BATCH_TASK_RETRY_ATTEMPT" in script
        assert "promoted_ucv_cross_entropy_seed_${SOURCE_SEED}_time_36h.pt" in script


def test_aggregate_does_not_download_saved_policy_weights():
    builder = _builder_module()
    job = builder.build_job(
        _builder_args(builder, "best_response_guided_repair", "aggregate")
    )
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--exclude='(^|/)policies(/|$)'" in script


def test_root_readme_orders_experiments_after_35():
    text = (REPOSITORY / "README.md").read_text()
    positions = [text.index(f"## Experiment {number}:") for number in range(35, 40)]
    assert positions == sorted(positions)
    for launcher in (
        "run_best_response_guided_repair.sh",
        "run_kl_exploitability_descent.sh",
        "run_neurd_fine_tuning.sh",
        "run_ppo_self_play.sh",
    ):
        assert launcher in text

