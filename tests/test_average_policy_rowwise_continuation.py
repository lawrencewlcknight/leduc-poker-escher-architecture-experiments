from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from experiments.leduc_poker.average_policy_rowwise_continuation.analyse import (
    aggregate_workers,
)
from experiments.leduc_poker.average_policy_rowwise_continuation.config import (
    ARM_IDS,
    BATCH_SIZE,
    DIAGNOSTIC_EQUIVALENT_UPDATES,
    EQUAL_EXAMPLE_ARM,
    EQUAL_UPDATE_ARM,
    PRODUCTION_SEEDS,
    SOURCE_SHA256,
    UNIQUE_INFORMATION_STATES,
    contract_manifest,
    runtime_config,
)
from experiments.leduc_poker.average_policy_rowwise_continuation.fitting import (
    fit_rowwise_continuously,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    write_csv,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "average_policy_rowwise_continuation_batch.py"
    spec = importlib.util.spec_from_file_location("exp42_batch", path)
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
        experiment_41_run_id="exp41-reference",
        bucket_root="gs://example/results",
        run_id="exp42-test",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_freezes_equal_example_budgets_and_source_digests():
    assert PRODUCTION_SEEDS == (104729, 130363, 155921, 181081, 205759)
    assert DIAGNOSTIC_EQUIVALENT_UPDATES[-1] == 1700
    assert all(len(SOURCE_SHA256[seed]) == 64 for seed in PRODUCTION_SEEDS)
    for seed in PRODUCTION_SEEDS:
        config = runtime_config(seed=seed)
        assert config["equal_example_budget"] == 1700 * UNIQUE_INFORMATION_STATES[seed]
        assert config["equal_update_budget"] == 1700
        assert config["batch_size"] == BATCH_SIZE
    manifest = contract_manifest()
    assert manifest["grouping_in_training_path"] is False
    assert manifest["training_requires_game_tree"] is False


def _tiny_source():
    return SimpleNamespace(
        infostates=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
            dtype=np.float32,
        ),
        policies=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.75, 0.25], [0.25, 0.75]],
            dtype=np.float32,
        ),
        legal_masks=np.ones((4, 2), dtype=np.float32),
        iterations=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        iteration=4,
        gamma=2.0,
    )


def test_rowwise_arms_hit_exact_resource_budgets():
    for arm_id in ARM_IDS:
        torch.manual_seed(9)
        model = torch.nn.Linear(2, 2)
        records = []
        fit_rowwise_continuously(
            model=model,
            source=_tiny_source(),
            arm_id=arm_id,
            unique_information_states=3,
            batch_size=4,
            learning_rate=0.01,
            gradient_clip_norm=10.0,
            diagnostic_equivalent_updates=(0, 2, 4),
            sampling_seed=17,
            callback=lambda progress, examples, steps, loss, fitted: records.append(
                (progress, examples, steps)
            ),
        )
        assert [row[0] for row in records] == [0, 2, 4]
        if arm_id == EQUAL_EXAMPLE_ARM:
            assert records[-1][1] == 12
        else:
            assert records[-1][1:] == (16, 4)


def test_training_module_contains_no_grouping_operation():
    from experiments.leduc_poker.average_policy_rowwise_continuation import fitting

    source = inspect.getsource(fitting)
    assert "np.unique" not in source
    assert "grouped_replay_dataset" not in source


def test_cloud_builder_is_bounded_and_imports_both_references():
    builder = _builder_module()
    train = builder.build_job(_builder_args(builder, "train"))
    group = train["taskGroups"][0]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == 5
    assert group["parallelism"] == 5
    assert group["taskSpec"]["maxRunDuration"] == "7200s"
    assert "Invalid Experiment 42 task metadata" in script
    aggregate = builder.build_job(_builder_args(builder, "aggregate"))
    aggregate_script = aggregate["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "$EXP41_RUN_ID/analysis" in aggregate_script
    assert "--exclude='.*policies.*'" in aggregate_script


def test_cloud_builder_imports_without_scientific_site_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(REPOSITORY / "gcp" / "average_policy_rowwise_continuation_batch.py"),
            "--help",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_shaped_aggregation_joins_experiment_41(tmp_path):
    worker = tmp_path / "workers" / "task_000_rowwise_policy_seed_0"
    updates = (0, 1, 2, 4)
    rows = []
    for arm_index, arm_id in enumerate(ARM_IDS):
        for update in updates:
            value = 0.08 - update * 0.005 + arm_index * 0.001
            rows.append(
                {
                    "source_seed": 0,
                    "arm_id": arm_id,
                    "equivalent_update": update,
                    "examples_seen": update * (12 if arm_id == EQUAL_EXAMPLE_ARM else 8),
                    "optimizer_steps": update,
                    "exploitability": value,
                    "change_from_archived_policy": value - 0.08,
                    "gap_to_exact_tabular_average": value - 0.04,
                    "last_minibatch_loss": 0.4,
                    "diagnostic_row_cross_entropy": 0.4,
                    "fit_elapsed_seconds": update,
                    "mean_l1": 0.2,
                    "max_l1": 0.4,
                    "reach_weighted_l1": 0.1,
                    "mean_kl": 0.03,
                    "reach_weighted_kl": 0.02,
                }
            )
    write_csv(worker / "fit_metrics.csv", rows)
    write_json(
        worker / "worker_result.json",
        {
            "experiment_name": "average_policy_rowwise_continuation",
            "source_seed": 0,
            "smoke": True,
            "status": "complete",
            "repository_commit": "f" * 40,
            "runtime_schedule": {
                "diagnostic_equivalent_updates": list(updates),
                "required_checkpoint_updates": [1, 2, 4],
                "unique_information_states": 12,
                "batch_size": 8,
            },
            "source": {"sha256": "source"},
            "dataset": {
                "replay_rows": 20,
                "grouping_performed_in_training_path": False,
            },
            "baselines": {
                "archived_neural_policy": 0.08,
                "exact_tabular_average": 0.04,
            },
            "arm_runtime_seconds": {arm_id: 1.0 for arm_id in ARM_IDS},
            "snapshots": [],
            "peak_rss_mb": 100.0,
            "wall_clock_seconds": 2.0,
            "artifacts": {"metrics": "fit_metrics.csv"},
        },
    )
    reference = tmp_path / "reference" / "analysis"
    write_json(
        reference / "aggregate_summary.json",
        {"status": "complete", "contract": {"experiment_id": 41}},
    )
    write_csv(
        reference / "fit_metrics.csv",
        [
            {"source_seed": 0, "optimizer_update": update,
             "exploitability": 0.08 - update * 0.006}
            for update in updates
        ],
    )
    write_csv(
        reference / "worker_manifest.csv",
        [{"source_seed": 0, "source_sha256": "source",
          "unique_information_states": 12}],
    )
    result = aggregate_workers(
        workers_root=tmp_path / "workers", seeds=(0,),
        output_dir=tmp_path / "analysis", experiment_41_root=reference,
        smoke=True,
    )
    assert result["status"] == "complete"
    assert result["num_source_seeds"] == 1
    assert (tmp_path / "analysis" / "rowwise_vs_grouped_exploitability.png").is_file()


def test_root_readme_places_experiment_42_after_41():
    text = (REPOSITORY / "README.md").read_text()
    assert text.index("## Experiment 41:") < text.index("## Experiment 42:")
    assert "run_average_policy_rowwise_continuation.sh" in text
