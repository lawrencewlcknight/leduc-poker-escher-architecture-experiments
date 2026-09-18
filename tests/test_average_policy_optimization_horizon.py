from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.leduc_poker.average_policy_optimization_horizon.analyse import (
    aggregate_workers,
)
from experiments.leduc_poker.average_policy_optimization_horizon.config import (
    DIAGNOSTIC_UPDATES,
    PRODUCTION_SEEDS,
    REQUIRED_CHECKPOINT_UPDATES,
    contract_manifest,
)
from experiments.leduc_poker.average_policy_optimization_horizon.fitting import (
    GroupedReplayDataset,
    fit_continuously,
    grouped_replay_dataset,
    objective_value,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    write_csv,
    write_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "average_policy_optimization_horizon_batch.py"
    spec = importlib.util.spec_from_file_location("exp41_batch", path)
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
        run_id="exp41-test",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_has_one_continuous_fixed_horizon_schedule():
    assert PRODUCTION_SEEDS == (104729, 130363, 155921, 181081, 205759)
    assert REQUIRED_CHECKPOINT_UPDATES == (400, 800, 1200, 1600, 2000)
    assert DIAGNOSTIC_UPDATES[0] == 0
    assert DIAGNOSTIC_UPDATES[-1] == 2000
    assert set(REQUIRED_CHECKPOINT_UPDATES).issubset(DIAGNOSTIC_UPDATES)
    manifest = contract_manifest()
    assert manifest["training_requires_game_tree"] is False
    assert manifest["checkpoint_selection"] == "none_fixed_horizon_convergence_map"


def test_grouped_targets_match_iteration_weighted_uniform_repair_objective():
    source = SimpleNamespace(
        infostates=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
        policies=np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], dtype=np.float32
        ),
        legal_masks=np.ones((3, 2), dtype=np.float32),
        iterations=np.asarray([1.0, 3.0, 2.0], dtype=np.float32),
        gamma=1.0,
    )
    dataset = grouped_replay_dataset(source)
    assert dataset.replay_rows == 3
    assert len(dataset.features) == 2
    lookup = {
        tuple(feature.tolist()): target
        for feature, target in zip(dataset.features, dataset.targets)
    }
    assert lookup[(1.0, 0.0)] == pytest.approx([0.25, 0.75])
    assert lookup[(0.0, 1.0)] == pytest.approx([0.25, 0.75])


def test_continuous_fit_does_not_restart_between_checkpoints():
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    dataset = GroupedReplayDataset(
        features=np.eye(2, dtype=np.float32),
        targets=np.eye(2, dtype=np.float32),
        masks=np.ones((2, 2), dtype=np.float32),
        replay_rows=2,
    )
    records = []

    def record(update, loss, fitted_model):
        records.append((update, loss, objective_value(fitted_model, dataset)))

    fit_continuously(
        model=model,
        dataset=dataset,
        learning_rate=0.1,
        gradient_clip_norm=10.0,
        evaluation_updates=(0, 1, 2, 4),
        callback=record,
    )
    assert [row[0] for row in records] == [0, 1, 2, 4]
    assert records[-1][2] < records[0][2]


def test_cloud_builder_is_bounded_resumable_and_excludes_policy_weights():
    builder = _builder_module()
    train = builder.build_job(_builder_args(builder, "train"))
    group = train["taskGroups"][0]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == 5
    assert group["parallelism"] == 5
    assert group["taskSpec"]["maxRetryCount"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "7200s"
    assert "Invalid Experiment 41 task metadata" in script
    assert "promoted_ucv_cross_entropy_seed_${SOURCE_SEED}_time_36h.pt" in script
    aggregate = builder.build_job(_builder_args(builder, "aggregate"))
    aggregate_script = aggregate["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "--exclude='.*policies.*'" in aggregate_script


def test_cloud_builder_imports_without_scientific_site_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(REPOSITORY / "gcp" / "average_policy_optimization_horizon_batch.py"),
            "--help",
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_production_shaped_aggregation_uses_five_seed_units(tmp_path):
    root = tmp_path / "workers"
    for index, seed in enumerate(PRODUCTION_SEEDS):
        worker = root / f"task_{index:03d}_policy_fit_seed_{seed}"
        rows = []
        for update in DIAGNOSTIC_UPDATES:
            exploitability = 0.055 + index * 0.001 - update * 0.00001
            rows.append(
                {
                    "source_seed": seed,
                    "optimizer_update": update,
                    "exploitability": exploitability,
                    "change_from_archived_policy": exploitability - (0.055 + index * 0.001),
                    "gap_to_empirical_reservoir": exploitability - 0.030,
                    "gap_to_exact_tabular_average": exploitability - 0.028,
                    "training_loss_before_update": 0.5,
                    "full_batch_cross_entropy": 0.6 - update * 0.0001,
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
                "experiment_name": "average_policy_optimization_horizon",
                "source_seed": seed,
                "smoke": False,
                "status": "complete",
                "repository_commit": "f" * 40,
                "contract": contract_manifest(),
                "runtime_schedule": {
                    "diagnostic_updates": list(DIAGNOSTIC_UPDATES),
                    "required_checkpoint_updates": list(
                        REQUIRED_CHECKPOINT_UPDATES
                    ),
                },
                "source": {"sha256": str(seed)},
                "dataset": {
                    "replay_rows": 1000,
                    "unique_information_states": 100,
                },
                "baselines": {
                    "archived_neural_policy": 0.055 + index * 0.001,
                    "empirical_reservoir_policy": 0.030,
                    "exact_tabular_average": 0.028,
                },
                "snapshots": [
                    {
                        "optimizer_update": update,
                        "relative_path": f"policies/{update}.pt",
                        "sha256": str(update),
                        "size_bytes": 1,
                    }
                    for update in REQUIRED_CHECKPOINT_UPDATES
                ],
                "peak_rss_mb": 100.0,
                "wall_clock_seconds": 10.0,
                "artifacts": {"metrics": "fit_metrics.csv"},
            },
        )
    result = aggregate_workers(
        workers_root=root,
        seeds=PRODUCTION_SEEDS,
        output_dir=tmp_path / "analysis",
        smoke=False,
    )
    assert result["status"] == "complete"
    assert result["num_source_seeds"] == 5
    assert result["descriptive_convergence"]["best_observed_update"] == 2000
    assert (tmp_path / "analysis" / "exploitability_by_optimizer_update.png").is_file()


def test_root_readme_places_experiment_41_after_40():
    text = (REPOSITORY / "README.md").read_text()
    assert text.index("## Experiment 40:") < text.index("## Experiment 41:")
    assert "run_average_policy_optimization_horizon.sh" in text
