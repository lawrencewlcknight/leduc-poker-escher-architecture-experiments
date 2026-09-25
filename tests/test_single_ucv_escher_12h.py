from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table,
)
from experiments.leduc_poker.single_ucv_escher_12h.config import (
    BOUNDED_CAPACITIES,
    CANDIDATE_ID,
    PRODUCTION_SEEDS,
    SMOKE_SEEDS,
    checkpoint_schedule,
    task_schedule,
    validate_contract,
)
from experiments.leduc_poker.single_ucv_escher_12h.historical_policy import (
    HistoricalEntry,
    streaming_weighted_reservoir_indices,
    weighted_average_table,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "single_ucv_escher_12h_batch.py"
    spec = importlib.util.spec_from_file_location("exp44_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _builder_args(builder, kind: str):
    return SimpleNamespace(
        kind=kind, repo_url=builder.REPO_URL, repo_ref="a" * 40,
        bucket_root="gs://example/results", run_id="exp44-test",
        parallelism=3,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project", region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_uses_three_fresh_seeds_and_two_hour_schedule():
    assert len(PRODUCTION_SEEDS) == len(set(PRODUCTION_SEEDS)) == 3
    assert task_schedule() == tuple((CANDIDATE_ID, seed) for seed in PRODUCTION_SEEDS)
    assert [row["checkpoint_id"] for row in checkpoint_schedule()] == [
        "time_02h", "time_04h", "time_06h", "time_08h", "time_10h", "time_12h"
    ]
    assert BOUNDED_CAPACITIES == (16, 32, 64, 128)
    validate_contract(
        seeds=PRODUCTION_SEEDS, schedule=checkpoint_schedule(), smoke=False
    )
    validate_contract(
        seeds=SMOKE_SEEDS, schedule=checkpoint_schedule(smoke=True), smoke=True
    )


def test_weighted_historical_average_matches_identical_components():
    game = pyspiel.load_game("leduc_poker")

    def uniform(state):
        result = np.zeros(game.num_distinct_actions(), dtype=np.float64)
        legal = state.legal_actions()
        result[legal] = 1.0 / len(legal)
        return result

    table = build_policy_table(game, uniform)
    observed = weighted_average_table(game, [(table, 1.0), (table, 9.0)])
    assert set(observed) == set(table)
    assert all(np.allclose(observed[key], table[key]) for key in table)


def test_streaming_weighted_reservoir_is_deterministic_and_bounded(tmp_path):
    entries = [
        HistoricalEntry(index, float(index), tmp_path / f"{index}.pt", "x", 1, {})
        for index in range(1, 11)
    ]
    left = streaming_weighted_reservoir_indices(entries, 8, seed=91)
    right = streaming_weighted_reservoir_indices(entries, 8, seed=91)
    assert left == right
    assert len(left) == 8
    assert all(0 <= index < len(entries) for index in left)


def test_cloud_contract_has_three_parallel_standard_workers():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == group["parallelism"] == 3
    assert group["taskCountPerNode"] == 1
    assert group["taskSpec"]["maxRunDuration"] == "64800s"
    assert group["taskSpec"]["maxRetryCount"] == 1
    assert policy["machineType"] == "n2-standard-8"
    assert policy["provisioningModel"] == "STANDARD"
    assert "EXP44_REMOTE_TASK_URI" in script
    assert "Invalid Experiment 44 task metadata" in script
    assert "$HOME" not in script


def test_aggregate_excludes_large_historical_and_continuation_files():
    builder = _builder_module()
    job = builder.build_job(_builder_args(builder, "aggregate"))
    script = job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "training_states|diagnostics|historical_networks" in script
    assert "single_ucv_escher_12h.run aggregate" in script


def test_root_readme_places_experiment_44_after_43():
    text = (REPOSITORY / "README.md").read_text()
    assert text.index("## Experiment 43:") < text.index("## Experiment 44:")
    section = text.split("## Experiment 44:", 1)[1]
    assert "run_single_ucv_escher_12h.sh smoke-local" in section
    assert "run_single_ucv_escher_12h.sh run" in section
