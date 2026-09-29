"""Contracts for evaluation-only architecture Experiment 45."""

from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace

import pytest
import pyspiel
from open_spiel.python import policy

from experiments.leduc_poker.grouped_ucv_temporal_head_to_head import run as runner
from experiments.leduc_poker.grouped_ucv_temporal_head_to_head.config import (
    CANDIDATE_ID, EXPECTED_CONFIG, HOURS, SEEDS, SMOKE_HOURS, SMOKE_SEEDS,
    checkpoint_id,
)
from experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.statistics import (
    build_inference_tables,
)
from escher_poker.checkpoint_analysis import exact_seat_averaged_value_for_a

ROOT = Path(__file__).parents[1]


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "source"
    (folder / "analysis").mkdir(parents=True)
    (folder / "snapshots").mkdir()
    rows, metrics = [], []
    for hour in SMOKE_HOURS:
        seed = SMOKE_SEEDS[0]
        filename = f"{CANDIDATE_ID}_seed_{seed}_{checkpoint_id(hour)}.pkl"
        payload = {
            "candidate_id": CANDIDATE_ID, "checkpoint_id": checkpoint_id(hour),
            "seed": seed, "repository_commit": "a"*40, "game": "leduc_poker",
            "framework": "pytorch", "checkpoint_type": "active_time",
            "checkpoint_target_active_hours": hour,
            "active_seconds": hour*3600+1, "nodes_touched": hour*100,
            "completed_iteration": hour, "frozen_config": deepcopy(EXPECTED_CONFIG),
            "policy_network_layers": [136, 136, 136],
        }
        path = folder / "snapshots" / filename
        path.write_bytes(pickle.dumps(payload))
        row = {key: payload[key] for key in (
            "candidate_id", "checkpoint_id", "seed", "repository_commit",
            "checkpoint_type", "checkpoint_target_active_hours", "active_seconds",
            "nodes_touched", "completed_iteration",
        )}
        rows.append({**row, "filename": filename, "sha256": runner.digest(path)})
        metrics.append({
            "seed": seed, "checkpoint_id": checkpoint_id(hour),
            "policy_id": CANDIDATE_ID, "nodes_touched": hour*100,
            "completed_iteration": hour, "exploitability": .1,
        })
    runner.write_csv(folder / "analysis/snapshot_inventory.csv", rows)
    runner.write_csv(folder / "analysis/checkpoint_policy_metrics.csv", metrics)
    runner.write_json(folder / "analysis/aggregate_summary.json", {"contract": {
        "experiment_id": 35, "candidate_id": CANDIDATE_ID,
        "production_seeds": SEEDS, "candidate_config": EXPECTED_CONFIG,
    }})
    return folder


def test_complete_production_design():
    assert len(SEEDS) == 5 and HOURS == tuple(range(2, 37, 2))
    assert len(SEEDS) * len(list(itertools.combinations(HOURS, 2))) == 765
    assert EXPECTED_CONFIG["average_policy_network_layers"] == [136]*3
    assert EXPECTED_CONFIG["average_policy_train_steps"] == 20000


def test_strict_inventory_and_source_provenance(source):
    rows = runner.source_records(source, SMOKE_SEEDS, SMOKE_HOURS)
    assert len(rows) == 3
    with pytest.raises(ValueError, match="Missing required"):
        runner.source_records(source, SEEDS, HOURS)


def test_corrupted_checkpoint_rejected_before_loading(source):
    rows = runner.selected_inventory(source, SMOKE_SEEDS, SMOKE_HOURS)
    (source / "snapshots" / rows[0]["filename"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.source_records(source, SMOKE_SEEDS, SMOKE_HOURS)


@pytest.mark.parametrize("field,value", [
    ("candidate_id", "legacy_row_policy"),
    ("filename", "../escape.pkl"),
    ("sha256", "not-a-checksum"),
])
def test_wrong_candidate_or_unsafe_manifest_rejected(source, field, value):
    path = source / "analysis/snapshot_inventory.csv"
    rows = runner.read_csv(path)
    rows[0][field] = value
    runner.write_csv(path, rows)
    with pytest.raises(ValueError):
        runner.selected_inventory(source, SMOKE_SEEDS, SMOKE_HOURS)


def test_wrong_architecture_rejected_even_with_valid_hash(source):
    path = source / "analysis/snapshot_inventory.csv"
    rows = runner.read_csv(path)
    snapshot = source / "snapshots" / rows[0]["filename"]
    payload = pickle.loads(snapshot.read_bytes())
    payload["frozen_config"]["q_ensemble_size"] = 3
    snapshot.write_bytes(pickle.dumps(payload))
    rows[0]["sha256"] = runner.digest(snapshot)
    runner.write_csv(path, rows)
    with pytest.raises(ValueError, match="Not the confirmed grouped candidate"):
        runner.source_records(source, SMOKE_SEEDS, SMOKE_HOURS)


def test_duplicate_record_rejected(source):
    path = source / "analysis/snapshot_inventory.csv"
    rows = runner.read_csv(path)
    runner.write_csv(path, rows + rows[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        runner.selected_inventory(source, SMOKE_SEEDS, SMOKE_HOURS)


def test_seat_averaging_is_zero_for_self_and_antisymmetric():
    game = pyspiel.load_game("leduc_poker")
    a, b = policy.TabularPolicy(game), policy.TabularPolicy(game)
    b.action_probability_array[:] = b.legal_actions_mask
    b.action_probability_array[:, 0] *= 3
    b.action_probability_array[:] /= b.action_probability_array.sum(axis=1, keepdims=True)
    ab = exact_seat_averaged_value_for_a(game, a, b)["A_EV_seat_averaged"]
    ba = exact_seat_averaged_value_for_a(game, b, a)["A_EV_seat_averaged"]
    assert abs(ab) > 1e-6
    assert ab == pytest.approx(-ba, abs=1e-12)
    assert exact_seat_averaged_value_for_a(game, a, a)["A_EV_seat_averaged"] == pytest.approx(0, abs=1e-12)


def test_inference_keeps_five_seeds_not_765_pseudo_replicates():
    pairs = [
        {"seed": seed, "checkpoint_a": b, "checkpoint_b": a,
         "A_EV_seat_averaged": .001*(b-a)}
        for seed in SEEDS for a, b in itertools.combinations(HOURS, 2)
    ]
    seed_rows, summary, secondary = build_inference_tables(pairs, HOURS)
    assert len(seed_rows) == 5 and len(secondary) == 153
    assert summary[0]["n_seeds"] == 5
    assert summary[0]["exact_one_sided_sign_flip_p"] == 1/32
    assert all(r["holm_adjusted_p"] == 1 for r in secondary)


def test_outputs_and_cache_reuse(source, tmp_path, monkeypatch):
    def fake_evaluate(rows):
        metrics = [{
            "seed": int(r["seed"]), "checkpoint": int(float(r["checkpoint_target_active_hours"])),
            "actual_active_hours": float(r["active_seconds"])/3600,
            "nodes_touched": int(r["nodes_touched"]), "completed_iteration": int(r["completed_iteration"]),
            "exploitability": .1, "source_exploitability": .1, "reload_absolute_error": 0,
            "self_play_seat_averaged_ev": 0,
        } for r in rows]
        pairs = [{"seed": SMOKE_SEEDS[0], "checkpoint_a": b, "checkpoint_b": a,
                  "A_EV_as_player0": .01, "A_EV_as_player1": .01, "A_EV_seat_averaged": .01}
                 for a, b in itertools.combinations(SMOKE_HOURS, 2)]
        return {"metrics": metrics, "pairs": pairs}
    monkeypatch.setattr(runner, "evaluate_seed", fake_evaluate)
    output = tmp_path / "output"
    result = runner.run(source, output, smoke=True)
    assert result["num_snapshots"] == 3
    assert result["num_unordered_same_seed_pairs"] == 3
    assert len(list((output / "analysis").glob("*.png"))) == 5
    # Simulate interruption after complete seed evaluation but before success marker.
    (output / "SUCCESS.json").unlink()
    monkeypatch.setattr(runner, "evaluate_seed", lambda _: pytest.fail("cache not reused"))
    runner.run(source, output, smoke=True)
    with pytest.raises(FileExistsError):
        runner.run(source, output, smoke=True)
    metadata = json.loads((output / "experiment_metadata.json").read_text())
    assert not metadata["training_performed"]


def test_cloud_script_has_no_training_or_controller_and_bounded_cost():
    spec = importlib.util.spec_from_file_location(
        "exp45_builder", ROOT / "gcp/grouped_ucv_temporal_head_to_head_batch.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    job = builder.build_job(SimpleNamespace(
        bucket="gs://example", run_id="exp45-h2h-test", source_run_id="exp35-source",
        repo_ref="a"*40, service_account="runner@example.iam.gserviceaccount.com",
    ))
    task = job["taskGroups"][0]["taskSpec"]
    script = task["runnables"][0]["script"]["text"]
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert task["maxRunDuration"] == "14400s" and task["maxRetryCount"] == 0
    assert job["taskGroups"][0]["taskCount"] == 1
    assert "training_states" not in script
    assert " smoke " in script and " run " in script
    assert script.index(" smoke ") < script.index(" run ")
    assert "batch jobs submit" not in script
    assert 'exit "$code"' in script and "upload_code" in script
