import argparse
import json
import subprocess

import numpy as np
import pytest

from experiments.leduc_poker.rnad_ucv_resolving.config import SEEDS, protocol
from experiments.leduc_poker.rnad_ucv_resolving.run import (
    aggregate, case_id, cases, fingerprint, mean_interval, validate_completed,
)
from experiments.leduc_poker.grouped_ucv_temporal_head_to_head.run import digest, write_json
from gcp.exp49_rnad_resolving_batch import build_job


def test_protocol_and_case_coverage():
    full = cases("full")
    assert len(full) == 690
    assert len({case_id(c) for c in full}) == len(full)
    assert {c["seed"] for c in full} == set(SEEDS)
    for seed in SEEDS:
        for hour in (12, 36):
            selected = [c for c in full if c["seed"] == seed and c["hour"] == hour
                        and c["arm"] == "rnad_gadget" and c["seconds"] == .5]
            assert {c["repeat"] for c in selected} == set(range(8))
    assert all(c["hour"] == 36 for c in full if c["arm"] == "rnad_bayesian")
    assert len(cases("pilot")) == 12
    assert protocol()["utility_units"] == "native OpenSpiel Leduc chips"


def test_seed_level_interval_not_nested_pseudoreplication():
    interval = mean_interval([.1, .2, .3, .4, .5])
    assert interval["n_training_seeds"] == 5
    assert interval["mean"] == pytest.approx(.3)
    assert interval["se"] == pytest.approx(np.std([.1, .2, .3, .4, .5], ddof=1) / np.sqrt(5))
    assert mean_interval([.1])["ci95_low"] is None


def test_no_success_on_incomplete_run(tmp_path):
    manifest = dict(cases=cases("pilot"), mode="pilot", workers=1, contract_hash="test")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Missing required task"):
        aggregate(tmp_path)
    assert not (tmp_path / "SUCCESS.json").exists()


def test_resume_rejects_mismatch(tmp_path):
    (tmp_path / "tasks").mkdir()
    case = cases("pilot")[0]
    path = tmp_path / "tasks" / (case_id(case) + ".json")
    path.write_text(json.dumps(dict(case=case, contract_hash="wrong")))
    with pytest.raises(ValueError, match="incompatible"):
        validate_completed(tmp_path, case, "correct")
    assert fingerprint(protocol()) == fingerprint(protocol())


@pytest.mark.parametrize("mode", ["pilot", "full"])
def test_batch_cpu_and_shell_contract(tmp_path, mode):
    args = argparse.Namespace(bucket="gs://results", run_id="exp49-rnad-test",
                              source_uri="gs://source/exp35", repo_ref="a" * 40,
                              service_account="runner@example.iam.gserviceaccount.com",
                              workers=4, mode=mode, pilot_run_id="exp49-rnad-pilot",
                              max_hours=24)
    job = build_job(args)
    assert job["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-16"
    assert job["taskGroups"][0]["taskCount"] == 1
    task = job["taskGroups"][0]["taskSpec"]
    assert task["maxRetryCount"] == 0
    assert task["computeResource"]["cpuMilli"] == 16000
    script = task["runnables"][0]["script"]["text"]
    assert "CUDA_VISIBLE_DEVICES=\"\"" in script
    assert "--source-uri gs://source/exp35" in script
    assert "--workers 4" in script
    assert "--exclude='^SUCCESS\\.json$'" in script
    if mode == "full":
        assert "--pilot-output" in script
    path = tmp_path / "job.sh"
    path.write_text(script)
    subprocess.run(["bash", "-n", str(path)], check=True)


def test_full_job_needs_pilot():
    with pytest.raises(ValueError, match="completed pilot"):
        build_job(argparse.Namespace(mode="full", workers=4, pilot_run_id=None))


def test_complete_pilot_aggregation_and_hash_checks(tmp_path):
    expected = cases("pilot")
    manifest = dict(mode="pilot", workers=4, cases=expected, contract_hash="fixture")
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "experiment_metadata.json", {"synthetic_test_fixture": True})
    prep = dict(contract_hash="fixture", seed=SEEDS[0], hour=36, source_sha256="fixture",
                blueprint_loading_indexing_seconds=.1, blueprint_training_nodes=10,
                blueprint_completed_iteration=1, blueprint_active_seconds=129600,
                fit={"preparation_seconds": .1},
                baseline={"exploitability": .03, "panel_evs": {"1": 0.0}})
    write_json(tmp_path / "preparation" / f"seed{SEEDS[0]}_h36.json", prep)
    for case in expected:
        name = case_id(case)
        path = tmp_path / "policies" / f"{name}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez(path, probabilities=np.array([[.5, .5]]))
        timing = {"solve_and_diagnostics_seconds": 1., "exact_evaluation_seconds": .1}
        scores = dict(exploitability=.025, exploitability_change=-.005, ev_vs_blueprint=.001,
                      panel_mean_ev=.001, panel_ev_change=.001, exact_evaluation_seconds=.1,
                      panel_evs={"1": .001})
        write_json(tmp_path / "tasks" / f"{name}.json", dict(
            case=case, contract_hash="fixture", policy_file=f"policies/{name}.npz",
            policy_sha256=digest(path), timing=timing, scores=scores,
            roots=[{"root": "fixture", "hero": 0, "elapsed_seconds": .1}]))
    aggregate(tmp_path)
    result = json.loads((tmp_path / "analysis/summary.json").read_text())
    assert result["cases_complete"] == 12
    assert all(r["n_training_seeds"] == 1 for r in result["groups"])
    assert all(r["ci95_low"] is None for r in result["groups"])
    assert result["runtime_extrapolation"]["serial_case_hours"] > 0
    assert (tmp_path / "SUCCESS.json").exists()
    assert (tmp_path / "analysis/experiment_metadata.json").exists()
    assert (tmp_path / "analysis/critic_preparation.csv").exists()
