from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pyspiel

from experiments.leduc_poker.causal_rare_state_audit import audit as audit_module
from experiments.leduc_poker.causal_rare_state_audit.audit import (
    _infoset_catalog,
    _repair_counts,
    audit_source,
)
from experiments.leduc_poker.causal_rare_state_audit.config import (
    PRODUCTION_SEEDS,
    RANKING_ORDER,
    REPAIR_FRACTIONS,
    SOURCE_CHECKPOINTS,
    validate_contract,
)
from experiments.leduc_poker.average_policy_redistillation.distill import SourceData
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    PRODUCTION_SEEDS as EXPERIMENT_29_SEEDS,
)
from vr_deep_cfr.solver import MLP


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "causal_rare_state_audit_batch.py"
    spec = importlib.util.spec_from_file_location("exp30_batch", path)
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
        run_id="exp30-test",
        experiment_29_run_id="exp29-source",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_reuses_all_five_experiment_29_trajectories():
    assert PRODUCTION_SEEDS == EXPERIMENT_29_SEEDS
    assert SOURCE_CHECKPOINTS == ("time_24h", "time_36h")
    assert REPAIR_FRACTIONS[0] == 0.0
    assert REPAIR_FRACTIONS[-1] == 1.0
    assert RANKING_ORDER[-1] == "random"
    validate_contract(seeds=PRODUCTION_SEEDS, smoke=False)


def test_repair_schedule_is_unique_and_ends_in_complete_repair():
    schedule = _repair_counts(288)
    counts = [count for _, count in schedule]
    assert counts[0] == 0
    assert counts[-1] == 288
    assert counts == sorted(set(counts))


def test_exact_policy_surgery_recovers_full_tabular_policy(monkeypatch):
    game = pyspiel.load_game("leduc_poker")
    catalog = _infoset_catalog(game)
    action_size = game.num_distinct_actions()
    tensors = np.stack([row["tensor"] for row in catalog.values()])
    masks = np.zeros((len(catalog), action_size), dtype=np.float32)
    policies = np.zeros_like(masks)
    exact_table = {}
    exact_denominators = {}
    for index, (key, row) in enumerate(catalog.items()):
        legal = list(row["legal_actions"])
        masks[index, legal] = 1.0
        policies[index, legal] = 1.0 / len(legal)
        exact_table[key] = policies[index].astype(np.float64)
        exact_denominators[key] = 1.0
    model = MLP(tensors.shape[1], [8], action_size)
    source = SourceData(
        checkpoint_id="time_36h",
        seed=0,
        iteration=1,
        gamma=1.0,
        network_layers=(8,),
        learning_rate=1e-3,
        batch_size=2,
        train_steps=1,
        infostates=tensors,
        policies=policies,
        legal_masks=masks,
        iterations=np.ones(len(catalog), dtype=np.float32),
        exact_table=exact_table,
        exact_denominators=exact_denominators,
        source_model_state=model.state_dict(),
    )

    # The end-to-end smoke uses OpenSpiel's real best-response oracle. This
    # unit test isolates the policy-replacement invariant with a cheap,
    # deterministic table functional so the normal test suite remains fast.
    monkeypatch.setattr(
        audit_module,
        "exact_exploitability",
        lambda _, table: float(
            sum(np.square(np.asarray(policy, dtype=np.float64)).sum() for policy in table.values())
        ),
    )
    result = audit_source(source, game, smoke=True)
    complete = [
        row
        for row in result.repair_curves
        if row["num_repaired"] == len(catalog)
    ]
    assert complete
    assert all(
        np.isclose(
            row["hybrid_exploitability"],
            result.source_summary["exact_average_exploitability"],
            atol=1e-12,
        )
        for row in complete
    )


def test_cloud_job_downloads_only_two_source_states_per_worker():
    builder = _builder_module()
    job = builder.build_job(_args(builder, "train"))
    group = job["taskGroups"][0]
    policy = job["allocationPolicy"]["instances"][0]["policy"]
    script = group["taskSpec"]["runnables"][0]["script"]["text"]
    assert group["taskCount"] == 5
    assert group["parallelism"] == 5
    assert group["taskSpec"]["maxRetryCount"] == 0
    assert group["taskSpec"]["maxRunDuration"] == "43200s"
    assert policy["machineType"] == "n2-standard-4"
    assert policy["provisioningModel"] == "STANDARD"
    assert "for CHECKPOINT in time_24h time_36h" in script
    assert "promoted_ucv_cross_entropy_seed_" in script
    assert "causal_rare_state_audit.run worker" in script
    assert "$HOME" not in script


def test_remote_controller_runs_smoke_before_audit():
    builder = _builder_module()
    smoke = builder.build_job(_args(builder, "smoke"))
    smoke_script = smoke["taskGroups"][0]["taskSpec"]["runnables"][0][
        "script"
    ]["text"]
    assert "causal_rare_state_audit.run smoke" in smoke_script
    controller = builder.build_job(_args(builder, "controller"))
    controller_script = controller["taskGroups"][0]["taskSpec"]["runnables"][
        0
    ]["script"]["text"]
    assert "EXP30_REMOTE_CONTROLLER=1" in controller_script
    assert 'run_causal_rare_state_audit.sh "$CONTROLLER_ACTION"' in controller_script


def test_local_run_returns_after_controller_submission(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "gcloud.log"
    fake = fake_bin / "gcloud"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$GCLOUD_LOG"\n')
    fake.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GCLOUD_LOG": str(log),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west1",
        "BUCKET": "gs://example/results",
        "SA_EMAIL": "batch@example.iam.gserviceaccount.com",
        "REPO_REF": "a" * 40,
        "EXP29_RUN_ID": "exp29-source",
        "RUN_ID": "exp30-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_causal_rare_state_audit.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=15,
    )
    calls = log.read_text().splitlines()
    assert len(calls) == 1
    assert "batch jobs submit exp30-test-run-controller" in calls[0]
    assert "laptop may now be disconnected" in completed.stdout


def test_readme_places_experiment_30_after_experiment_29():
    readme = (REPOSITORY / "README.md").read_text()
    assert readme.index("## Experiment 29:") < readme.index("## Experiment 30:")
    section = readme.split("## Experiment 30:", 1)[1]
    assert "run_causal_rare_state_audit.sh smoke-local" in section
    assert "run_causal_rare_state_audit.sh run" in section
    assert "retrains\nnothing" in section
