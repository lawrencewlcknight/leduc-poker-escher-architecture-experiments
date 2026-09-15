from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pyspiel

from experiments.leduc_poker.causal_rare_state_audit.audit import _infoset_catalog
from experiments.leduc_poker.consequence_weighted_distillation.config import (
    ARM_ORDER,
    PRODUCTION_SEEDS,
    PROXY_ORDER,
    SELECTABLE_PROXY_ORDER,
    SOURCE_CHECKPOINTS,
    TARGETED_BATCH_FRACTION,
    validate_contract,
)
from experiments.leduc_poker.consequence_weighted_distillation.distill import (
    _normalised_clipped_weights,
)
from experiments.leduc_poker.consequence_weighted_distillation.proxy import (
    counterfactual_action_values,
    select_proxy,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import read_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    PRODUCTION_SEEDS as EXPERIMENT_29_SEEDS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "consequence_weighted_distillation_batch.py"
    spec = importlib.util.spec_from_file_location("exp33_batch", path)
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
        run_id="exp33-test",
        experiment_29_run_id="exp29-source",
        experiment_30_run_id="exp30-source",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def test_contract_reuses_five_experiment_29_sources():
    assert PRODUCTION_SEEDS == EXPERIMENT_29_SEEDS
    assert SOURCE_CHECKPOINTS == ("time_24h", "time_36h")
    assert len(PROXY_ORDER) >= len(SELECTABLE_PROXY_ORDER) >= 3
    assert len(ARM_ORDER) == 6
    assert 0.0 < TARGETED_BATCH_FRACTION < 1.0
    validate_contract(seeds=PRODUCTION_SEEDS, smoke=False)


def test_counterfactual_values_cover_every_leduc_information_set():
    game = pyspiel.load_game("leduc_poker")
    catalog = _infoset_catalog(game)
    uniform = {}
    for key, row in catalog.items():
        policy = np.zeros(game.num_distinct_actions(), dtype=np.float64)
        policy[list(row["legal_actions"])] = 1.0 / len(row["legal_actions"])
        uniform[key] = policy
    values = counterfactual_action_values(game, uniform)
    assert set(values) == set(catalog)
    assert all(row["counterfactual_reach"] > 0.0 for row in values.values())
    assert all(row["action_value_span"] >= 0.0 for row in values.values())


def test_weight_transform_is_finite_positive_and_mean_one():
    weights = _normalised_clipped_weights(np.asarray([0.0, 0.1, 1.0, 10.0]))
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert np.isclose(float(np.mean(weights)), 1.0)


def test_proxy_selection_cannot_choose_error_or_oracle_target(tmp_path):
    rows = []
    for proxy_index, proxy in enumerate(PROXY_ORDER):
        rows.append(
            {
                "checkpoint_id": "time_24h",
                "repair_fraction": 0.10,
                "proxy_id": proxy,
                "positive_single_gain_capture": 0.1 + proxy_index,
                "fraction_distillation_gap_recovered": 0.1,
                "spearman_positive_gain": 0.1,
            }
        )
    result = select_proxy(rows, tmp_path)
    assert result["selected_proxy_id"] in SELECTABLE_PROXY_ORDER
    assert read_json(tmp_path / "selected_proxy.json")["status"] == "complete"


def test_cloud_jobs_enforce_proxy_selection_barrier():
    builder = _builder_module()
    proxy = builder.build_job(_args(builder, "proxy"))
    distill = builder.build_job(_args(builder, "distill"))
    assert proxy["taskGroups"][0]["taskCount"] == 5
    assert distill["taskGroups"][0]["taskCount"] == 5
    assert proxy["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    proxy_script = proxy["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    distill_script = distill["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "EXP30_RUN_ID" in proxy_script
    assert "information_set_audit.csv" in proxy_script
    assert "selected_proxy.json" in distill_script
    assert "distill-worker" in distill_script
    assert "$HOME" not in proxy_script + distill_script


def test_root_readme_places_experiment_33_after_32():
    text = (REPOSITORY / "README.md").read_text()
    assert text.index("## Experiment 32:") < text.index("## Experiment 33:")
    section = text.split("## Experiment 33:", 1)[1]
    assert "run_consequence_weighted_distillation.sh smoke-local" in section
    assert "run_consequence_weighted_distillation.sh run" in section


def test_launcher_checks_the_artifacts_each_source_experiment_produces(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "gcloud.log"
    fake = fake_bin / "gcloud"
    fake.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$GCLOUD_LOG"\n'
        'if [ "$1 $2" = "batch jobs" ] && [ "$3" = "submit" ]; then exit 0; fi\n'
        "exit 0\n"
    )
    fake.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GCLOUD_LOG": str(log),
        "PROJECT_ID": "example-project",
        "REGION": "europe-west1",
        "BUCKET": "gs://example/results",
        "SA_EMAIL": "batch@example.iam.gserviceaccount.com",
        "REPO_REF": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "EXP29_RUN_ID": "exp29-source",
        "EXP30_RUN_ID": "exp30-source",
        "RUN_ID": "exp33-test-run",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_consequence_weighted_distillation.sh", "run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    calls = log.read_text().splitlines()
    assert any("exp29-source/analysis/aggregate_manifest.json" in row for row in calls)
    assert any("exp30-source/analysis/aggregate_summary.json" in row for row in calls)
    assert any("batch jobs submit exp33-test-run-controller" in row for row in calls)
    assert "laptop may now be disconnected" in completed.stdout
