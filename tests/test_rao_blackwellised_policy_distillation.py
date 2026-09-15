from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyspiel
import torch

from experiments.leduc_poker.average_policy_redistillation.distill import SourceData
from experiments.leduc_poker.rao_blackwellised_policy_distillation.config import (
    ARM_ORDER,
    BASELINE_ARM,
    CURRENT_SHARED,
    PRODUCTION_SEEDS,
    ROUTED_EXPERTS,
    WIDE_SHARED,
    validate_contract,
)
from experiments.leduc_poker.rao_blackwellised_policy_distillation.distill import (
    build_catalog,
    build_model,
    grouped_dataset,
    objective_equivalence,
    parameter_count,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    PRODUCTION_SEEDS as EXPERIMENT_29_SEEDS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _builder_module():
    path = REPOSITORY / "gcp" / "rao_blackwellised_policy_distillation_batch.py"
    spec = importlib.util.spec_from_file_location("exp34_batch", path)
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
        run_id="exp34-test",
        experiment_29_run_id="exp29-source",
        parallelism=5,
        service_account="batch@example.iam.gserviceaccount.com",
        project_id="example-project",
        region="europe-west1",
        controller_action="orchestrate",
    )


def _source() -> tuple[SourceData, object]:
    game = pyspiel.load_game("leduc_poker")
    catalog = build_catalog(game)
    selected = list(catalog.rows.items())[:2]
    features, policies, masks, iterations = [], [], [], []
    exact_table, denominators = {}, {}
    for state_index, (key, row) in enumerate(selected):
        legal = list(row["legal_actions"])
        first = np.zeros(game.num_distinct_actions(), dtype=np.float32)
        second = np.zeros_like(first)
        first[legal[0]] = 1.0
        second[legal[-1]] = 1.0
        mask = np.zeros_like(first)
        mask[legal] = 1.0
        for policy, iteration in ((first, 2.0), (second, 8.0)):
            features.append(row["tensor"])
            policies.append(policy)
            masks.append(mask)
            iterations.append(iteration)
        exact_table[key] = (first + second) / 2.0
        denominators[key] = float(state_index + 1)
    source = SourceData(
        checkpoint_id="time_24h",
        seed=1,
        iteration=10,
        gamma=2.0,
        network_layers=(64, 64, 64),
        learning_rate=1e-3,
        batch_size=4,
        train_steps=5,
        infostates=np.asarray(features, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        legal_masks=np.asarray(masks, dtype=np.float32),
        iterations=np.asarray(iterations, dtype=np.float32),
        exact_table=exact_table,
        exact_denominators=denominators,
        source_model_state={},
    )
    return source, catalog


def test_contract_is_six_arm_factorial_over_experiment_29_sources():
    assert PRODUCTION_SEEDS == EXPERIMENT_29_SEEDS
    assert len(ARM_ORDER) == 6
    assert BASELINE_ARM in ARM_ORDER
    validate_contract(seeds=PRODUCTION_SEEDS, smoke=False)


def test_grouped_targets_preserve_soft_target_cross_entropy_exactly():
    source, catalog = _source()
    grouped = grouped_dataset(source, catalog)
    assert len(grouped.features) == 2
    probabilities = source.legal_masks / source.legal_masks.sum(axis=1, keepdims=True)
    row_value, grouped_value = objective_equivalence(source, catalog, probabilities)
    assert np.isclose(row_value, grouped_value, rtol=1e-6, atol=1e-7)


def test_wide_control_is_parameter_matched_to_routed_experts():
    current = build_model(CURRENT_SHARED, 30, 3)
    wide = build_model(WIDE_SHARED, 30, 3)
    routed = build_model(ROUTED_EXPERTS, 30, 3)
    assert parameter_count(current) == 10_499
    assert parameter_count(wide) == 41_891
    assert parameter_count(routed) == 41_996
    assert abs(parameter_count(wide) - parameter_count(routed)) / parameter_count(routed) < 0.01
    output = routed(torch.zeros((4, 30)), torch.arange(4))
    assert output.shape == (4, 3)


def test_cloud_jobs_enforce_development_selection_validation_barrier():
    builder = _builder_module()
    development = builder.build_job(_args(builder, "development"))
    validation = builder.build_job(_args(builder, "validation"))
    assert development["taskGroups"][0]["taskCount"] == 5
    assert validation["taskGroups"][0]["taskCount"] == 5
    assert development["allocationPolicy"]["instances"][0]["policy"]["machineType"] == "n2-standard-8"
    development_script = development["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    validation_script = validation["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
    assert "time_24h.pt" in development_script
    assert "development-worker" in development_script
    assert "time_36h.pt" in validation_script
    assert "selected_configs.json" in validation_script
    assert "validation-worker" in validation_script
    assert "$HOME" not in development_script + validation_script


def test_root_readme_places_experiment_34_after_33():
    text = (REPOSITORY / "README.md").read_text()
    assert text.index("## Experiment 33:") < text.index("## Experiment 34:")
    section = text.split("## Experiment 34:", 1)[1]
    assert "run_rao_blackwellised_policy_distillation.sh smoke-local" in section
    assert "run_rao_blackwellised_policy_distillation.sh run" in section
