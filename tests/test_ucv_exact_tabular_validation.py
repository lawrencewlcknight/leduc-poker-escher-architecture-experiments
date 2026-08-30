"""Contracts for exact tabular UCV implementation validation."""

from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from experiments.leduc_poker.ucv_exact_tabular_validation.config import (
    BASELINE_FREE,
    BASE_CONFIG,
    CALIBRATION_DISABLED,
    CHECKPOINT_TARGETS,
    DEFAULT_SEEDS,
    FIXED_BETA_ONE,
    FULL_ADAPTIVE,
    PREDICTION_GATE_ZERO,
    TARGET_NODES,
    VARIANTS,
    checkpoint_contract,
)
from experiments.leduc_poker.ucv_exact_tabular_validation.diagnostics import (
    ExactUCVOracle,
    frozen_state_fingerprint,
    policy_table_for_mode,
    predictability_audit,
)
from experiments.leduc_poker.ucv_exact_tabular_validation.run import (
    _build_solver,
    run_sequential,
)


def _trained_tiny_solver():
    config = deepcopy(BASE_CONFIG)
    config.update(
        {
            "num_traversals": 2,
            "max_num_iterations": 2,
            "advantage_network_train_steps": 1,
            "ave_policy_network_train_steps": 1,
            "baseline_network_train_steps": 1,
            "calibration_train_steps": 1,
            "advantage_batch_size": 2,
            "ave_policy_batch_size": 2,
            "baseline_batch_size": 2,
            "calibration_batch_size": 2,
            "advantage_buffer_size": 128,
            "ave_policy_buffer_size": 128,
            "baseline_buffer_size": 128,
            "calibration_buffer_size": 128,
        }
    )
    solver = _build_solver(99991, config)
    solver._solve_start_time = time.perf_counter()
    solver.iteration()
    return solver


def test_production_contract_is_three_sequential_15m_seeds():
    assert DEFAULT_SEEDS == (0, 1, 2)
    assert TARGET_NODES == 15_000_000
    assert CHECKPOINT_TARGETS == (
        ("early", 1_500_000),
        ("middle", 7_500_000),
        ("late", 15_000_000),
    )
    assert checkpoint_contract(300) == (
        ("early", 30),
        ("middle", 150),
        ("late", 300),
    )


def test_counterfactual_contract_separates_both_predictor_meanings():
    assert VARIANTS[FULL_ADAPTIVE]["beta_mode"] == "adaptive"
    assert VARIANTS[FIXED_BETA_ONE]["beta_mode"] == "fixed_one"
    assert VARIANTS[PREDICTION_GATE_ZERO]["policy_mode"] == "cumulative_only"
    assert VARIANTS[CALIBRATION_DISABLED]["calibration_mode"] == "disabled"
    assert VARIANTS[CALIBRATION_DISABLED]["sampling_mode"] == "uniform"
    assert VARIANTS[BASELINE_FREE]["control_mode"] == "no_control"


def test_run_sequential_never_overlaps_or_reorders_seeds():
    calls = []

    def run_one(seed):
        calls.append(("start", seed))
        calls.append(("finish", seed))
        return seed * 2

    assert run_sequential([7, 3, 11], run_one) == [14, 6, 22]
    assert calls == [
        ("start", 7),
        ("finish", 7),
        ("start", 3),
        ("finish", 3),
        ("start", 11),
        ("finish", 11),
    ]


def test_predictability_source_order_contract_passes():
    audit = predictability_audit()
    assert audit["status"] == "pass"
    assert all(audit["checks"].values())


def test_exact_oracle_uses_implemented_estimator_without_mutation():
    solver = _trained_tiny_solver()
    before = frozen_state_fingerprint(solver)
    tables = {
        mode: policy_table_for_mode(solver, mode)
        for mode in {spec["policy_mode"] for spec in VARIANTS.values()}
    }
    rows_by_variant = {}
    for variant_id, spec in VARIANTS.items():
        rows = ExactUCVOracle(
            solver,
            policy_table=tables[spec["policy_mode"]],
            variant_id=variant_id,
            fold=0,
        ).rows()
        assert rows
        assert max(abs(row["action_value_bias"]) for row in rows) < 1e-12
        assert max(abs(row["advantage_bias"]) for row in rows) < 1e-12
        rows_by_variant[variant_id] = rows
    assert before == frozen_state_fingerprint(solver)
    assert all(
        np.isclose(row["beta"], 1.0)
        for row in rows_by_variant[FIXED_BETA_ONE]
    )
    assert all(
        np.isclose(row["beta"], 0.0) for row in rows_by_variant[BASELINE_FREE]
    )
    assert min(
        row["sampling_probability"]
        for row in rows_by_variant[CALIBRATION_DISABLED]
    ) >= (1.0 / 3.0) - 1e-12


def test_readmes_document_production_and_smoke_commands():
    repository = Path(__file__).resolve().parents[1]
    experiment_readme = (
        repository
        / "experiments"
        / "leduc_poker"
        / "ucv_exact_tabular_validation"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (repository / "README.md").read_text(encoding="utf-8")
    for readme in (experiment_readme, root_readme):
        assert "Experiment 20" in readme
        assert "ucv_exact_tabular_validation.run --smoke" in readme
        assert "ucv_exact_tabular_validation.run" in readme
        assert "n2-standard-8 172800 8000 32000 100" in readme
    assert "Experiment 19: frozen four-algorithm held-out benchmark" in root_readme
    assert "run_four_algorithm_heldout_benchmark.sh run" in root_readme
    assert "roles/batch.jobsEditor" in root_readme
    assert (
        root_readme.index("## Experiment 18:")
        < root_readme.index("## Experiment 19:")
        < root_readme.index("## Experiment 20:")
        < root_readme.index("## Add an architecture experiment")
    )
