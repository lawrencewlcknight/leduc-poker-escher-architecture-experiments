from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from escher_poker.live_resolving import (
    StitchedResolvedPolicy,
    collect_public_roots,
    is_second_round_root,
    public_root_key,
    solve_public_roots,
)
from experiments.leduc_poker.tabular_cfr_live_resolving.config import (
    PRODUCTION_SEEDS,
    SEARCH_UPDATE_BUDGETS as CFR_BUDGETS,
)
from experiments.leduc_poker.ucv_live_resolving.config import (
    RESOLVER_REPLICATES,
    SEARCH_UPDATE_BUDGETS as UCV_BUDGETS,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def _exact_exploitability(game, candidate) -> float:
    tabular = policy.tabular_policy_from_callable(
        game, candidate.action_probabilities
    )
    return float(exploitability.nash_conv(game, tabular) / 2.0)


def test_public_roots_retain_all_private_histories_and_exclude_private_keying():
    game = pyspiel.load_game("leduc_poker")
    roots = collect_public_roots(game, policy.UniformRandomPolicy(game))
    assert len(roots) == 30
    assert sum(len(rows) for rows in roots.values()) == 600
    for key, histories in roots.items():
        assert len(histories) > 1
        assert abs(sum(row.probability for row in histories) - 1.0) < 1e-12
        assert all(is_second_round_root(row.state) for row in histories)
        assert all(public_root_key(row.state) == key for row in histories)


def test_cfr_plus_resolver_produces_playable_improving_uniform_continuation():
    game = pyspiel.load_game("leduc_poker")
    blueprint = policy.UniformRandomPolicy(game)
    roots = collect_public_roots(game, blueprint)
    table, diagnostics = solve_public_roots(
        roots, resolver_kind="cfr_plus", target_updates=4, seed=31
    )
    resolved = StitchedResolvedPolicy(game, blueprint, table)
    assert len(diagnostics) == len(roots)
    assert _exact_exploitability(game, resolved) < _exact_exploitability(
        game, blueprint
    )


def test_ucv_and_baseline_free_resolvers_are_matched_and_playable():
    game = pyspiel.load_game("leduc_poker")
    blueprint = policy.UniformRandomPolicy(game)
    roots = collect_public_roots(game, blueprint)
    for kind, expected_control in (
        ("external_sampling", False),
        ("ucv_external_sampling", True),
    ):
        table, diagnostics = solve_public_roots(
            roots,
            resolver_kind=kind,
            target_updates=4,
            seed=32,
            exploration=0.05,
        )
        resolved = StitchedResolvedPolicy(game, blueprint, table)
        assert _exact_exploitability(game, resolved) >= 0.0
        assert all(row["control_variate_enabled"] is expected_control for row in diagnostics)
        assert sum(int(row["control_variate_samples"]) for row in diagnostics) > 0


def test_frozen_contracts_have_five_blueprints_and_overlapping_effort_ranges():
    assert len(PRODUCTION_SEEDS) == 5
    assert RESOLVER_REPLICATES == 8
    assert CFR_BUDGETS == tuple(2**index for index in range(9))
    assert UCV_BUDGETS == tuple(2**index for index in range(6, 15))


def test_cloud_builder_and_launchers_are_syntactically_valid(tmp_path):
    builder = REPOSITORY / "gcp" / "live_resolving_batch.py"
    for experiment_id in (31, 32):
        output = tmp_path / f"exp{experiment_id}.json"
        command = [
            "python3",
            str(builder),
            "--experiment-id",
            str(experiment_id),
            "--kind",
            "train",
            "--output",
            str(output),
            "--run-id",
            f"exp{experiment_id}-test",
            "--bucket-root",
            "gs://test-bucket",
            "--service-account",
            "batch@test.iam.gserviceaccount.com",
            "--repo-ref",
            "deadbeef",
            "--experiment-29-run-id",
            "exp29-test",
        ]
        if experiment_id == 32:
            command.extend(["--experiment-31-run-id", "exp31-test"])
        subprocess.run(command, cwd=REPOSITORY, check=True)
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["taskGroups"][0]["taskCount"] == 5
        script = payload["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"]
        assert "time_36h.pkl" in script

        controller_output = tmp_path / f"exp{experiment_id}-controller.json"
        controller_command = command.copy()
        controller_command[controller_command.index("train")] = "controller"
        controller_command[controller_command.index(str(output))] = str(controller_output)
        subprocess.run(controller_command, cwd=REPOSITORY, check=True)
        controller = json.loads(controller_output.read_text(encoding="utf-8"))
        controller_disk = controller["allocationPolicy"]["instances"][0]["policy"][
            "bootDisk"
        ]["sizeGb"]
        assert int(controller_disk) >= 40

    for launcher in (
        "run_live_resolving_experiment.sh",
        "run_tabular_cfr_live_resolving.sh",
        "run_ucv_live_resolving.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(REPOSITORY / "gcp" / launcher)], check=True
        )


def test_root_readme_documents_both_experiments():
    text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    assert "Experiment 31: tabular CFR+ live resolving" in text
    assert "./gcp/run_tabular_cfr_live_resolving.sh smoke-local" in text
    assert "Experiment 32: UCV-sampled live resolving" in text
    assert "./gcp/run_ucv_live_resolving.sh run" in text


def test_live_launcher_rejects_experiment_29_placeholder():
    environment = {
        "PATH": "/usr/bin:/bin",
        "PROJECT_ID": "test",
        "REGION": "europe-west1",
        "BUCKET": "gs://test",
        "SA_EMAIL": "test@example.invalid",
        "REPO_REF": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
        ).strip(),
        "EXP29_RUN_ID": "YOUR_COMPLETED_EXPERIMENT_29_RUN_ID",
        "RUN_ID": "exp31-placeholder-test",
    }
    completed = subprocess.run(
        ["bash", "gcp/run_live_resolving_experiment.sh", "31", "dry-run"],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "still a placeholder" in completed.stderr
