"""Contracts for Experiment 17 six-algorithm head-to-head analysis."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.config import (  # noqa: E402
    ALGORITHM_ORDER,
    DEFAULT_SEEDS,
    EXPERIMENT_ID,
    MEASURED_SEQUENTIAL_TRAINING_HOURS,
    RECOMMENDED_BATCH_TIMEOUT_SECONDS,
    TARGET_NODES,
    VR_ALGORITHMS,
    VR_CONFIG,
    validate_contract,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.policies import (  # noqa: E402
    TorchStateDictPolicy,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    exact_sign_flip_p,
    holm_adjust,
)
from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (  # noqa: E402
    VR_CONFIG as EXPERIMENT_7_VR_CONFIG,
)
from vr_deep_cfr.policy_snapshots import (  # noqa: E402
    LoadedVRPolicy,
    save_policy_snapshot,
)
from vr_deep_cfr.solver import MLP  # noqa: E402


def _first_decision(game):
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    return state


def test_production_contract_has_six_algorithms_five_common_seeds_and_15m_nodes():
    assert EXPERIMENT_ID == 17
    assert len(ALGORITHM_ORDER) == 6
    assert len(VR_ALGORITHMS) == 2
    assert DEFAULT_SEEDS == (1234, 2025, 31415, 27182, 16180)
    assert TARGET_NODES == 15_000_000
    assert VR_CONFIG == EXPERIMENT_7_VR_CONFIG
    validate_contract(DEFAULT_SEEDS, TARGET_NODES, VR_CONFIG)
    with pytest.raises(ValueError, match="exactly five"):
        validate_contract(DEFAULT_SEEDS[:1], TARGET_NODES, VR_CONFIG)
    validate_contract(
        DEFAULT_SEEDS[:1],
        50,
        VR_CONFIG,
        require_five_seeds=False,
    )


def test_runtime_estimate_uses_completed_experiment_7_measurements():
    assert MEASURED_SEQUENTIAL_TRAINING_HOURS == pytest.approx(65.294406153238)
    assert RECOMMENDED_BATCH_TIMEOUT_SECONDS == 345_600


def test_five_seed_exact_two_sided_test_cannot_reach_point_zero_five():
    values = [1.0] * 5
    assert exact_sign_flip_p(values, two_sided=True) == pytest.approx(2 / 32)
    assert exact_sign_flip_p(values, two_sided=False) == pytest.approx(1 / 32)


def test_holm_adjustment_is_monotone_and_bounded():
    rows = [{"p": 0.01}, {"p": 0.03}, {"p": 0.2}]
    holm_adjust(rows, "p", "adjusted")
    assert [row["adjusted"] for row in rows] == pytest.approx([0.03, 0.06, 0.2])


def test_deep_cfr_and_dream_snapshot_adapters_are_playable(tmp_path):
    game = pyspiel.load_game("leduc_poker")
    input_size = game.information_state_tensor_size()
    output_size = game.num_distinct_actions()

    deep_path = tmp_path / "deep.pt"
    deep_state = {
        "model.0._weight": torch.full((4, input_size), 0.01),
        "model.0._bias": torch.zeros(4),
        "model.1._weight": torch.full((output_size, 4), 0.02),
        "model.1._bias": torch.arange(output_size, dtype=torch.float32),
    }
    torch.save(
        {
            "type": "deep_cfr_policy_snapshot",
            "policy_state_dict": deep_state,
        },
        deep_path,
    )

    dream_path = tmp_path / "dream.pt"
    dream_state = {
        "net.0.layer.weight": torch.full((4, input_size), 0.01),
        "net.0.layer.bias": torch.zeros(4),
        "net.2.layer.weight": torch.full((output_size, 4), 0.02),
        "net.2.layer.bias": torch.arange(output_size, dtype=torch.float32),
    }
    torch.save(
        {
            "kind": "dream_policy_snapshot",
            "policy_network_state_dict": dream_state,
        },
        dream_path,
    )

    state = _first_decision(game)
    for algorithm_id, path in (("deep_cfr", deep_path), ("dream", dream_path)):
        probabilities = TorchStateDictPolicy(
            game, path, algorithm_id
        ).action_probabilities(state)
        assert set(probabilities) == set(state.legal_actions())
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert np.all(np.asarray(list(probabilities.values())) >= 0)


def test_vr_snapshot_round_trip_preserves_final_average_policy(tmp_path):
    game = pyspiel.load_game("leduc_poker")
    model = MLP(game.information_state_tensor_size(), [4, 4], game.num_distinct_actions())
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(0.01 * index)
    solver = SimpleNamespace(
        ave_policy_trainer=SimpleNamespace(model=model),
        network_layers=[4, 4],
        infostate_size=game.information_state_tensor_size(),
        action_size=game.num_distinct_actions(),
        nodes_touched=15_000_123,
        num_iteration=97,
    )
    path = tmp_path / "vr.pt"
    save_policy_snapshot(
        solver,
        path,
        algorithm_id="vr_deep_dcfr_plus",
        algorithm_label="VR-DeepDCFR+",
        seed=1234,
        config={"game_name": "leduc_poker"},
    )
    loaded = LoadedVRPolicy(game, path)
    state = _first_decision(game)
    probabilities = loaded.action_probabilities(state)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert loaded.metadata["nodes_touched"] == 15_000_123
    for name, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor, model.state_dict()[name])


def test_readmes_document_exact_evaluation_gcp_config_and_runtime():
    root = Path(__file__).parents[1]
    experiment_readme = (
        root
        / "experiments/leduc_poker/six_algorithm_final_policy_head_to_head/README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    for text in (experiment_readme, root_readme):
        assert "Experiment 17" in text
        assert "n2-standard-8 345600 8000 32000 100" in text
        assert "65.3 hours" in text
        assert "0.0625" in text
        assert "exact" in text.lower()
        assert "bash gcp/run_experiment_17.sh" in text


def test_gcp_staging_uses_versioned_accessible_source_bundle():
    root = Path(__file__).parents[1]
    fetch_script = (root / "gcp/fetch_experiment_17_snapshots.sh").read_text(
        encoding="utf-8"
    )
    wrapper = (root / "gcp/run_experiment_17.sh").read_text(encoding="utf-8")

    assert (
        "gs://clever-overview-399515-leduc-poker-dream-results/"
        "experiment-17-inputs/six-algorithm-final-policy-head-to-head-v1"
    ) in fetch_script
    assert "EXPERIMENT_17_SOURCE_BUNDLE_ROOT" in fetch_script
    assert "leduc-poker-results" not in fetch_script
    assert "leduc-poker-escher-results" not in fetch_script
    assert "fetch_experiment_17_snapshots.sh \"$SNAPSHOT_ROOT\"" in wrapper
    assert '--snapshot-root "$SNAPSHOT_ROOT"' in wrapper
