"""Contracts for Experiment 16 temporal checkpoint head-to-head analysis."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")
torch = pytest.importorskip("torch")

from escher_poker.policy_snapshots import (  # noqa: E402
    LoadedESCHERPolicy,
    load_pickle,
    policy_snapshot_path,
    save_torch_policy_snapshot,
)
from experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.config import (  # noqa: E402
    CANDIDATE_CONFIG,
    CHECKPOINT_NODE_THRESHOLDS,
    CHECKPOINT_SCHEDULE,
    DEFAULT_SEEDS,
    RECOMMENDED_BATCH_TIMEOUT_MINUTES,
    validate_config,
)
from experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run import (  # noqa: E402
    _checkpoint_thresholds,
    _parser,
)
from experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.statistics import (  # noqa: E402
    build_inference_tables,
    exact_one_sided_sign_flip_p,
)
from experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.config import (  # noqa: E402
    CANDIDATE_CONFIG as EXPERIMENT_7_CANDIDATE_CONFIG,
)
from vr_deep_cfr.solver import MLP  # noqa: E402


def test_defaults_reproduce_experiment_7_with_experiment_27_inference_power():
    assert CANDIDATE_CONFIG == EXPERIMENT_7_CANDIDATE_CONFIG
    assert DEFAULT_SEEDS == (1234, 2025, 31415, 27182, 16180)
    assert CHECKPOINT_SCHEDULE == (1, 2, 3, 4, 5)
    assert CHECKPOINT_NODE_THRESHOLDS == (
        3_000_000,
        6_000_000,
        9_000_000,
        12_000_000,
        15_000_000,
    )
    assert CANDIDATE_CONFIG["evaluation_frequency"] == 1
    assert CANDIDATE_CONFIG["max_num_iterations"] == 120
    assert RECOMMENDED_BATCH_TIMEOUT_MINUTES == 5_760


def test_node_target_override_preserves_five_evenly_spaced_stages():
    args = _parser().parse_args(["--target-nodes", "50"])
    assert _checkpoint_thresholds(args) == (10, 20, 30, 40, 50)


def test_checkpoint_contract_rejects_training_that_cannot_snapshot_fitted_policies():
    invalid = deepcopy(CANDIDATE_CONFIG)
    invalid["evaluation_frequency"] = 2
    with pytest.raises(ValueError, match="policy fit after every"):
        validate_config(invalid)


def test_exact_seed_level_inference_matches_experiment_27_protocol():
    assert exact_one_sided_sign_flip_p([1, 1, 1, 1, 1]) == pytest.approx(1 / 32)
    schedule = (1, 2, 3, 4, 5)
    rows = []
    for seed, multiplier in enumerate((1.0, 1.1, 1.2, 1.3, 1.4), start=1):
        for later in schedule:
            for earlier in schedule:
                if later <= earlier:
                    continue
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint_a": later,
                        "checkpoint_b": earlier,
                        "A_EV_seat_averaged": 0.001 * (later - earlier) * multiplier,
                    }
                )

    seed_rows, summary_rows, pair_rows = build_inference_tables(rows, schedule)

    assert len(seed_rows) == 5
    assert all(row["num_later_vs_earlier_pairs"] == 10 for row in seed_rows)
    assert len(summary_rows) == 3
    assert summary_rows[0]["n_seeds"] == 5
    assert summary_rows[0]["exact_one_sided_sign_flip_p"] == pytest.approx(1 / 32)
    assert len(pair_rows) == 10
    assert all("holm_adjusted_p" in row for row in pair_rows)


def _first_player_decision(game):
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    return state


def test_pytorch_average_policy_snapshot_is_playable_and_exact(tmp_path):
    game = pyspiel.load_game("leduc_poker")
    input_size = game.information_state_tensor_size()
    num_actions = game.num_distinct_actions()
    model = MLP(input_size, [8, 8], num_actions)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(0.01 * index)

    solver = SimpleNamespace(
        ave_policy_trainer=SimpleNamespace(model=model),
        network_layers=[8, 8],
        infostate_size=input_size,
        action_size=num_actions,
        num_iteration=7,
        nodes_touched=123,
    )
    path = policy_snapshot_path(tmp_path, 1234, 1, "checkpointed")
    save_torch_policy_snapshot(
        solver,
        path,
        seed=1234,
        iteration=1,
        arm="checkpointed",
        config={"game_name": "leduc_poker"},
        stage_label="test",
        checkpoint_target_nodes=100,
    )

    snapshot = load_pickle(path)
    loaded = LoadedESCHERPolicy(game, path)
    state = _first_player_decision(game)
    player = state.current_player()
    actual = loaded.action_probabilities(state)

    info_state = torch.as_tensor(
        state.information_state_tensor(player), dtype=torch.float32
    )
    legal_mask = torch.as_tensor(
        state.legal_actions_mask(player), dtype=torch.float32
    )
    with torch.no_grad():
        logits = model(info_state)
        expected = torch.softmax(
            torch.where(legal_mask == 1, logits, torch.full_like(logits, -1e21)),
            dim=-1,
        ).numpy()

    assert snapshot["framework"] == "pytorch"
    assert snapshot["checkpoint_target_nodes"] == 100
    assert sum(actual.values()) == pytest.approx(1.0)
    assert set(actual) == set(state.legal_actions())
    assert np.asarray([actual[action] for action in actual]) == pytest.approx(
        np.asarray([expected[action] for action in actual])
    )


def test_readmes_document_local_and_single_batch_smoke_commands():
    root = Path(__file__).parents[1]
    package_readme = (
        root
        / "experiments"
        / "leduc_poker"
        / "unbiased_escher_temporal_checkpoint_head_to_head"
        / "README.md"
    ).read_text(encoding="utf-8")
    root_readme = (root / "README.md").read_text(encoding="utf-8")

    for text in (package_readme, root_readme):
        assert "--target-nodes 50" in text
        assert "345600" in text
        assert "Experiment 16" in text
