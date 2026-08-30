"""Validate endpoint archives and run the exact four-policy league."""

from __future__ import annotations

from collections import defaultdict
import itertools
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyspiel  # noqa: E402
from open_spiel.python import policy  # noqa: E402
from open_spiel.python.algorithms import expected_game_score, exploitability  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.policies import (  # noqa: E402
    load_policy,
    validate_policy_probabilities,
)
from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.statistics import (  # noqa: E402
    holm_adjust,
    summary,
)

from .common import read_json, sha256, write_csv, write_json  # noqa: E402
from .config import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_ORDER,
    ENDPOINT_ORDER,
    GAME_NAME,
    TARGET_ACTIVE_SECONDS,
    TARGET_NODES,
)


EQUIVALENCE_EPSILON = 1e-3


def _exact_ev(game, policy_a, policy_b) -> dict:
    as_player_0 = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [policy_a, policy_b]
        )[0]
    )
    as_player_1 = float(
        expected_game_score.policy_value(
            game.new_initial_state(), [policy_b, policy_a]
        )[1]
    )
    return {
        "algorithm_a_ev_as_player_0": as_player_0,
        "algorithm_a_ev_as_player_1": as_player_1,
        "algorithm_a_seat_averaged_ev": 0.5 * (as_player_0 + as_player_1),
    }


def _plot_matrix(matrix: np.ndarray, output_path: Path, endpoint_id: str) -> None:
    labels = [ALGORITHMS[item]["algorithm_label"] for item in ALGORITHM_ORDER]
    limit = max(float(np.max(np.abs(matrix))), 1e-9)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Algorithm")
    set_chart_title(ax, f"Held-out exact two-seat EV: {endpoint_id}")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="EV for row algorithm")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_exploitability(rows: Sequence[Mapping], output_path: Path, endpoint_id: str) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["algorithm_id"]].append(float(row["exploitability"]))
    means = [float(np.mean(grouped[item])) for item in ALGORITHM_ORDER]
    sems = [
        float(np.std(grouped[item], ddof=1) / np.sqrt(len(grouped[item])))
        if len(grouped[item]) > 1
        else 0.0
        for item in ALGORITHM_ORDER
    ]
    labels = [ALGORITHMS[item]["algorithm_label"] for item in ALGORITHM_ORDER]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(np.arange(len(labels)), means, yerr=sems, capsize=4)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, f"Held-out endpoint exploitability: {endpoint_id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_endpoint_analysis(
    *,
    snapshot_inventory: Sequence[Mapping],
    seeds: Sequence[int],
    endpoint_id: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed = {
        (str(row["algorithm_id"]), int(row["seed"])): dict(row)
        for row in snapshot_inventory
    }
    expected = {
        (algorithm_id, int(seed))
        for algorithm_id in ALGORITHM_ORDER
        for seed in seeds
    }
    if set(indexed) != expected:
        raise ValueError(
            f"{endpoint_id} inventory mismatch; missing={sorted(expected - set(indexed))}, "
            f"extra={sorted(set(indexed) - expected)}"
        )

    game = pyspiel.load_game(GAME_NAME)
    policies = {}
    loaded_rows = []
    for algorithm_id in ALGORITHM_ORDER:
        for seed in seeds:
            row = indexed[(algorithm_id, int(seed))]
            loaded = load_policy(game, algorithm_id, row["path"])
            validate_policy_probabilities(game, loaded)
            tabular = policy.tabular_policy_from_callable(
                game, loaded.action_probabilities
            )
            policies[(algorithm_id, int(seed))] = tabular
            loaded_rows.append(
                {
                    **row,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "probability_validation": "passed_exhaustive_tabular_conversion",
                }
            )

    metric_rows = []
    for (algorithm_id, seed), tabular in policies.items():
        nash_conv = float(exploitability.nash_conv(game, tabular))
        self_play = float(
            expected_game_score.policy_value(
                game.new_initial_state(), [tabular, tabular]
            )[0]
        )
        metric_rows.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                "seed": seed,
                "nash_conv": nash_conv,
                "exploitability": nash_conv / 2.0,
                "self_play_value_player_0": self_play,
            }
        )

    same_seed_rows = []
    for seed in seeds:
        for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
            values = _exact_ev(
                game,
                policies[(algorithm_a, int(seed))],
                policies[(algorithm_b, int(seed))],
            )
            effect = values["algorithm_a_seat_averaged_ev"]
            same_seed_rows.append(
                {
                    "algorithm_a": algorithm_a,
                    "algorithm_a_label": ALGORITHMS[algorithm_a]["algorithm_label"],
                    "algorithm_b": algorithm_b,
                    "algorithm_b_label": ALGORITHMS[algorithm_b]["algorithm_label"],
                    "seed": int(seed),
                    **values,
                    "classification_for_a": (
                        "win"
                        if effect > EQUIVALENCE_EPSILON
                        else "loss"
                        if effect < -EQUIVALENCE_EPSILON
                        else "practical_tie"
                    ),
                }
            )

    inference_rows = []
    for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
        effects = [
            row["algorithm_a_seat_averaged_ev"]
            for row in same_seed_rows
            if row["algorithm_a"] == algorithm_a and row["algorithm_b"] == algorithm_b
        ]
        inference_rows.append(
            {
                "algorithm_a": algorithm_a,
                "algorithm_a_label": ALGORITHMS[algorithm_a]["algorithm_label"],
                "algorithm_b": algorithm_b,
                "algorithm_b_label": ALGORITHMS[algorithm_b]["algorithm_label"],
                **summary(effects),
            }
        )
    holm_adjust(
        inference_rows,
        "two_sided_exact_sign_flip_p",
        "holm_adjusted_two_sided_p",
    )

    cross_seed_rows = []
    for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
        for seed_a in seeds:
            for seed_b in seeds:
                cross_seed_rows.append(
                    {
                        "algorithm_a": algorithm_a,
                        "algorithm_a_label": ALGORITHMS[algorithm_a]["algorithm_label"],
                        "seed_a": int(seed_a),
                        "algorithm_b": algorithm_b,
                        "algorithm_b_label": ALGORITHMS[algorithm_b]["algorithm_label"],
                        "seed_b": int(seed_b),
                        **_exact_ev(
                            game,
                            policies[(algorithm_a, int(seed_a))],
                            policies[(algorithm_b, int(seed_b))],
                        ),
                    }
                )

    strength_by_seed = []
    for algorithm_id in ALGORITHM_ORDER:
        for seed in seeds:
            effects = []
            for row in same_seed_rows:
                if int(row["seed"]) != int(seed):
                    continue
                if row["algorithm_a"] == algorithm_id:
                    effects.append(row["algorithm_a_seat_averaged_ev"])
                elif row["algorithm_b"] == algorithm_id:
                    effects.append(-row["algorithm_a_seat_averaged_ev"])
            strength_by_seed.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "seed": int(seed),
                    "mean_ev_vs_other_algorithms": float(np.mean(effects)),
                    "num_opponents": len(effects),
                }
            )
    strength_summary = []
    for algorithm_id in ALGORITHM_ORDER:
        effects = [
            row["mean_ev_vs_other_algorithms"]
            for row in strength_by_seed
            if row["algorithm_id"] == algorithm_id
        ]
        strength_summary.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                **summary(effects),
            }
        )
    strength_summary.sort(key=lambda row: row["mean_ev"], reverse=True)
    for rank, row in enumerate(strength_summary, start=1):
        row["rank_by_mean_exact_ev"] = rank

    matrix = np.zeros((len(ALGORITHM_ORDER), len(ALGORITHM_ORDER)), dtype=float)
    for row in inference_rows:
        a = ALGORITHM_ORDER.index(row["algorithm_a"])
        b = ALGORITHM_ORDER.index(row["algorithm_b"])
        matrix[a, b] = float(row["mean_ev"])
        matrix[b, a] = -float(row["mean_ev"])
    matrix_rows = [
        {
            "algorithm_id": algorithm_id,
            **{
                opponent: float(matrix[index, opponent_index])
                for opponent_index, opponent in enumerate(ALGORITHM_ORDER)
            },
        }
        for index, algorithm_id in enumerate(ALGORITHM_ORDER)
    ]

    write_csv(output_dir / "loaded_policy_inventory.csv", loaded_rows)
    write_csv(output_dir / "endpoint_policy_metrics.csv", metric_rows)
    write_csv(output_dir / "head_to_head_same_seed_pairwise.csv", same_seed_rows)
    write_csv(output_dir / "head_to_head_pairwise_inference.csv", inference_rows)
    write_csv(output_dir / "head_to_head_cross_seed_league.csv", cross_seed_rows)
    write_csv(output_dir / "algorithm_strength_by_seed.csv", strength_by_seed)
    write_csv(output_dir / "algorithm_strength_summary.csv", strength_summary)
    write_csv(output_dir / "head_to_head_mean_ev_matrix.csv", matrix_rows)
    summary_path = output_dir / "aggregate_summary.json"
    write_json(
        summary_path,
        {
            "endpoint_id": endpoint_id,
            "evaluation": "exact OpenSpiel expected value in both seats",
            "sampled_games": 0,
            "training_seeds_are_inferential_unit": True,
            "num_algorithms": len(ALGORITHM_ORDER),
            "num_training_seeds": len(seeds),
            "num_same_seed_pairwise_effects": len(same_seed_rows),
            "num_cross_seed_exact_matchups": len(cross_seed_rows),
            "num_exact_seat_assignments": 2 * (len(same_seed_rows) + len(cross_seed_rows)),
            "equivalence_epsilon": EQUIVALENCE_EPSILON,
            "pairwise_inference": inference_rows,
            "algorithm_strength_summary": strength_summary,
            "multiplicity_note": (
                "Holm correction covers six pairwise two-sided tests. With eight seeds, "
                "the minimum exact two-sided sign-flip p-value is 0.0078125."
            ),
        },
    )
    _plot_matrix(matrix, output_dir / "head_to_head_mean_ev_heatmap.png", endpoint_id)
    _plot_exploitability(
        metric_rows, output_dir / "endpoint_exploitability.png", endpoint_id
    )
    return {
        "endpoint_id": endpoint_id,
        "summary": str(summary_path),
        "num_same_seed_pairwise_effects": len(same_seed_rows),
        "num_cross_seed_exact_matchups": len(cross_seed_rows),
    }


def aggregate_workers(
    *, workers_root: Path, seeds: Sequence[int], output_dir: Path, smoke: bool = False
) -> dict:
    workers_root = Path(workers_root).resolve()
    output_dir = Path(output_dir).resolve()
    expected = {
        (algorithm_id, int(seed))
        for algorithm_id in ALGORITHM_ORDER
        for seed in seeds
    }
    worker_results = {}
    for result_path in workers_root.rglob("worker_result.json"):
        result = read_json(result_path)
        key = (str(result["algorithm_id"]), int(result["seed"]))
        if key in worker_results:
            raise ValueError(f"Duplicate worker result for {key}")
        if result.get("status") != "complete":
            raise ValueError(f"Incomplete worker result: {result_path}")
        expected_nodes = 1 if smoke else TARGET_NODES
        expected_seconds = 0.0 if smoke else TARGET_ACTIVE_SECONDS
        if (
            bool(result.get("smoke")) != bool(smoke)
            or int(result.get("target_nodes", -1)) != expected_nodes
            or float(result.get("target_active_seconds", -1)) != expected_seconds
        ):
            raise ValueError(f"Worker contract mismatch: {result_path}")
        worker_results[key] = (result_path, result)
    if set(worker_results) != expected:
        raise ValueError(
            f"Worker set mismatch; missing={sorted(expected - set(worker_results))}, "
            f"extra={sorted(set(worker_results) - expected)}"
        )

    endpoint_inventory = {endpoint_id: [] for endpoint_id in ENDPOINT_ORDER}
    worker_manifest = []
    for key in sorted(worker_results):
        result_path, result = worker_results[key]
        records = {row["endpoint_id"]: row for row in result["snapshots"]}
        if set(records) != set(ENDPOINT_ORDER):
            raise ValueError(f"Endpoint mismatch in {result_path}")
        for endpoint_id in ENDPOINT_ORDER:
            record = dict(records[endpoint_id])
            if record.get("repository_commit") != result.get("repository_commit"):
                raise ValueError(f"Snapshot/worker commit mismatch in {result_path}")
            snapshot_path = result_path.parent / record["relative_path"]
            if not snapshot_path.is_file():
                raise FileNotFoundError(snapshot_path)
            observed_hash = sha256(snapshot_path)
            if observed_hash != record["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {snapshot_path}")
            endpoint_inventory[endpoint_id].append(
                {**record, "path": str(snapshot_path.resolve())}
            )
        worker_manifest.append(
            {
                "algorithm_id": key[0],
                "seed": key[1],
                "result_path": str(result_path),
                "repository_commit": result["repository_commit"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "worker_manifest.csv", worker_manifest)
    endpoint_results = []
    for endpoint_id in ENDPOINT_ORDER:
        endpoint_dir = output_dir / endpoint_id
        write_csv(endpoint_dir / "snapshot_inventory.csv", endpoint_inventory[endpoint_id])
        endpoint_results.append(
            run_endpoint_analysis(
                snapshot_inventory=endpoint_inventory[endpoint_id],
                seeds=seeds,
                endpoint_id=endpoint_id,
                output_dir=endpoint_dir,
            )
        )
    result = {
        "status": "complete",
        "num_workers": len(worker_results),
        "num_snapshots": sum(len(rows) for rows in endpoint_inventory.values()),
        "seeds": [int(seed) for seed in seeds],
        "endpoint_results": endpoint_results,
    }
    write_json(output_dir / "aggregate_manifest.json", result)
    return result


__all__ = ["aggregate_workers", "run_endpoint_analysis"]
