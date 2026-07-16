"""Shared utilities for constrained ESCHER hyperparameter searches."""

from __future__ import annotations

from collections import defaultdict
import json
import random
import time
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, expected_game_score

from .constants import DEFAULT_FINAL_WINDOW, LEDUC_GAME_VALUE_PLAYER_0
from .experiment_utils import (
    cleanup_tensorflow_memory,
    final_window_mean,
    first_nodes_to_threshold,
    first_time_to_threshold,
    json_safe,
    make_escher_solver,
    safe_stats,
)
from .seeding import set_seed_tf

HP_OUTPUT_KEYS = [
    "learning_rate",
    "num_iterations",
    "num_traversals",
    "num_val_fn_traversals",
    "policy_network_layers",
    "regret_network_layers",
    "value_network_layers",
    "batch_size_regret",
    "batch_size_value",
    "batch_size_average_policy",
    "memory_capacity",
    "policy_network_train_steps",
    "regret_network_train_steps",
    "value_network_train_steps",
    "reinitialize_regret_networks",
    "reinitialize_value_network",
    "expl",
    "val_expl",
    "importance_sampling",
    "importance_sampling_threshold",
    "clear_value_buffer",
    "val_bootstrap",
    "use_balanced_probs",
    "val_op_prob",
    "all_actions",
]

AGGREGATE_METRICS = [
    "final_exploitability",
    "best_exploitability",
    "final_window_mean_exploitability",
    "exploitability_auc",
    "final_policy_value",
    "final_policy_value_error",
    "final_nodes_touched",
    "final_wall_clock_seconds",
    "nodes_to_exploitability_threshold",
    "seconds_to_exploitability_threshold",
    "final_policy_loss",
    "final_value_loss",
    "final_value_test_loss",
    "final_regret_loss_player_0",
    "final_regret_loss_player_1",
]


def _hashable(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _hashable(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def config_signature(config: Dict[str, Any]) -> tuple:
    """Return a compact signature for deduplicating candidate configurations."""
    keys = [
        "learning_rate",
        "num_traversals",
        "num_val_fn_traversals",
        "policy_network_layers",
        "regret_network_layers",
        "value_network_layers",
        "batch_size_regret",
        "batch_size_value",
        "batch_size_average_policy",
        "memory_capacity",
        "policy_network_train_steps",
        "regret_network_train_steps",
        "value_network_train_steps",
        "reinitialize_regret_networks",
        "reinitialize_value_network",
        "expl",
        "val_expl",
        "importance_sampling",
        "importance_sampling_threshold",
        "clear_value_buffer",
        "val_bootstrap",
        "use_balanced_probs",
        "val_op_prob",
        "all_actions",
    ]
    return tuple((key, _hashable(config[key])) for key in keys if key in config)


def with_stage_overrides(
    config: Dict[str, Any],
    stage_name: str,
    num_iterations: int,
    evaluation_interval: int,
) -> Dict[str, Any]:
    staged = dict(config)
    staged["stage"] = stage_name
    staged["num_iterations"] = int(num_iterations)
    staged["check_exploitability_every"] = int(evaluation_interval)
    return staged


def sample_candidate_configs(
    base_config: Dict[str, Any],
    search_space: Dict[str, Sequence[Any]],
    n_candidates: int,
    rng_seed: int,
    variant_id_prefix: str = "candidate",
) -> List[Dict[str, Any]]:
    """Sample unique random candidate configs around ``base_config``."""
    rng = random.Random(rng_seed)
    candidates = []
    seen = {config_signature(base_config)}
    attempts = 0
    while len(candidates) < n_candidates and attempts < 500:
        attempts += 1
        candidate = dict(base_config)
        for key, values in search_space.items():
            candidate[key] = rng.choice(list(values))
        sig = config_signature(candidate)
        if sig in seen:
            continue
        seen.add(sig)
        candidate["variant_id"] = f"{variant_id_prefix}_{len(candidates) + 1:02d}"
        candidates.append(candidate)
    if len(candidates) < n_candidates:
        raise RuntimeError("Could not generate enough unique random candidates.")
    return candidates


def config_subset(config: Dict[str, Any]) -> Dict[str, Any]:
    """Compact hyperparameter subset for row-level output."""
    return {key: json_safe(config[key]) for key in HP_OUTPUT_KEYS if key in config}


def _safe_last(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values[-1]) if values.size else np.nan


def _safe_min(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else np.nan


def _mean_finite(values: Iterable[float]) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else np.nan


def run_single_hyperparameter_seed(
    seed: int,
    config: Dict[str, Any],
    stage_name: str,
) -> Dict[str, Any]:
    """Train one ESCHER configuration on one seed and return curves plus summary."""
    set_seed_tf(seed)
    game = pyspiel.load_game(config["game_name"])
    solver = None
    start = time.time()
    error = ""
    try:
        solver = make_escher_solver(game, config)
        _regret_losses, policy_loss, convs, nodes_touched, avg_policy_values, diagnostics = solver.solve()
        elapsed = time.time() - start
        exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
        nodes_touched = np.asarray(nodes_touched, dtype=np.float64)
        avg_policy_values = np.asarray(avg_policy_values, dtype=np.float64)
        diagnostics = {key: np.asarray(value) for key, value in diagnostics.items()}
        iterations = diagnostics.get("iteration", np.asarray([], dtype=int)).astype(int)
        wall_clock = diagnostics.get("wall_clock_seconds", np.asarray([], dtype=float)).astype(float)

        final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
        final_nash_conv = exploitability.nash_conv(game, final_policy)
        final_policy_value = expected_game_score.policy_value(
            game.new_initial_state(),
            [final_policy] * game.num_players(),
        )[0]
    except Exception as exc:  # pragma: no cover - operational robustness
        elapsed = time.time() - start
        error = str(exc)
        policy_loss = np.nan
        exploitability_curve = np.asarray([np.nan], dtype=np.float64)
        nodes_touched = np.asarray([np.nan], dtype=np.float64)
        avg_policy_values = np.asarray([np.nan], dtype=np.float64)
        diagnostics = {}
        iterations = np.asarray([int(config["num_iterations"])], dtype=int)
        wall_clock = np.asarray([elapsed], dtype=np.float64)
        final_nash_conv = np.nan
        final_policy_value = np.nan
    finally:
        if solver is not None:
            del solver
        cleanup_tensorflow_memory()

    value_error = np.abs(avg_policy_values - LEDUC_GAME_VALUE_PLAYER_0)
    final_exploitability = _safe_last(exploitability_curve)
    summary = {
        "stage": stage_name,
        "variant_id": config["variant_id"],
        "seed": int(seed),
        "status": "failed" if error else "completed",
        "error": error,
        "final_exploitability": final_exploitability,
        "best_exploitability": _safe_min(exploitability_curve),
        "final_window_mean_exploitability": final_window_mean(
            exploitability_curve,
            DEFAULT_FINAL_WINDOW,
        ),
        "exploitability_auc": _mean_finite(exploitability_curve),
        "final_policy_value": float(final_policy_value) if np.isfinite(final_policy_value) else np.nan,
        "final_policy_value_error": (
            float(abs(final_policy_value - LEDUC_GAME_VALUE_PLAYER_0))
            if np.isfinite(final_policy_value)
            else np.nan
        ),
        "best_policy_value_error": _safe_min(value_error),
        "final_nodes_touched": _safe_last(nodes_touched),
        "final_wall_clock_seconds": _safe_last(wall_clock),
        "nodes_to_exploitability_threshold": first_nodes_to_threshold(
            nodes_touched,
            exploitability_curve,
            config["exploitability_threshold"],
        ),
        "seconds_to_exploitability_threshold": first_time_to_threshold(
            wall_clock,
            exploitability_curve,
            config["exploitability_threshold"],
        ),
        "final_policy_loss": float(np.asarray(policy_loss)) if policy_loss is not None else np.nan,
        "final_value_loss": _safe_last(diagnostics.get("value_loss", [])),
        "final_value_test_loss": _safe_last(diagnostics.get("value_test_loss", [])),
        "final_regret_loss_player_0": _safe_last(diagnostics.get("regret_loss_player_0", [])),
        "final_regret_loss_player_1": _safe_last(diagnostics.get("regret_loss_player_1", [])),
        "final_nash_conv_recomputed": float(final_nash_conv) if np.isfinite(final_nash_conv) else np.nan,
    }
    summary.update({
        f"hp_{key}": json.dumps(value) if isinstance(value, list) else value
        for key, value in config_subset(config).items()
    })

    return {
        "stage": stage_name,
        "variant_id": config["variant_id"],
        "seed": int(seed),
        "config": dict(config),
        "iterations": iterations,
        "nodes_touched": nodes_touched,
        "wall_clock_seconds": wall_clock,
        "exploitability": exploitability_curve,
        "average_policy_value": avg_policy_values,
        "policy_value_error": value_error,
        "diagnostics": diagnostics,
        "summary": summary,
    }


def run_hyperparameter_stage(
    configs: List[Dict[str, Any]],
    seeds: List[int],
    stage_name: str,
    num_iterations: int,
    evaluation_interval: int,
) -> List[Dict[str, Any]]:
    """Run every variant/seed pair for a search stage."""
    stage_results = []
    for config in configs:
        stage_config = with_stage_overrides(config, stage_name, num_iterations, evaluation_interval)
        for seed in seeds:
            stage_results.append(run_single_hyperparameter_seed(seed, stage_config, stage_name))
    return stage_results


def aggregate_summaries(
    summary_rows: List[Dict[str, Any]],
    group_key: str = "variant_id",
) -> List[Dict[str, Any]]:
    """Aggregate scalar summary rows by variant or another grouping key."""
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row[group_key]].append(row)

    aggregate_rows = []
    for group_value, rows in sorted(grouped.items()):
        aggregate = {group_key: group_value}
        for metric in AGGREGATE_METRICS:
            stats = safe_stats([row.get(metric, np.nan) for row in rows])
            aggregate[f"{metric}_mean"] = stats["mean"]
            aggregate[f"{metric}_std"] = stats["std"]
            aggregate[f"{metric}_se"] = stats["se"]
            aggregate[f"{metric}_n_finite"] = stats["n_finite"]
        aggregate["n_runs"] = len(rows)
        aggregate["n_completed"] = sum(row.get("status") == "completed" for row in rows)
        first = rows[0]
        for key, value in first.items():
            if key.startswith("hp_"):
                aggregate[key] = value
        aggregate_rows.append(aggregate)
    return aggregate_rows


def select_confirmation_variants(
    screening_aggregate_rows: List[Dict[str, Any]],
    baseline_variant_id: str,
    top_k: int,
) -> List[str]:
    """Rank screening candidates by final-window exploitability, then AUC."""
    candidates = [
        row for row in screening_aggregate_rows
        if row["variant_id"] != baseline_variant_id
    ]
    ranked = sorted(
        candidates,
        key=lambda row: (
            np.inf
            if not np.isfinite(row["final_window_mean_exploitability_mean"])
            else row["final_window_mean_exploitability_mean"],
            np.inf
            if not np.isfinite(row["exploitability_auc_mean"])
            else row["exploitability_auc_mean"],
        ),
    )
    return [row["variant_id"] for row in ranked[:top_k]]


def paired_differences_vs_baseline(
    confirmation_summary_rows: List[Dict[str, Any]],
    baseline_variant_id: str,
) -> List[Dict[str, Any]]:
    """Compute paired confirmation deltas relative to the baseline."""
    by_variant_seed = {
        (row["variant_id"], row["seed"]): row
        for row in confirmation_summary_rows
    }
    paired_rows = []
    for row in confirmation_summary_rows:
        variant_id = row["variant_id"]
        if variant_id == baseline_variant_id:
            continue
        baseline_row = by_variant_seed.get((baseline_variant_id, row["seed"]))
        if baseline_row is None:
            continue
        paired_rows.append({
            "variant_id": variant_id,
            "seed": int(row["seed"]),
            "delta_final_exploitability": (
                row["final_exploitability"] - baseline_row["final_exploitability"]
            ),
            "delta_best_exploitability": (
                row["best_exploitability"] - baseline_row["best_exploitability"]
            ),
            "delta_final_window_mean_exploitability": (
                row["final_window_mean_exploitability"]
                - baseline_row["final_window_mean_exploitability"]
            ),
            "delta_exploitability_auc": (
                row["exploitability_auc"] - baseline_row["exploitability_auc"]
            ),
            "delta_final_policy_value": (
                row["final_policy_value"] - baseline_row["final_policy_value"]
            ),
            "delta_policy_value_error": (
                row["final_policy_value_error"] - baseline_row["final_policy_value_error"]
            ),
            "delta_wall_clock_seconds": (
                row["final_wall_clock_seconds"] - baseline_row["final_wall_clock_seconds"]
            ),
            "delta_nodes_touched": (
                row["final_nodes_touched"] - baseline_row["final_nodes_touched"]
            ),
        })
    return paired_rows
