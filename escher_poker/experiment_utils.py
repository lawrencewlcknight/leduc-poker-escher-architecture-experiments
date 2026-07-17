"""Shared experiment utilities for ESCHER runs."""

from __future__ import annotations

import csv
from datetime import datetime
import gc
import json
from pathlib import Path
import pickle
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from scipy import stats
import tensorflow as tf

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability, expected_game_score

from .constants import DEFAULT_FINAL_WINDOW, LEDUC_GAME_VALUE_PLAYER_0
from .seeding import set_seed_tf
from .solver import ESCHERSolver


def make_escher_solver(
    game,
    config: Dict[str, Any],
    *,
    num_iterations: Optional[int] = None,
) -> ESCHERSolver:
    """Construct an ``ESCHERSolver`` from a plain experiment config dict."""
    return ESCHERSolver(
        game,
        policy_network_layers=tuple(config["policy_network_layers"]),
        regret_network_layers=tuple(config["regret_network_layers"]),
        value_network_layers=tuple(config["value_network_layers"]),
        num_iterations=int(config["num_iterations"] if num_iterations is None else num_iterations),
        num_traversals=int(config["num_traversals"]),
        num_val_fn_traversals=int(config["num_val_fn_traversals"]),
        learning_rate=float(config["learning_rate"]),
        learning_rate_schedule=str(config.get("learning_rate_schedule", "constant")),
        learning_rate_end=(
            None
            if config.get("learning_rate_end") is None
            else float(config.get("learning_rate_end"))
        ),
        learning_rate_decay_rate=float(config.get("learning_rate_decay_rate", 0.1)),
        learning_rate_warmup_iterations=int(
            config.get("learning_rate_warmup_iterations", 0)
        ),
        batch_size_regret=int(config["batch_size_regret"]),
        batch_size_value=int(config["batch_size_value"]),
        batch_size_average_policy=int(config["batch_size_average_policy"]),
        memory_capacity=int(config["memory_capacity"]),
        regret_replay_mode=str(config.get("regret_replay_mode", "reservoir")),
        regret_replay_rare_history_quota=int(
            config.get("regret_replay_rare_history_quota", 64)
        ),
        regret_replay_weight_floor=float(
            config.get("regret_replay_weight_floor", 1e-6)
        ),
        policy_network_train_steps=int(config["policy_network_train_steps"]),
        regret_network_train_steps=int(config["regret_network_train_steps"]),
        value_network_train_steps=int(config["value_network_train_steps"]),
        check_exploitability_every=int(config["check_exploitability_every"]),
        compute_exploitability=bool(config["compute_exploitability"]),
        reinitialize_regret_networks=bool(config["reinitialize_regret_networks"]),
        reinitialize_value_network=bool(config["reinitialize_value_network"]),
        save_average_policy_memories=config.get("save_average_policy_memories"),
        save_regret_memories=config.get("save_regret_memories"),
        tfrecord_compression=config.get("tfrecord_compression"),
        save_policy_weights=bool(config["save_policy_weights"]),
        train_device=config["train_device"],
        infer_device=config["infer_device"],
        verbose=bool(config["verbose"]),
        expl=float(config.get("expl", 1.0)),
        val_expl=float(config.get("val_expl", 0.01)),
        importance_sampling=bool(config.get("importance_sampling", True)),
        importance_sampling_threshold=float(
            config.get("importance_sampling_threshold", 100.0)
        ),
        clear_value_buffer=bool(config.get("clear_value_buffer", True)),
        val_bootstrap=bool(config.get("val_bootstrap", False)),
        use_balanced_probs=bool(config.get("use_balanced_probs", False)),
        balanced_sampling_mix=float(config.get("balanced_sampling_mix", 1.0)),
        track_sampling_coverage=bool(
            config.get("track_sampling_coverage", False)
        ),
        val_op_prob=float(config.get("val_op_prob", 0.0)),
        all_actions=bool(config.get("all_actions", True)),
        use_reach_weighted_avg_policy_loss=bool(
            config.get("use_reach_weighted_avg_policy_loss", False)
        ),
        average_policy_weighting=str(config.get("average_policy_weighting", "linear")),
        reuse_regret_traversals_for_value=bool(
            config.get("reuse_regret_traversals_for_value", False)
        ),
        on_policy_joint_regret_updates=bool(
            config.get("on_policy_joint_regret_updates", False)
        ),
        value_test_traversals=int(config.get("value_test_traversals", 20)),
        bootstrap_value_with_separate_traversal=bool(
            config.get("bootstrap_value_with_separate_traversal", False)
        ),
        zero_regret_fallback=str(config.get("zero_regret_fallback", "argmax")),
        policy_network_activation=str(
            config.get("policy_network_activation", "leakyrelu")
        ),
        regret_network_activation=str(
            config.get("regret_network_activation", "leakyrelu")
        ),
        value_network_activation=str(
            config.get("value_network_activation", "leakyrelu")
        ),
        policy_network_layer_norm=bool(
            config.get("policy_network_layer_norm", True)
        ),
        regret_network_layer_norm=bool(
            config.get("regret_network_layer_norm", True)
        ),
        value_network_layer_norm=bool(
            config.get("value_network_layer_norm", True)
        ),
        policy_network_residual_mode=str(
            config.get("policy_network_residual_mode", "same_width")
        ),
        regret_network_residual_mode=str(
            config.get("regret_network_residual_mode", "same_width")
        ),
        value_network_residual_mode=str(
            config.get("value_network_residual_mode", "same_width")
        ),
        policy_network_head_depth=int(config.get("policy_network_head_depth", 0)),
        regret_network_head_depth=int(config.get("regret_network_head_depth", 0)),
        policy_network_head_units=config.get("policy_network_head_units"),
        regret_network_head_units=config.get("regret_network_head_units"),
        regret_network_output_mode=str(
            config.get("regret_network_output_mode", "direct")
        ),
        regret_target_baseline=str(
            config.get("regret_target_baseline", "author_state_value")
        ),
        regret_target_processing=str(config.get("regret_target_processing", "none")),
        regret_target_clip_value=float(config.get("regret_target_clip_value", 1.0)),
        regret_target_standardize_epsilon=float(
            config.get("regret_target_standardize_epsilon", 1e-6)
        ),
        regret_target_fixed_scale=float(
            config.get("regret_target_fixed_scale", 1.0)
        ),
        regret_target_ema_decay=float(
            config.get("regret_target_ema_decay", 0.99)
        ),
    )


def create_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    """Create a timestamped run directory under ``output_root``."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / f"{experiment_name}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def json_safe(value: Any) -> Any:
    """Convert common NumPy/scalar types to JSON-serialisable values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return None if not np.isfinite(val) else val
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def to_float(value: Any) -> float:
    """Convert TensorFlow/NumPy scalar-like values to ``float`` when possible."""
    if value is None:
        return np.nan
    try:
        return float(value.numpy())
    except Exception:
        try:
            return float(np.asarray(value))
        except Exception:
            return np.nan


def safe_array(values: Iterable[Any]) -> np.ndarray:
    """Return a float64 NumPy array, including for empty metric lists."""
    return np.asarray(values, dtype=np.float64)


def auc(x: Iterable[float], y: Iterable[float]) -> float:
    """Trapezoidal area under ``y`` with respect to ``x`` over finite points."""
    x_arr = safe_array(x)
    y_arr = safe_array(y)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(finite) < 2:
        return np.nan
    return float(np.trapz(y_arr[finite], x_arr[finite]))


def normalised_auc(x: Iterable[float], y: Iterable[float]) -> float:
    """Area under curve divided by the finite span of ``x``."""
    x_arr = safe_array(x)
    y_arr = safe_array(y)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x_finite = x_arr[finite]
    span = float(np.max(x_finite) - np.min(x_finite))
    if span <= 0:
        return np.nan
    return float(np.trapz(y_arr[finite], x_finite) / span)


def first_nodes_to_threshold(
    nodes: Iterable[float],
    metric: Iterable[float],
    threshold: float,
) -> float:
    idx = np.where(np.asarray(metric) <= threshold)[0]
    return np.nan if len(idx) == 0 else float(np.asarray(nodes)[idx[0]])


def first_time_to_threshold(
    times: Iterable[float],
    metric: Iterable[float],
    threshold: float,
) -> float:
    idx = np.where(np.asarray(metric) <= threshold)[0]
    return np.nan if len(idx) == 0 else float(np.asarray(times)[idx[0]])


def final_window_mean(values: Iterable[float], window: int = DEFAULT_FINAL_WINDOW) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    return float(np.mean(values[-min(window, values.size):]))


def safe_stats(values: Iterable[float]) -> Dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "n_finite": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "se": float(stats.sem(finite)) if finite.size > 1 else 0.0,
        "n_finite": int(finite.size),
    }


def cleanup_tensorflow_memory() -> None:
    """Release Python and Keras/TensorFlow state between independent solver runs."""
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


def run_single_seed(
    seed: int,
    config: Dict[str, Any],
    export_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run one ESCHER seed and return curves plus a compact summary."""
    set_seed_tf(seed)
    game = pyspiel.load_game(config["game_name"])
    solver = make_escher_solver(game, config)

    (
        _regret_losses,
        _policy_loss,
        convs,
        nodes_touched,
        avg_policy_values,
        diagnostics,
    ) = solver.solve()

    exploitability_curve = np.asarray(convs, dtype=np.float64) / 2.0
    nodes_touched = np.asarray(nodes_touched, dtype=np.float64)
    avg_policy_values = np.asarray(avg_policy_values, dtype=np.float64)
    value_error = np.abs(avg_policy_values - LEDUC_GAME_VALUE_PLAYER_0)
    diagnostics = {k: np.asarray(v) for k, v in diagnostics.items()}
    iterations = diagnostics["iteration"].astype(int)
    wall_clock = diagnostics["wall_clock_seconds"].astype(float)

    final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
    final_nash_conv = exploitability.nash_conv(game, final_policy)
    final_policy_value = expected_game_score.policy_value(
        game.new_initial_state(), [final_policy] * game.num_players()
    )[0]

    summary = {
        "seed": int(seed),
        "final_exploitability": float(exploitability_curve[-1]),
        "best_exploitability": float(np.min(exploitability_curve)),
        "final_window_mean_exploitability": final_window_mean(exploitability_curve),
        "final_policy_value": float(final_policy_value),
        "final_policy_value_error": float(abs(final_policy_value - LEDUC_GAME_VALUE_PLAYER_0)),
        "best_policy_value_error": float(np.min(value_error)),
        "final_nodes_touched": float(nodes_touched[-1]),
        "final_wall_clock_seconds": float(wall_clock[-1]),
        "nodes_to_exploitability_threshold": first_nodes_to_threshold(
            nodes_touched, exploitability_curve, config["exploitability_threshold"]
        ),
        "seconds_to_exploitability_threshold": first_time_to_threshold(
            wall_clock, exploitability_curve, config["exploitability_threshold"]
        ),
        "final_policy_loss": float(diagnostics["policy_loss"][-1]),
        "final_value_loss": float(diagnostics["value_loss"][-1]),
        "final_value_test_loss": float(diagnostics["value_test_loss"][-1]),
        "final_regret_loss_player_0": float(diagnostics["regret_loss_player_0"][-1]),
        "final_regret_loss_player_1": float(diagnostics["regret_loss_player_1"][-1]),
        "final_average_policy_buffer_size": int(diagnostics["average_policy_buffer_size"][-1]),
        "final_regret_buffer_size_player_0": int(diagnostics["regret_buffer_size_player_0"][-1]),
        "final_regret_buffer_size_player_1": int(diagnostics["regret_buffer_size_player_1"][-1]),
        "final_nash_conv_recomputed": float(final_nash_conv),
    }

    result = {
        "seed": int(seed),
        "iterations": iterations,
        "nodes_touched": nodes_touched,
        "wall_clock_seconds": wall_clock,
        "exploitability": exploitability_curve,
        "average_policy_value": avg_policy_values,
        "policy_value_error": value_error,
        "diagnostics": diagnostics,
        "summary": summary,
    }

    if bool(config.get("save_final_checkpoints", False)) and export_dir is not None:
        checkpoint_dir = Path(export_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_dir / f"seed_{seed}_final_model.pkl", "wb") as f:
            pickle.dump(solver.extract_full_model(), f)

    del solver
    cleanup_tensorflow_memory()
    return result


def evaluate_final_policy(game, solver: ESCHERSolver) -> Dict[str, float]:
    """Evaluate the solver's current playable average-policy network."""
    final_policy = policy.tabular_policy_from_callable(game, solver.action_probabilities)
    final_nash_conv = exploitability.nash_conv(game, final_policy)
    final_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [final_policy] * game.num_players()
    )
    return {
        "final_nash_conv_recomputed": float(final_nash_conv),
        "final_exploitability": float(final_nash_conv / 2.0),
        "final_policy_value": float(final_policy_values[0]),
        "final_policy_value_error": float(abs(final_policy_values[0] - LEDUC_GAME_VALUE_PLAYER_0)),
    }


def run_single_seed_variant(
    seed: int,
    config: Dict[str, Any],
    export_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run one seed for a named experiment variant.

    This runner is intended for ablations where final policy quality is always
    measured, but intermediate exploitability checkpoints may be disabled.
    """
    solver = None
    game = None
    try:
        set_seed_tf(seed)
        game = pyspiel.load_game(config["game_name"])
        solver = make_escher_solver(game, config)

        start = time.time()
        initial_eval = None
        initial_eval_wall_clock = np.nan
        if bool(config.get("evaluate_initial_policy", False)):
            # ESCHER builds all networks during solver construction. Exact
            # tabular evaluation therefore consumes no training samples and
            # does not advance the solver's traversal RNG stream.
            initial_eval = evaluate_final_policy(game, solver)
            initial_eval_wall_clock = time.time() - start
        (
            _regret_losses,
            policy_loss,
            convs,
            nodes_touched,
            avg_policy_values,
            diagnostics,
        ) = solver.solve()
        elapsed = time.time() - start

        final_eval = evaluate_final_policy(game, solver)
        if initial_eval is not None:
            # Include both added exact evaluations in Experiment 2's end-to-end
            # timing; legacy experiments retain their existing timing contract.
            elapsed = time.time() - start

        convs = safe_array(convs)
        nodes_touched = safe_array(nodes_touched)
        avg_policy_values = safe_array(avg_policy_values)
        intermediate_exploitability = (
            convs / 2.0 if convs.size else np.asarray([], dtype=np.float64)
        )
        intermediate_value_error = (
            np.abs(avg_policy_values - LEDUC_GAME_VALUE_PLAYER_0)
            if avg_policy_values.size
            else np.asarray([], dtype=np.float64)
        )
        diagnostics = {k: np.asarray(v) for k, v in diagnostics.items()}

        iterations = diagnostics.get("iteration", np.asarray([], dtype=int)).astype(int)
        wall_clock = diagnostics.get(
            "wall_clock_seconds",
            np.asarray([], dtype=np.float64),
        ).astype(float)
        if initial_eval is not None:
            wall_clock = wall_clock + float(initial_eval_wall_clock)
        # The final iteration need not coincide with an intermediate checkpoint
        # (Experiment 2 ends at iteration 404 with a 10-iteration interval).
        # Use the solver's actual terminal node counter and end-to-end runtime
        # for matched-budget stopping and the separately evaluated final policy.
        final_nodes = float(solver.get_num_nodes())
        final_wall_clock = (
            float(wall_clock[-1])
            if nodes_touched.size and nodes_touched[-1] == final_nodes
            else float(elapsed)
        )
        analysis_nodes = nodes_touched
        analysis_wall_clock = wall_clock
        analysis_exploitability = intermediate_exploitability
        if initial_eval is not None:
            analysis_nodes = np.concatenate(([0.0], nodes_touched))
            analysis_wall_clock = np.concatenate(
                ([float(initial_eval_wall_clock)], wall_clock)
            )
            analysis_exploitability = np.concatenate(
                ([float(initial_eval["final_exploitability"])], intermediate_exploitability)
            )

        summary = {
            "variant_id": config["variant_id"],
            "variant_label": config["variant_label"],
            "seed": int(seed),
            "compute_intermediate_exploitability": bool(config["compute_exploitability"]),
            "check_exploitability_every": int(config["check_exploitability_every"]),
            "policy_network_train_steps_per_event": int(config["policy_network_train_steps"]),
            "intermediate_policy_training_events_expected": int(
                config["intermediate_policy_training_events_expected"]
            ),
            "final_policy_training_events_expected": int(
                config["final_policy_training_events_expected"]
            ),
            "total_policy_training_events_expected": int(
                config["total_policy_training_events_expected"]
            ),
            "policy_gradient_steps_expected": int(config["policy_gradient_steps_expected"]),
            "elapsed_seconds": float(elapsed),
            "evaluate_initial_policy": initial_eval is not None,
            "initial_exploitability": (
                float(initial_eval["final_exploitability"])
                if initial_eval is not None
                else np.nan
            ),
            "initial_policy_value": (
                float(initial_eval["final_policy_value"])
                if initial_eval is not None
                else np.nan
            ),
            "initial_policy_value_error": (
                float(initial_eval["final_policy_value_error"])
                if initial_eval is not None
                else np.nan
            ),
            "num_intermediate_points": int(analysis_exploitability.size),
            "intermediate_final_exploitability": (
                float(intermediate_exploitability[-1])
                if intermediate_exploitability.size
                else np.nan
            ),
            "intermediate_best_exploitability": (
                float(np.min(analysis_exploitability))
                if analysis_exploitability.size
                else np.nan
            ),
            "intermediate_final_window_mean_exploitability": final_window_mean(
                analysis_exploitability
            ),
            "intermediate_exploitability_auc_nodes": auc(
                analysis_nodes, analysis_exploitability
            ),
            "intermediate_exploitability_normalised_auc_nodes": normalised_auc(
                analysis_nodes, analysis_exploitability
            ),
            "nodes_to_intermediate_exploitability_threshold": first_nodes_to_threshold(
                analysis_nodes,
                analysis_exploitability,
                config["exploitability_threshold"],
            ),
            "seconds_to_intermediate_exploitability_threshold": first_time_to_threshold(
                analysis_wall_clock,
                analysis_exploitability,
                config["exploitability_threshold"],
            ),
            "final_nodes_touched": final_nodes,
            "final_wall_clock_seconds": final_wall_clock,
            "final_policy_loss": to_float(policy_loss),
            **final_eval,
        }

        for key in [
            "policy_loss",
            "value_loss",
            "value_test_loss",
            "regret_loss_player_0",
            "regret_loss_player_1",
            "raw_regret_target_variance_player_0",
            "raw_regret_target_variance_player_1",
            "processed_regret_target_variance_player_0",
            "processed_regret_target_variance_player_1",
            "processed_regret_target_abs_mean_player_0",
            "processed_regret_target_abs_mean_player_1",
            "regret_target_standardization_mean_player_0",
            "regret_target_standardization_mean_player_1",
            "regret_target_standardization_scale_player_0",
            "regret_target_standardization_scale_player_1",
            "regret_target_processing_mean_player_0",
            "regret_target_processing_mean_player_1",
            "regret_target_processing_scale_player_0",
            "regret_target_processing_scale_player_1",
            "regret_target_clip_fraction_player_0",
            "regret_target_clip_fraction_player_1",
            "regret_target_sign_flip_fraction_player_0",
            "regret_target_sign_flip_fraction_player_1",
            "raw_regret_target_positive_fraction_player_0",
            "raw_regret_target_positive_fraction_player_1",
            "processed_regret_target_positive_fraction_player_0",
            "processed_regret_target_positive_fraction_player_1",
            "regret_target_sample_count_player_0",
            "regret_target_sample_count_player_1",
            "regret_target_bellman_residual_mean_player_0",
            "regret_target_bellman_residual_mean_player_1",
            "regret_target_bellman_residual_abs_mean_player_0",
            "regret_target_bellman_residual_abs_mean_player_1",
            "regret_target_bellman_residual_rmse_player_0",
            "regret_target_bellman_residual_rmse_player_1",
            "regret_target_policy_weighted_target_abs_mean_player_0",
            "regret_target_policy_weighted_target_abs_mean_player_1",
            "regret_target_all_legal_targets_negative_fraction_player_0",
            "regret_target_all_legal_targets_negative_fraction_player_1",
            "average_policy_buffer_size",
            "regret_buffer_size_player_0",
            "regret_buffer_size_player_1",
            "regret_replay_stream_count_player_0",
            "regret_replay_stream_count_player_1",
            "regret_replay_retention_fraction_player_0",
            "regret_replay_retention_fraction_player_1",
            "regret_replay_unique_infosets_player_0",
            "regret_replay_unique_infosets_player_1",
            "regret_replay_samples_per_infoset_min_player_0",
            "regret_replay_samples_per_infoset_min_player_1",
            "regret_replay_samples_per_infoset_mean_player_0",
            "regret_replay_samples_per_infoset_mean_player_1",
            "regret_replay_samples_per_infoset_max_player_0",
            "regret_replay_samples_per_infoset_max_player_1",
            "regret_replay_samples_per_infoset_cv_player_0",
            "regret_replay_samples_per_infoset_cv_player_1",
            "regret_replay_stored_weight_mean_player_0",
            "regret_replay_stored_weight_mean_player_1",
            "fixed_sampling_effective_balanced_mix",
            "fixed_sampling_legal_action_probability_min",
            "fixed_sampling_infoset_count_player_0",
            "fixed_sampling_infoset_count_player_1",
            "fixed_sampling_history_count_player_0",
            "fixed_sampling_history_count_player_1",
            "fixed_sampling_own_history_reach_min_player_0",
            "fixed_sampling_own_history_reach_min_player_1",
            "fixed_sampling_own_history_reach_mean_player_0",
            "fixed_sampling_own_history_reach_mean_player_1",
            "fixed_sampling_own_history_reach_cv_player_0",
            "fixed_sampling_own_history_reach_cv_player_1",
            "sampling_coverage_unique_infosets_player_0",
            "sampling_coverage_unique_infosets_player_1",
            "sampling_coverage_visits_min_player_0",
            "sampling_coverage_visits_min_player_1",
            "sampling_coverage_visits_mean_player_0",
            "sampling_coverage_visits_mean_player_1",
            "sampling_coverage_visits_max_player_0",
            "sampling_coverage_visits_max_player_1",
            "sampling_coverage_visits_cv_player_0",
            "sampling_coverage_visits_cv_player_1",
            "sampling_coverage_observed_own_reach_min_player_0",
            "sampling_coverage_observed_own_reach_min_player_1",
            "sampling_coverage_observed_own_reach_mean_player_0",
            "sampling_coverage_observed_own_reach_mean_player_1",
            "sampling_coverage_observed_own_reach_max_player_0",
            "sampling_coverage_observed_own_reach_max_player_1",
            "value_buffer_size",
            "value_test_buffer_size",
        ]:
            if key in diagnostics and len(diagnostics[key]):
                value = diagnostics[key][-1]
                summary[f"last_intermediate_{key}"] = (
                    float(value) if np.issubdtype(np.asarray(value).dtype, np.number) else value
                )
            else:
                summary[f"last_intermediate_{key}"] = np.nan

        curve_rows = []
        if initial_eval is not None:
            curve_rows.append({
                "variant_id": config["variant_id"],
                "variant_label": config["variant_label"],
                "seed": int(seed),
                "checkpoint_index": 0,
                "iteration": 0,
                "nodes_touched": 0.0,
                "wall_clock_seconds": float(initial_eval_wall_clock),
                "exploitability": float(initial_eval["final_exploitability"]),
                "average_policy_value": float(initial_eval["final_policy_value"]),
                "policy_value_error": float(initial_eval["final_policy_value_error"]),
                "checkpoint_kind": "initial_untrained_policy",
                "is_initial_policy_evaluation": True,
                "is_final_policy_evaluation": False,
            })
        if intermediate_exploitability.size:
            for idx, (iteration, node_count, wall_time, expl, value, value_err) in enumerate(
                zip(
                    iterations,
                    nodes_touched,
                    wall_clock,
                    intermediate_exploitability,
                    avg_policy_values,
                    intermediate_value_error,
                )
            ):
                row = {
                    "variant_id": config["variant_id"],
                    "variant_label": config["variant_label"],
                    "seed": int(seed),
                    "checkpoint_index": int(len(curve_rows)),
                    "iteration": int(iteration),
                    "nodes_touched": float(node_count),
                    "wall_clock_seconds": float(wall_time),
                    "exploitability": float(expl),
                    "average_policy_value": float(value),
                    "policy_value_error": float(value_err),
                    "checkpoint_kind": "outer_iteration",
                    "is_initial_policy_evaluation": False,
                    "is_final_policy_evaluation": False,
                }
                for key, arr in diagnostics.items():
                    if len(arr) > idx:
                        row[key] = to_float(arr[idx])
                curve_rows.append(row)

        curve_rows.append({
            "variant_id": config["variant_id"],
            "variant_label": config["variant_label"],
            "seed": int(seed),
            "checkpoint_index": int(len(curve_rows)),
            "iteration": int(config["num_iterations"]),
            "nodes_touched": final_nodes,
            "wall_clock_seconds": final_wall_clock,
            "exploitability": float(final_eval["final_exploitability"]),
            "average_policy_value": float(final_eval["final_policy_value"]),
            "policy_value_error": float(final_eval["final_policy_value_error"]),
            "checkpoint_kind": "final_policy_evaluation",
            "is_initial_policy_evaluation": False,
            "is_final_policy_evaluation": True,
        })

        result = {
            "seed": int(seed),
            "variant_id": config["variant_id"],
            "summary": summary,
            "curves": curve_rows,
        }

        if bool(config.get("save_final_checkpoints", False)) and export_dir is not None:
            checkpoint_dir = Path(export_dir) / "checkpoints" / config["variant_id"]
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_dir / f"seed_{seed}_final_model.pkl", "wb") as f:
                pickle.dump(solver.extract_full_model(), f)

        return result
    finally:
        del solver
        del game
        cleanup_tensorflow_memory()


def export_metadata(run_dir: Path, config: Dict[str, Any], seeds: List[int]) -> None:
    metadata = {
        "config": config,
        "seeds": seeds,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)


def export_seed_summary(run_dir: Path, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary_rows = [result["summary"] for result in results]
    summary_csv = run_dir / "seed_summary.csv"
    fields = list(summary_rows[0].keys())
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate = {}
    for field in fields:
        if field == "seed":
            continue
        aggregate[field] = safe_stats([row[field] for row in summary_rows])

    with open(run_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(aggregate), f, indent=2)
    return aggregate


def export_checkpoint_curves(run_dir: Path, results: List[Dict[str, Any]]) -> None:
    curve_csv = run_dir / "checkpoint_curves.csv"
    curve_fields = [
        "seed", "iteration", "nodes_touched", "wall_clock_seconds", "exploitability",
        "average_policy_value", "policy_value_error", "policy_loss", "value_loss",
        "value_test_loss", "regret_loss_player_0", "regret_loss_player_1",
        "average_policy_buffer_size", "regret_buffer_size_player_0",
        "regret_buffer_size_player_1", "value_buffer_size", "value_test_buffer_size",
    ]
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=curve_fields)
        writer.writeheader()
        for result in results:
            diag = result["diagnostics"]
            for i, iteration in enumerate(result["iterations"]):
                writer.writerow({
                    "seed": result["seed"],
                    "iteration": int(iteration),
                    "nodes_touched": float(result["nodes_touched"][i]),
                    "wall_clock_seconds": float(result["wall_clock_seconds"][i]),
                    "exploitability": float(result["exploitability"][i]),
                    "average_policy_value": float(result["average_policy_value"][i]),
                    "policy_value_error": float(result["policy_value_error"][i]),
                    "policy_loss": float(diag["policy_loss"][i]),
                    "value_loss": float(diag["value_loss"][i]),
                    "value_test_loss": float(diag["value_test_loss"][i]),
                    "regret_loss_player_0": float(diag["regret_loss_player_0"][i]),
                    "regret_loss_player_1": float(diag["regret_loss_player_1"][i]),
                    "average_policy_buffer_size": int(diag["average_policy_buffer_size"][i]),
                    "regret_buffer_size_player_0": int(diag["regret_buffer_size_player_0"][i]),
                    "regret_buffer_size_player_1": int(diag["regret_buffer_size_player_1"][i]),
                    "value_buffer_size": int(diag["value_buffer_size"][i]),
                    "value_test_buffer_size": int(diag["value_test_buffer_size"][i]),
                })
