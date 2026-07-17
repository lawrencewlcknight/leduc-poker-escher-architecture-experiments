"""Run Experiment 3 and combine its curves with saved Experiment 1 results."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import gc
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib_cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    DEFAULT_FINAL_WINDOW,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)

from .config import (  # noqa: E402
    ADAPTIVE_CONFIG,
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    DEFAULT_SEEDS,
    EXPERIMENT_1_NODE_TARGETS,
    EXPERIMENT_1_SOURCE,
    REFERENCE_CURVES,
)

LOGGER = logging.getLogger("adaptive_residual_predictive_escher")

ALGORITHMS = {
    "escher_exp28": {"algorithm_label": "ESCHER (Experiment 28)"},
    "vr_deep_dcfr_plus": {"algorithm_label": "VR-DeepDCFR+"},
    "vr_deep_pdcfr_plus": {"algorithm_label": "VR-DeepPDCFR+"},
    ALGORITHM_ID: {"algorithm_label": ALGORITHM_LABEL},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_json_safe(list(rows)))


def _stats(values: Iterable[float]) -> Dict[str, float | int]:
    finite = np.asarray(list(values), dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "n_finite": 0}
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return {
        "mean": float(np.mean(finite)),
        "std": std,
        "se": std / math.sqrt(finite.size),
        "n_finite": int(finite.size),
    }


def _normalised_auc(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if np.count_nonzero(finite) < 2:
        return np.nan
    x_values = x_values[finite]
    y_values = y_values[finite]
    span = float(np.max(x_values) - np.min(x_values))
    if span <= 0.0:
        return np.nan
    return float(np.trapz(y_values, x_values) / span)


def _first_x_to_threshold(x, y, threshold):
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    matches = np.where(
        np.isfinite(x_values) & np.isfinite(y_values) & (y_values <= threshold)
    )[0]
    return np.nan if not matches.size else float(x_values[matches[0]])


def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_reference_curves(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Experiment 1 reference curves not found: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("seed", "checkpoint_index"):
                row[key] = int(float(row[key]))
            for key in (
                "iteration",
                "episode",
                "nodes_touched",
                "wall_clock_seconds",
                "exploitability",
                "average_policy_value",
                "policy_value_error",
            ):
                row[key] = float(row[key]) if row.get(key) not in {None, ""} else np.nan
            row["is_final_policy_evaluation"] = _parse_bool(
                row.get("is_final_policy_evaluation", False)
            )
            row["result_source"] = "saved_experiment_1"
            rows.append(row)
    expected = {"escher_exp28", "vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"}
    if {row["algorithm_id"] for row in rows} != expected:
        raise ValueError("Reference curves do not contain all three Experiment 1 arms")
    return rows


def _reference_final_rows(reference_rows):
    final_rows = []
    for algorithm_id in (
        "escher_exp28",
        "vr_deep_dcfr_plus",
        "vr_deep_pdcfr_plus",
    ):
        for seed in DEFAULT_SEEDS:
            candidates = [
                row
                for row in reference_rows
                if row["algorithm_id"] == algorithm_id and int(row["seed"]) == seed
            ]
            if not candidates:
                continue
            explicit = [row for row in candidates if row["is_final_policy_evaluation"]]
            final_rows.append((explicit or candidates)[-1])
    return final_rows


def _run_adaptive(seed: int, config: Dict[str, Any], target_nodes: int):
    import torch

    from adaptive_escher import AdaptiveResidualPredictiveEscher
    from vr_deep_cfr.logger import Logger

    control_fields = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in control_fields}
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    solver = AdaptiveResidualPredictiveEscher(**kwargs)
    solver.target_nodes_touched = int(target_nodes)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config["evaluate_initial_policy"])
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config["early_evaluation_node_thresholds"]
    )
    raw_checkpoints = solver.solve()

    curves = []
    for checkpoint_index, raw in enumerate(raw_checkpoints):
        value = float(raw["average_policy_value"])
        row = {
            "algorithm_id": ALGORITHM_ID,
            "algorithm_label": ALGORITHM_LABEL,
            "variant_id": ALGORITHM_ID,
            "variant_label": ALGORITHM_LABEL,
            "seed": int(seed),
            "checkpoint_index": int(checkpoint_index),
            "iteration": int(raw["iteration"]),
            "episode": int(raw["episode"]),
            "nodes_touched": float(raw["nodes_touched"]),
            "wall_clock_seconds": float(raw["wall_clock_seconds"]),
            "exploitability": float(raw["exp"]),
            "average_policy_value": value,
            "policy_value_error": abs(value - LEDUC_GAME_VALUE_PLAYER_0),
            "average_policy_loss": float(raw.get("average_policy_loss", np.nan)),
            "regret_loss_player_0": float(raw.get("regret_loss_0", np.nan)),
            "regret_loss_player_1": float(raw.get("regret_loss_1", np.nan)),
            "baseline_loss_player_0": float(raw.get("baseline_loss_0", np.nan)),
            "baseline_loss_player_1": float(raw.get("baseline_loss_1", np.nan)),
            "checkpoint_kind": str(raw.get("checkpoint_kind", "outer_iteration")),
            "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
            "is_initial_policy_evaluation": (
                raw.get("checkpoint_kind") == "initial_untrained_policy"
            ),
            "is_final_policy_evaluation": False,
            "result_source": "experiment_3_new_run",
        }
        for key in (
            "adaptive_lambda_schedule_floor",
            "adaptive_lambda_mean",
            "adaptive_lambda_min",
            "adaptive_lambda_max",
            "q_residual_abs_mean",
            "residual_correction_abs_mean",
            "policy_weighted_advantage_abs_mean",
            "adaptive_estimator_sample_count",
            "full_support_traverser_sampling_min_probability",
            "q_target_version",
        ):
            row[key] = float(raw.get(key, np.nan))
        curves.append(row)

    final = curves[-1]
    training_curves = [
        row for row in curves if not row.get("is_initial_policy_evaluation", False)
    ]
    exploitabilities = [row["exploitability"] for row in training_curves]
    nodes = [row["nodes_touched"] for row in training_curves]
    wall_times = [row["wall_clock_seconds"] for row in training_curves]
    node_delta = float(final["nodes_touched"] - target_nodes)
    summary = {
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "seed": int(seed),
        "final_exploitability": float(final["exploitability"]),
        "best_exploitability": float(np.min(exploitabilities)),
        "final_window_mean_exploitability": float(
            np.mean(exploitabilities[-min(DEFAULT_FINAL_WINDOW, len(exploitabilities)):])
        ),
        "final_policy_value": float(final["average_policy_value"]),
        "final_policy_value_error": float(final["policy_value_error"]),
        "final_nash_conv_recomputed": 2.0 * float(final["exploitability"]),
        "final_nodes_touched": float(final["nodes_touched"]),
        "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
        "num_iterations_completed": int(final["iteration"]),
        "num_intermediate_points": len(curves),
        "exploitability_normalised_auc_nodes": _normalised_auc(nodes, exploitabilities),
        "nodes_to_exploitability_threshold": _first_x_to_threshold(
            nodes, exploitabilities, EXPLOITABILITY_THRESHOLD
        ),
        "seconds_to_exploitability_threshold": _first_x_to_threshold(
            wall_times, exploitabilities, EXPLOITABILITY_THRESHOLD
        ),
        "target_nodes_touched": float(target_nodes),
        "node_budget_delta": node_delta,
        "node_budget_relative_delta": node_delta / float(target_nodes),
        "final_average_policy_buffer_size": len(solver.ave_policy_trainer.buffer),
        "final_advantage_buffer_size_player_0": len(solver.regret_trainers[0].buffer),
        "final_advantage_buffer_size_player_1": len(solver.regret_trainers[1].buffer),
        "final_history_value_buffer_size": len(solver.q_value_trainer.buffer),
        "final_adaptive_lambda_mean": float(final["adaptive_lambda_mean"]),
        "final_adaptive_lambda_floor": float(final["adaptive_lambda_schedule_floor"]),
        "final_q_residual_abs_mean": float(final["q_residual_abs_mean"]),
        "final_policy_weighted_advantage_abs_mean": float(
            final["policy_weighted_advantage_abs_mean"]
        ),
        "q_target_versions": int(solver.q_value_trainer.target_version),
    }
    del solver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"seed": int(seed), "summary": summary, "curves": curves}


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_adaptive(
        int(payload["seed"]),
        payload["config"],
        int(payload["target_nodes_touched"]),
    )
    _write_json(output_path, result)
    return 0


def _run_subprocess(run_dir, seed, config, target_nodes):
    stem = f"{ALGORITHM_ID}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    _write_json(
        input_path,
        {"seed": seed, "config": config, "target_nodes_touched": target_nodes},
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.adaptive_residual_predictive_escher.run",
        "--worker-input-json",
        str(input_path),
        "--worker-output-json",
        str(output_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"{stem} failed; see {log_path}")
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)


def _mean_curve(rows, algorithm_id):
    selected = [
        row
        for row in rows
        if row["algorithm_id"] == algorithm_id
        and not bool(row.get("is_final_policy_evaluation", False))
    ]
    checkpoints = sorted({int(row["checkpoint_index"]) for row in selected})
    xs, means, ses = [], [], []
    for checkpoint in checkpoints:
        at_checkpoint = [
            row for row in selected if int(row["checkpoint_index"]) == checkpoint
        ]
        x = np.asarray([row["nodes_touched"] for row in at_checkpoint], dtype=float)
        y = np.asarray([row["exploitability"] for row in at_checkpoint], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            xs.append(float(np.mean(x[finite])))
            means.append(float(np.mean(y[finite])))
            ses.append(float(_stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _plot_combined_exploitability(run_dir, curve_rows):
    colors = {
        "escher_exp28": "#8c564b",
        "vr_deep_dcfr_plus": "#1f77b4",
        "vr_deep_pdcfr_plus": "#2ca02c",
        ALGORITHM_ID: "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for algorithm_id, spec in ALGORITHMS.items():
        algorithm_rows = [
            row
            for row in curve_rows
            if row["algorithm_id"] == algorithm_id
            and not bool(row.get("is_final_policy_evaluation", False))
        ]
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = sorted(
                [row for row in algorithm_rows if int(row["seed"]) == seed],
                key=lambda row: row["nodes_touched"],
            )
            ax.plot(
                [row["nodes_touched"] for row in seed_rows],
                [row["exploitability"] for row in seed_rows],
                color=colors[algorithm_id],
                linewidth=1,
                alpha=0.16,
            )
        x, mean, se = _mean_curve(curve_rows, algorithm_id)
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            color=colors[algorithm_id],
            label=spec["algorithm_label"],
        )
        ax.fill_between(
            x,
            mean - se,
            mean + se,
            color=colors[algorithm_id],
            alpha=0.14,
        )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(
        ax,
        "Experiment 3 adaptive predictive ESCHER vs Experiment 1",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        run_dir / "combined_exploitability_by_nodes.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_final(run_dir, final_rows):
    labels, means, ses = [], [], []
    for algorithm_id, spec in ALGORITHMS.items():
        values = [
            row["final_exploitability"]
            for row in final_rows
            if row["algorithm_id"] == algorithm_id
        ]
        stats = _stats(values)
        labels.append(spec["algorithm_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 3 and Experiment 1: final exploitability")
    fig.tight_layout()
    fig.savefig(
        run_dir / "combined_final_exploitability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _reference_summaries(reference_rows):
    summaries = []
    for row in _reference_final_rows(reference_rows):
        summaries.append(
            {
                "algorithm_id": row["algorithm_id"],
                "algorithm_label": row["algorithm_label"],
                "seed": int(row["seed"]),
                "final_exploitability": float(row["exploitability"]),
                "final_policy_value": float(row["average_policy_value"]),
                "final_policy_value_error": float(row["policy_value_error"]),
                "final_nodes_touched": float(row["nodes_touched"]),
                "final_wall_clock_seconds": float(row["wall_clock_seconds"]),
                "result_source": "saved_experiment_1",
            }
        )
    return summaries


def _aggregate(summary_rows):
    result = {}
    for algorithm_id in ALGORITHMS:
        rows = [row for row in summary_rows if row["algorithm_id"] == algorithm_id]
        numeric_fields = {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        result[algorithm_id] = {
            field: _stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return result


def _paired_differences(summary_rows):
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    pairs = []
    for baseline_id in ("escher_exp28", "vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"):
        for seed in DEFAULT_SEEDS:
            baseline = indexed.get((baseline_id, seed))
            candidate = indexed.get((ALGORITHM_ID, seed))
            if baseline is None or candidate is None:
                continue
            pairs.append(
                {
                    "baseline_algorithm_id": baseline_id,
                    "baseline_algorithm_label": ALGORITHMS[baseline_id]["algorithm_label"],
                    "seed": seed,
                    "exploitability_difference": (
                        candidate["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "nodes_difference": (
                        candidate["final_nodes_touched"] - baseline["final_nodes_touched"]
                    ),
                }
            )
    return pairs


def _parse_seeds(value):
    if value is None:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _apply_overrides(args, config):
    overrides = {
        "num_traversals": args.traversals,
        "max_num_iterations": args.max_iterations,
        "advantage_network_train_steps": args.advantage_train_steps,
        "ave_policy_network_train_steps": args.policy_train_steps,
        "baseline_network_train_steps": args.q_train_steps,
        "advantage_batch_size": args.batch_size,
        "ave_policy_batch_size": args.batch_size,
        "baseline_batch_size": args.batch_size,
        "advantage_buffer_size": args.buffer_size,
        "ave_policy_buffer_size": args.buffer_size,
        "baseline_buffer_size": args.buffer_size,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if args.early_evaluation_nodes is not None:
        config["early_evaluation_node_thresholds"] = (args.early_evaluation_nodes,)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/adaptive_residual_predictive_escher",
    )
    parser.add_argument("--reference-curves", type=Path, default=REFERENCE_CURVES)
    parser.add_argument("--seeds")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--target-nodes", type=int)
    parser.add_argument("--traversals", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--advantage-train-steps", type=int)
    parser.add_argument("--policy-train-steps", type=int)
    parser.add_argument("--q-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--early-evaluation-nodes", type=int)
    parser.add_argument("--worker-input-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            raise ValueError("Both worker paths are required")
        return _run_worker(args.worker_input_json, args.worker_output_json)

    seeds = _parse_seeds(args.seeds)
    if any(seed not in EXPERIMENT_1_NODE_TARGETS for seed in seeds):
        raise ValueError("Experiment 3 supports paired Experiment 1 seeds 0, 1 and 2")
    config = deepcopy(ADAPTIVE_CONFIG)
    _apply_overrides(args, config)
    reference_rows = _load_reference_curves(args.reference_curves)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"adaptive_residual_predictive_escher_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    targets = {
        seed: int(args.target_nodes or EXPERIMENT_1_NODE_TARGETS[seed])
        for seed in seeds
    }
    metadata = {
        "experiment_id": 3,
        "seeds": seeds,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "adaptive_config": config,
        "paired_node_targets": targets,
        "experiment_1_source": EXPERIMENT_1_SOURCE,
        "reference_curves_file": str(args.reference_curves),
        "protocol": {
            "baseline_reuse": (
                "All ESCHER and VR curves are immutable saved Experiment 1 results; "
                "only the adaptive algorithm is trained."
            ),
            "node_matching": (
                "The adaptive run stops after the first complete outer iteration "
                "crossing the paired Experiment 1 ESCHER node total."
            ),
            "sampling": (
                "Fixed uniform full-support sampling at traverser nodes; opponent "
                "actions sampled from the current strategy, matching ESCHER."
            ),
            "q_snapshot": (
                "Persistent online Q network; one target Q snapshot is frozen for both "
                "players' collection in an outer iteration and for each training phase, "
                "then hard-synchronised after Q optimisation."
            ),
            "lambda_predictability": (
                "Lambda uses only past residual EMAs and a deterministic schedule floor; "
                "the current sampled return is not observed before lambda is selected."
            ),
        },
    }
    _write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    for seed in seeds:
        try:
            LOGGER.info("Running adaptive solver seed %s, target %s", seed, targets[seed])
            result = _run_subprocess(run_dir, seed, config, targets[seed])
            results.append(result)
            _write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {"seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
            )
            _write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("Adaptive seed %s failed: %s", seed, exc)
            if not args.continue_on_error:
                return 2

    new_summaries = [result["summary"] for result in results]
    new_curves = [row for result in results for row in result["curves"]]
    reference_for_seeds = [row for row in reference_rows if int(row["seed"]) in seeds]
    combined_curves = [*reference_for_seeds, *new_curves]
    combined_summaries = [
        *_reference_summaries(reference_for_seeds),
        *new_summaries,
    ]
    paired = _paired_differences(combined_summaries)
    aggregate = _aggregate(combined_summaries)

    _write_csv(run_dir / "adaptive_seed_summary.csv", new_summaries)
    _write_csv(run_dir / "adaptive_checkpoint_curves.csv", new_curves)
    _write_csv(run_dir / "combined_checkpoint_curves.csv", combined_curves)
    _write_csv(run_dir / "combined_seed_summary.csv", combined_summaries)
    _write_csv(run_dir / "paired_differences.csv", paired)
    _write_json(run_dir / "aggregate_summary.json", aggregate)
    _write_json(
        run_dir / "summary.json",
        {
            "adaptive_seed_summary": new_summaries,
            "combined_aggregate": aggregate,
            "failures": failures,
        },
    )
    if combined_curves:
        _plot_combined_exploitability(run_dir, combined_curves)
        _plot_final(run_dir, combined_summaries)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
