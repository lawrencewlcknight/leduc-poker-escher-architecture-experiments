"""Run the five-times-longer ESCHER/VR-DeepCFR+ matched-node experiment."""

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
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    DEFAULT_FINAL_WINDOW,
    EXPLOITABILITY_THRESHOLD,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)

from .config import (  # noqa: E402
    ALGORITHMS,
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_SEEDS,
    ESCHER_CONFIG,
    EXPECTED_BATCH_RUNTIME_HOURS,
    NODE_BUDGET_MULTIPLIER,
    UPSTREAM,
    VR_PAPER_CONFIG,
)

LOGGER = logging.getLogger("escher_vs_vr_deep_cfr_5x_nodes")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": np.nan, "std": np.nan, "se": np.nan, "n_finite": 0}
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "se": std / math.sqrt(values.size),
        "n_finite": int(values.size),
    }


def _auc(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan
    return float(np.trapz(y[finite], x[finite]))


def _normalised_auc(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return np.nan
    span = float(np.max(x[finite]) - np.min(x[finite]))
    return np.nan if span <= 0 else _auc(x[finite], y[finite]) / span


def _first_x_to_threshold(
    x: Sequence[float], y: Sequence[float], threshold: float
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    matches = np.where(np.isfinite(x) & np.isfinite(y) & (y <= threshold))[0]
    return np.nan if not matches.size else float(x[matches[0]])


def _run_escher(seed: int, config: Dict[str, Any]) -> Dict[str, Any]:
    from escher_poker.experiment_utils import run_single_seed_variant

    result = run_single_seed_variant(seed, config)
    summary = result["summary"]
    summary.update(
        algorithm_id="escher_exp28",
        algorithm_label=ALGORITHMS["escher_exp28"]["algorithm_label"],
        target_nodes_touched=float(result["summary"]["final_nodes_touched"]),
        node_budget_delta=0.0,
        node_budget_relative_delta=0.0,
        best_exploitability=summary.get("intermediate_best_exploitability", np.nan),
        final_window_mean_exploitability=summary.get(
            "intermediate_final_window_mean_exploitability", np.nan
        ),
        exploitability_auc_nodes=summary.get(
            "intermediate_exploitability_auc_nodes", np.nan
        ),
        exploitability_normalised_auc_nodes=summary.get(
            "intermediate_exploitability_normalised_auc_nodes", np.nan
        ),
        nodes_to_exploitability_threshold=summary.get(
            "nodes_to_intermediate_exploitability_threshold", np.nan
        ),
        seconds_to_exploitability_threshold=summary.get(
            "seconds_to_intermediate_exploitability_threshold", np.nan
        ),
        num_iterations_completed=int(config["num_iterations"]),
    )
    for row in result["curves"]:
        row["algorithm_id"] = "escher_exp28"
        row["algorithm_label"] = ALGORITHMS["escher_exp28"]["algorithm_label"]
    return result


def _run_vr(
    algorithm_id: str,
    seed: int,
    config: Dict[str, Any],
    target_nodes: int,
) -> Dict[str, Any]:
    import torch

    from vr_deep_cfr import VRDeepDCFRPlus, VRDeepPDCFRPlus
    from vr_deep_cfr.logger import Logger as VRLogger

    spec = ALGORITHMS[algorithm_id]
    solver_class = {
        "VRDeepDCFRPlus": VRDeepDCFRPlus,
        "VRDeepPDCFRPlus": VRDeepPDCFRPlus,
    }[spec["class_name"]]
    kwargs = {
        key: value
        for key, value in config.items()
        if key not in {
            "max_num_iterations",
            "preserve_evaluation_rng",
            "evaluate_initial_policy",
            "early_evaluation_node_thresholds",
        }
    }
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        alpha=float(spec["alpha"]),
        gamma=float(spec["gamma"]),
        seed=int(seed),
        logger=VRLogger(verbose=False),
    )
    if spec["reinitialize_imm_regret_networks"] is not None:
        kwargs["reinitialize_imm_regret_networks"] = bool(
            spec["reinitialize_imm_regret_networks"]
        )

    solver = solver_class(**kwargs)
    solver.target_nodes_touched = int(target_nodes)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    checkpoint_rows = solver.solve()

    curves = []
    for checkpoint_index, raw in enumerate(checkpoint_rows):
        value = float(raw["average_policy_value"])
        curves.append(
            {
                "algorithm_id": algorithm_id,
                "algorithm_label": spec["algorithm_label"],
                "variant_id": algorithm_id,
                "variant_label": spec["algorithm_label"],
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
            }
        )

    final = curves[-1]
    initial_checkpoint = next(
        (row for row in curves if row["checkpoint_kind"] == "initial_untrained_policy"),
        None,
    )
    early_checkpoint = next(
        (row for row in curves if row["checkpoint_kind"] == "early_node_threshold"),
        None,
    )
    exploitabilities = [row["exploitability"] for row in curves]
    nodes = [row["nodes_touched"] for row in curves]
    wall_times = [row["wall_clock_seconds"] for row in curves]
    node_delta = float(final["nodes_touched"] - target_nodes)
    summary = {
        "algorithm_id": algorithm_id,
        "algorithm_label": spec["algorithm_label"],
        "variant_id": algorithm_id,
        "variant_label": spec["algorithm_label"],
        "seed": int(seed),
        "evaluate_initial_policy": initial_checkpoint is not None,
        "initial_exploitability": (
            float(initial_checkpoint["exploitability"])
            if initial_checkpoint is not None
            else np.nan
        ),
        "initial_policy_value": (
            float(initial_checkpoint["average_policy_value"])
            if initial_checkpoint is not None
            else np.nan
        ),
        "initial_policy_value_error": (
            float(initial_checkpoint["policy_value_error"])
            if initial_checkpoint is not None
            else np.nan
        ),
        "early_evaluation_target_nodes": (
            float(early_checkpoint["checkpoint_target_nodes"])
            if early_checkpoint is not None
            else np.nan
        ),
        "early_evaluation_actual_nodes": (
            float(early_checkpoint["nodes_touched"])
            if early_checkpoint is not None
            else np.nan
        ),
        "early_evaluation_exploitability": (
            float(early_checkpoint["exploitability"])
            if early_checkpoint is not None
            else np.nan
        ),
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
        "exploitability_auc_nodes": _auc(nodes, exploitabilities),
        "exploitability_normalised_auc_nodes": _normalised_auc(nodes, exploitabilities),
        "nodes_to_exploitability_threshold": _first_x_to_threshold(
            nodes, exploitabilities, EXPLOITABILITY_THRESHOLD
        ),
        "seconds_to_exploitability_threshold": _first_x_to_threshold(
            wall_times, exploitabilities, EXPLOITABILITY_THRESHOLD
        ),
        "elapsed_seconds": float(final["wall_clock_seconds"]),
        "target_nodes_touched": float(target_nodes),
        "node_budget_delta": node_delta,
        "node_budget_relative_delta": node_delta / float(target_nodes),
        "final_average_policy_buffer_size": len(solver.ave_policy_trainer.buffer),
        "final_regret_buffer_size_player_0": len(solver.regret_trainers[0].buffer),
        "final_regret_buffer_size_player_1": len(solver.regret_trainers[1].buffer),
    }
    if solver.use_baseline:
        summary["final_history_value_buffer_size"] = len(solver.q_value_trainer.buffer)

    del solver
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return {"seed": int(seed), "summary": summary, "curves": curves}


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["algorithm_id"] == "escher_exp28":
        result = _run_escher(int(payload["seed"]), payload["config"])
    else:
        result = _run_vr(
            payload["algorithm_id"],
            int(payload["seed"]),
            payload["config"],
            int(payload["target_nodes_touched"]),
        )
    _write_json(output_path, result)
    return 0


def _run_subprocess(
    run_dir: Path,
    algorithm_id: str,
    seed: int,
    config: Dict[str, Any],
    target_nodes: int | None = None,
) -> Dict[str, Any]:
    stem = f"{algorithm_id}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    _write_json(
        input_path,
        {
            "algorithm_id": algorithm_id,
            "seed": int(seed),
            "config": config,
            "target_nodes_touched": target_nodes,
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run",
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
    with open(output_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate(summary_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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


def _paired_differences(summary_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for algorithm_id in ("vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"):
        for seed in sorted({int(row["seed"]) for row in summary_rows}):
            baseline = indexed.get(("escher_exp28", seed))
            candidate = indexed.get((algorithm_id, seed))
            if baseline is None or candidate is None:
                continue
            rows.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "seed": seed,
                    "exploitability_difference_vs_escher": (
                        candidate["final_exploitability"] - baseline["final_exploitability"]
                    ),
                    "policy_value_error_difference_vs_escher": (
                        candidate["final_policy_value_error"]
                        - baseline["final_policy_value_error"]
                    ),
                    "nodes_difference_vs_escher": (
                        candidate["final_nodes_touched"] - baseline["final_nodes_touched"]
                    ),
                    "nodes_relative_difference_vs_escher": (
                        candidate["final_nodes_touched"] / baseline["final_nodes_touched"] - 1.0
                    ),
                }
            )
    return rows


def _mean_curve(rows: Sequence[Mapping[str, Any]], algorithm_id: str, y_key: str):
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
        y = np.asarray([row[y_key] for row in at_checkpoint], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        xs.append(float(np.mean(x[finite])))
        means.append(float(np.mean(y[finite])))
        ses.append(float(_stats(y[finite])["se"]))
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _plot_curves(run_dir: Path, curve_rows: Sequence[Mapping[str, Any]]) -> None:
    specs = [
        (
            "exploitability",
            "Exploitability (NashConv / 2)",
            "Exploitability by nodes touched",
            "exploitability_by_nodes.png",
            NASH_EXPLOITABILITY_TARGET,
            NASH_EXPLOITABILITY_TARGET_LABEL,
        ),
        (
            "average_policy_value",
            "Average-policy value",
            "Average-policy value by nodes touched",
            "average_policy_value_by_nodes.png",
            LEDUC_GAME_VALUE_PLAYER_0,
            AVERAGE_POLICY_VALUE_TARGET_LABEL,
        ),
        (
            "policy_value_error",
            r"$|v(\sigma) - v^*_{\mathrm{Leduc}}|$",
            "Policy-value error by nodes touched",
            "policy_value_error_by_nodes.png",
            None,
            None,
        ),
    ]
    colors = dict(zip(ALGORITHMS, ("#8c564b", "#1f77b4", "#2ca02c")))
    for y_key, ylabel, title, filename, target, target_label in specs:
        fig, ax = plt.subplots(figsize=(10, 6))
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
                    [row[y_key] for row in seed_rows],
                    color=colors[algorithm_id],
                    linewidth=1,
                    alpha=0.18,
                )
            x, mean, se = _mean_curve(curve_rows, algorithm_id, y_key)
            ax.plot(
                x,
                mean,
                marker="o",
                linewidth=2,
                color=colors[algorithm_id],
                label=spec["algorithm_label"],
            )
            ax.fill_between(x, mean - se, mean + se, color=colors[algorithm_id], alpha=0.14)
        if target is not None:
            ax.axhline(target, color="black", linestyle="--", linewidth=1, label=target_label)
        ax.set_xlabel("Nodes touched")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, f"ESCHER vs VR-DeepCFR+: {title.lower()}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _plot_final(run_dir: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    labels, means, ses = [], [], []
    for algorithm_id, spec in ALGORITHMS.items():
        values = [
            row["final_exploitability"]
            for row in summary_rows
            if row["algorithm_id"] == algorithm_id
        ]
        stats = _stats(values)
        labels.append(spec["algorithm_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "ESCHER vs VR-DeepCFR+: final exploitability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_algorithm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_seeds(value: str | None) -> List[int]:
    if value is None:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _apply_overrides(args, escher: Dict[str, Any], vr: Dict[str, Any]) -> None:
    escher_overrides = {
        "num_iterations": args.escher_iterations,
        "num_traversals": args.escher_traversals,
        "num_val_fn_traversals": args.escher_value_traversals,
        "check_exploitability_every": args.escher_evaluation_interval,
        "policy_network_train_steps": args.escher_policy_train_steps,
        "regret_network_train_steps": args.escher_regret_train_steps,
        "value_network_train_steps": args.escher_value_train_steps,
        "batch_size_regret": args.escher_batch_size,
        "batch_size_value": args.escher_batch_size,
        "batch_size_average_policy": args.escher_batch_size,
        "memory_capacity": args.escher_memory_capacity,
    }
    vr_overrides = {
        "num_traversals": args.vr_traversals,
        "max_num_iterations": args.vr_max_iterations,
        "advantage_network_train_steps": args.vr_advantage_train_steps,
        "ave_policy_network_train_steps": args.vr_policy_train_steps,
        "baseline_network_train_steps": args.vr_baseline_train_steps,
        "advantage_batch_size": args.vr_batch_size,
        "ave_policy_batch_size": args.vr_batch_size,
        "baseline_batch_size": args.vr_batch_size,
        "advantage_buffer_size": args.vr_buffer_size,
        "ave_policy_buffer_size": args.vr_buffer_size,
        "baseline_buffer_size": args.vr_buffer_size,
    }
    for key, value in escher_overrides.items():
        if value is not None:
            escher[key] = value
    for key, value in vr_overrides.items():
        if value is not None:
            vr[key] = value
    if args.vr_early_evaluation_nodes is not None:
        vr["early_evaluation_node_thresholds"] = (
            int(args.vr_early_evaluation_nodes),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/escher_vs_vr_deep_cfr_5x_nodes")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--escher-iterations", type=int)
    parser.add_argument("--escher-traversals", type=int)
    parser.add_argument("--escher-value-traversals", type=int)
    parser.add_argument("--escher-evaluation-interval", type=int)
    parser.add_argument("--escher-policy-train-steps", type=int)
    parser.add_argument("--escher-regret-train-steps", type=int)
    parser.add_argument("--escher-value-train-steps", type=int)
    parser.add_argument("--escher-batch-size", type=int)
    parser.add_argument("--escher-memory-capacity", type=int)
    parser.add_argument("--vr-traversals", type=int)
    parser.add_argument("--vr-max-iterations", type=int)
    parser.add_argument("--vr-advantage-train-steps", type=int)
    parser.add_argument("--vr-policy-train-steps", type=int)
    parser.add_argument("--vr-baseline-train-steps", type=int)
    parser.add_argument("--vr-batch-size", type=int)
    parser.add_argument("--vr-buffer-size", type=int)
    parser.add_argument("--vr-early-evaluation-nodes", type=int)
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
    escher_config = deepcopy(ESCHER_CONFIG)
    vr_config = deepcopy(VR_PAPER_CONFIG)
    _apply_overrides(args, escher_config, vr_config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / f"escher_vs_vr_deep_cfr_5x_nodes_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metadata = {
        "seeds": seeds,
        "algorithms": ALGORITHMS,
        "escher_config": escher_config,
        "vr_paper_config": vr_config,
        "upstream": UPSTREAM,
        "node_budget_multiplier": NODE_BUDGET_MULTIPLIER,
        "expected_batch_runtime_hours": EXPECTED_BATCH_RUNTIME_HOURS,
        "configured_batch_timeout_seconds": BATCH_TIMEOUT_SECONDS,
        "protocol": {
            "node_matching": (
                "Each VR run stops after the first complete outer iteration that reaches "
                "or exceeds the final nodes touched by ESCHER for the same seed."
            ),
            "vr_evaluation_frequency": "Every outer iteration (evaluation_frequency=1).",
            "initial_evaluation": (
                "All algorithms are evaluated before training at zero training nodes."
            ),
            "vr_early_evaluation": (
                "Each VR algorithm is additionally evaluated after the first complete "
                "trajectory crossing 10,000 training nodes."
            ),
            "rng_isolation": (
                "Python, NumPy and PyTorch RNG states are restored after each average-policy "
                "fit and exact evaluation."
            ),
            "node_counter_scope": "Training traversal nodes only; exact evaluation nodes excluded.",
        },
        "released_yaml_differences": {
            "advantage_buffer_size": {"paper": 1_000_000, "released_yaml": 150_000},
            "baseline_network_train_steps": {"paper": 10_000, "released_yaml": 1_000},
        },
        "upstream_correctness_corrections": [
            "Attach the immediate-regret optimiser to the immediate-regret model.",
            (
                "Pass reinitialize_imm_regret_networks and use_regret_matching_argmax "
                "to the VR-DeepPDCFR+ trainer in their declared order."
            ),
            (
                "Apply immediate-regret reinitialisation when cumulative-advantage "
                "reinitialisation is disabled, as required by the paper configuration."
            ),
            "Track circular history-value buffer occupancy independently of its write index.",
        ],
    }
    _write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    target_by_seed = {}
    schedule = [("escher_exp28", seed) for seed in seeds]
    for algorithm_id in ("vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"):
        schedule.extend((algorithm_id, seed) for seed in seeds)

    for algorithm_id, seed in schedule:
        try:
            target = target_by_seed.get(seed)
            if algorithm_id != "escher_exp28" and target is None:
                raise RuntimeError(f"Missing paired ESCHER node budget for seed {seed}")
            LOGGER.info(
                "Running %s, seed %s%s",
                algorithm_id,
                seed,
                f", target={target}" if target else "",
            )
            result = _run_subprocess(
                run_dir,
                algorithm_id,
                seed,
                escher_config if algorithm_id == "escher_exp28" else vr_config,
                target,
            )
            results.append(result)
            if algorithm_id == "escher_exp28":
                target_by_seed[seed] = int(result["summary"]["final_nodes_touched"])
            _write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failure = {
                "algorithm_id": algorithm_id,
                "seed": seed,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _write_json(run_dir / "failed_runs.json", failures)
            LOGGER.error("%s seed %s failed: %s", algorithm_id, seed, exc)
            if not args.continue_on_error:
                return 2

    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    paired_rows = _paired_differences(summary_rows)
    aggregate = _aggregate(summary_rows)
    paired_aggregate = {
        algorithm_id: {
            field: _stats(row[field] for row in paired_rows if row["algorithm_id"] == algorithm_id)
            for field in (
                "exploitability_difference_vs_escher",
                "policy_value_error_difference_vs_escher",
                "nodes_difference_vs_escher",
                "nodes_relative_difference_vs_escher",
            )
        }
        for algorithm_id in ("vr_deep_dcfr_plus", "vr_deep_pdcfr_plus")
    }
    _write_csv(run_dir / "seed_summary.csv", summary_rows)
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows)
    _write_csv(run_dir / "paired_differences_vs_escher.csv", paired_rows)
    _write_json(run_dir / "aggregate_summary.json", aggregate)
    _write_json(run_dir / "paired_difference_summary.json", paired_aggregate)
    _write_json(
        run_dir / "summary.json",
        {"seed_summary": summary_rows, "aggregate": aggregate, "failures": failures},
    )
    if curve_rows:
        _plot_curves(run_dir, curve_rows)
        _plot_final(run_dir, summary_rows)
    LOGGER.info("Outputs saved to %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
