"""CLI for Experiment 28 candidate ESCHER architecture validation."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XDG_CACHE_HOME", str((Path("outputs") / ".cache").resolve()))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path("outputs") / ".matplotlib_cache").resolve()),
)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    LEDUC_GAME_VALUE_PLAYER_0,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from escher_poker.experiment_utils import (  # noqa: E402
    cleanup_tensorflow_memory,
    create_run_dir,
    json_safe,
    run_single_seed_variant,
    safe_stats,
)
from experiments.leduc_poker.escher_variant_config_utils import (  # noqa: E402
    make_variant_config,
)

from .config import CANDIDATE_VARIANT, DEFAULT_CONFIG, DEFAULT_SEEDS  # noqa: E402

_LOGGER = logging.getLogger("escher_poker.experiment.candidate_architecture")

SUMMARY_HP_FIELDS = [
    "num_iterations",
    "num_traversals",
    "num_val_fn_traversals",
    "importance_sampling",
    "zero_regret_fallback",
    "all_actions",
    "expl",
    "val_expl",
    "average_policy_weighting",
    "use_balanced_probs",
    "balanced_sampling_mix",
    "track_sampling_coverage",
    "policy_network_layers",
    "regret_network_layers",
    "value_network_layers",
    "policy_network_activation",
    "regret_network_activation",
    "value_network_activation",
    "policy_network_layer_norm",
    "regret_network_layer_norm",
    "value_network_layer_norm",
    "policy_network_residual_mode",
    "regret_network_residual_mode",
    "value_network_residual_mode",
    "policy_network_head_depth",
    "regret_network_head_depth",
    "policy_network_head_units",
    "regret_network_head_units",
    "regret_network_output_mode",
    "regret_target_baseline",
    "regret_target_processing",
    "regret_target_clip_value",
    "regret_target_standardize_epsilon",
    "regret_target_fixed_scale",
    "regret_target_ema_decay",
    "regret_replay_mode",
    "regret_replay_rare_history_quota",
    "regret_replay_weight_floor",
    "batch_size_regret",
    "batch_size_value",
    "batch_size_average_policy",
    "policy_network_train_steps",
    "regret_network_train_steps",
    "value_network_train_steps",
    "reinitialize_regret_networks",
    "reinitialize_value_network",
]


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y", "1"}:
        return True
    if lowered in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def _parse_int_tuple(value: Optional[str]):
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_seeds(value: Optional[str]) -> List[int]:
    if not value:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), sort_keys=True))
        f.write("\n")


def _append_partial_result(run_dir: Path, result: Dict[str, Any]) -> None:
    _append_jsonl(run_dir / "partial_seed_summary.jsonl", result["summary"])
    for row in result["curves"]:
        _append_jsonl(run_dir / "partial_checkpoint_curves.jsonl", row)


def _write_failures(run_dir: Path, failures: Sequence[Mapping]) -> None:
    _write_json(run_dir / "failed_seeds.json", list(failures))


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


def _build_config(args) -> Dict:
    config = deepcopy(DEFAULT_CONFIG)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "check_exploitability_every": args.evaluation_interval,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "learning_rate": args.learning_rate,
        "memory_capacity": args.memory_capacity,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "policy_network_layers": _parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": _parse_int_tuple(args.regret_network_layers),
        "value_network_layers": _parse_int_tuple(args.value_network_layers),
        "regret_target_standardize_epsilon": args.regret_target_standardize_epsilon,
        "compute_exploitability": args.compute_exploitability,
        "save_final_checkpoints": args.save_final_checkpoints,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return make_variant_config(config, {})


def _augment_summary(summary: Dict, config: Mapping) -> Dict:
    summary["variant_description"] = config.get("variant_description", "")
    for key in SUMMARY_HP_FIELDS:
        summary[f"hp_{key}"] = json_safe(config.get(key))
    return summary


def _numeric_summary(rows: Sequence[Mapping]) -> Dict[str, Dict]:
    fields: List[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and key not in fields:
                fields.append(key)
    return {
        field: safe_stats([float(row.get(field, np.nan)) for row in rows])
        for field in fields
    }


def _mean_by_checkpoint(rows: Sequence[Mapping], y_key: str):
    checkpoint_rows = [
        row for row in rows
        if not bool(row.get("is_final_policy_evaluation", False))
    ]
    checkpoint_indices = sorted({int(row["checkpoint_index"]) for row in checkpoint_rows})
    xs, means, ses = [], [], []
    for checkpoint_index in checkpoint_indices:
        rows_at_checkpoint = [
            row for row in checkpoint_rows
            if int(row["checkpoint_index"]) == checkpoint_index
        ]
        x_vals = np.asarray(
            [row.get("nodes_touched", np.nan) for row in rows_at_checkpoint],
            dtype=float,
        )
        y_vals = np.asarray(
            [row.get(y_key, np.nan) for row in rows_at_checkpoint],
            dtype=float,
        )
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(finite):
            continue
        xs.append(float(np.mean(x_vals[finite])))
        means.append(float(np.mean(y_vals[finite])))
        if np.count_nonzero(finite) > 1:
            ses.append(float(np.std(y_vals[finite], ddof=1) / np.sqrt(np.count_nonzero(finite))))
        else:
            ses.append(0.0)
    return np.asarray(xs), np.asarray(means), np.asarray(ses)


def _plot_final_exploitability(run_dir: Path, summary_rows: Sequence[Mapping]) -> None:
    rows = sorted(summary_rows, key=lambda row: int(row["seed"]))
    labels = [str(row["seed"]) for row in rows]
    values = np.asarray([row["final_exploitability"] for row in rows], dtype=float)
    mean = float(np.mean(values)) if values.size else np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.axhline(
        mean,
        color="tab:blue",
        linestyle="-",
        linewidth=1.5,
        label="Mean final exploitability",
    )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 28 candidate architecture: final exploitability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(run_dir: Path, curve_rows: Sequence[Mapping]) -> None:
    plot_specs = [
        (
            "exploitability",
            "Exploitability (NashConv / 2)",
            "Experiment 28 candidate architecture: exploitability by nodes touched",
            "exploitability_by_nodes.png",
            NASH_EXPLOITABILITY_TARGET,
            NASH_EXPLOITABILITY_TARGET_LABEL,
        ),
        (
            "average_policy_value",
            "Average-policy value",
            "Experiment 28 candidate architecture: average-policy value by nodes touched",
            "average_policy_value_by_nodes.png",
            LEDUC_GAME_VALUE_PLAYER_0,
            AVERAGE_POLICY_VALUE_TARGET_LABEL,
        ),
        (
            "policy_value_error",
            r"$|v(\sigma) - v^*_{\mathrm{Leduc}}|$",
            "Experiment 28 candidate architecture: policy-value error by nodes touched",
            "policy_value_error_by_nodes.png",
            None,
            None,
        ),
    ]
    seed_rows = [
        row for row in curve_rows
        if not bool(row.get("is_final_policy_evaluation", False))
    ]
    for y_key, ylabel, title, filename, target, target_label in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for seed in sorted({int(row["seed"]) for row in seed_rows}):
            rows = sorted(
                [row for row in seed_rows if int(row["seed"]) == seed],
                key=lambda row: float(row.get("nodes_touched", np.nan)),
            )
            if not rows:
                continue
            plotted = True
            ax.plot(
                [row["nodes_touched"] for row in rows],
                [row[y_key] for row in rows],
                alpha=0.25,
                linewidth=1,
            )
        xs, means, ses = _mean_by_checkpoint(curve_rows, y_key)
        if xs.size:
            plotted = True
            ax.plot(xs, means, marker="o", linewidth=2, label="Mean across seeds")
            if np.any(ses > 0):
                ax.fill_between(
                    xs,
                    means - ses,
                    means + ses,
                    alpha=0.2,
                    label="Mean +/- s.e.",
                )
        if not plotted:
            plt.close(fig)
            continue
        if target is not None:
            ax.axhline(
                target,
                color="black",
                linestyle="--",
                linewidth=1,
                label=target_label,
            )
        ax.set_xlabel("Nodes touched")
        ax.set_ylabel(ylabel)
        set_chart_title(ax, title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _worker_stem(seed: int) -> str:
    return f"{CANDIDATE_VARIANT['variant_id']}_seed_{int(seed)}"


def _run_worker(
    worker_input_json: Union[str, Path],
    worker_output_json: Union[str, Path],
) -> int:
    input_path = Path(worker_input_json)
    output_path = Path(worker_output_json)
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    result = run_single_seed_variant(
        int(payload["seed"]),
        payload["config"],
        export_dir=payload.get("export_dir"),
    )
    _write_json(output_path, result)
    return 0


def _run_seed_subprocess(
    seed: int,
    config: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    stem = _worker_stem(seed)
    worker_input = run_dir / "worker_inputs" / f"{stem}.json"
    worker_output = run_dir / "worker_results" / f"{stem}.json"
    worker_log = run_dir / "worker_logs" / f"{stem}.log"

    _write_json(worker_input, {
        "seed": int(seed),
        "config": config,
        "export_dir": str(run_dir),
    })

    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.escher_candidate_architecture_multiseed.run",
        "--worker-input-json",
        str(worker_input),
        "--worker-output-json",
        str(worker_output),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    with open(worker_log, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Worker failed with exit code {completed.returncode}. "
            f"See {worker_log} for details."
        )
    if not worker_output.exists():
        raise RuntimeError(f"Worker completed without writing {worker_output}")

    with open(worker_output, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_seed(
    seed: int,
    config: Dict[str, Any],
    run_dir: Path,
    *,
    subprocess_isolation_enabled: bool,
) -> Dict[str, Any]:
    if subprocess_isolation_enabled:
        return _run_seed_subprocess(seed, config, run_dir)
    return run_single_seed_variant(seed, config, export_dir=run_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Experiment 28: ESCHER candidate architecture over five seeds."
    )
    parser.add_argument(
        "--output-root",
        default="outputs/candidate_architecture_multiseed",
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--batch-size-regret", type=int, default=None)
    parser.add_argument("--batch-size-value", type=int, default=None)
    parser.add_argument("--batch-size-average-policy", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--regret-network-train-steps", type=int, default=None)
    parser.add_argument("--value-network-train-steps", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--regret-network-layers", default=None)
    parser.add_argument("--value-network-layers", default=None)
    parser.add_argument("--regret-target-standardize-epsilon", type=float, default=None)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--save-final-checkpoints", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument(
        "--disable-subprocess-isolation",
        action="store_true",
        help=(
            "Run all seeds in the parent process. By default each seed runs in "
            "a fresh Python worker so TensorFlow state is released between "
            "independent full ESCHER trainings."
        ),
    )
    parser.add_argument("--worker-input-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            parser.error("--worker-input-json and --worker-output-json must be used together")
        return _run_worker(args.worker_input_json, args.worker_output_json)

    subprocess_isolation_enabled = not args.disable_subprocess_isolation
    config = _build_config(args)
    seeds = _parse_seeds(args.seeds)
    run_dir = create_run_dir(args.output_root, config["experiment_name"])
    _configure_logging(run_dir, args.verbose)

    metadata = {
        "config": config,
        "seeds": seeds,
        "candidate_variant": CANDIDATE_VARIANT,
        "subprocess_isolation_enabled": bool(subprocess_isolation_enabled),
        "incremental_outputs": {
            "partial_seed_summary_jsonl": "partial_seed_summary.jsonl",
            "partial_checkpoint_curves_jsonl": "partial_checkpoint_curves.jsonl",
            "worker_results_dir": "worker_results",
            "worker_logs_dir": "worker_logs",
        },
    }
    _write_json(run_dir / "experiment_metadata.json", metadata)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seeds: %s", seeds)
    _LOGGER.info("Config: %s", config)
    _LOGGER.info("Subprocess isolation enabled: %s", subprocess_isolation_enabled)

    results = []
    failures = []
    with tqdm(total=len(seeds), desc="Candidate architecture seeds") as progress:
        for seed in seeds:
            try:
                result = _run_seed(
                    seed,
                    config,
                    run_dir,
                    subprocess_isolation_enabled=subprocess_isolation_enabled,
                )
                result["summary"] = _augment_summary(result["summary"], config)
                results.append(result)
                _append_partial_result(run_dir, result)
                _LOGGER.info(
                    "Seed %s final exploitability %.6f",
                    seed,
                    result["summary"]["final_exploitability"],
                )
            except Exception as exc:  # pragma: no cover - operational robustness
                failure = {
                    "seed": int(seed),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                _LOGGER.error("Seed %s failed: %s", seed, exc)
                _write_failures(run_dir, failures)
                if not args.continue_on_error:
                    break
            finally:
                cleanup_tensorflow_memory()
                progress.update(1)

    if failures:
        _write_failures(run_dir, failures)

    if not results:
        _LOGGER.error("No seeds completed successfully.")
        return 1

    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    aggregate = _numeric_summary(summary_rows)

    _write_csv(run_dir / "seed_summary.csv", summary_rows)
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows)
    _write_json(run_dir / "aggregate_summary.json", aggregate)
    _write_json(run_dir / "summary.json", {
        "seed_summary": summary_rows,
        "aggregate_summary": aggregate,
        "failed_seeds": failures,
    })

    _plot_final_exploitability(run_dir, summary_rows)
    _plot_curves(run_dir, curve_rows)

    _LOGGER.info(
        "Aggregate final exploitability: %s",
        aggregate.get("final_exploitability"),
    )
    _LOGGER.info("Saved candidate architecture outputs to: %s", run_dir.resolve())
    return 0 if not failures else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
