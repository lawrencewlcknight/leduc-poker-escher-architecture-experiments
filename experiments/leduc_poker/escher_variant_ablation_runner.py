"""Generic runner for matched-seed ESCHER variant ablations."""

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
    parse_variant_ids,
    variant_lookup,
)

DEFAULT_SUMMARY_HP_FIELDS = [
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
    "learning_rate",
    "learning_rate_schedule",
    "learning_rate_end",
    "learning_rate_decay_rate",
    "learning_rate_warmup_iterations",
    "memory_capacity",
    "regret_replay_mode",
    "regret_replay_rare_history_quota",
    "regret_replay_weight_floor",
    "batch_size_regret",
    "batch_size_value",
    "batch_size_average_policy",
    "policy_network_train_steps",
    "regret_network_train_steps",
    "value_network_train_steps",
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
    "reinitialize_regret_networks",
    "reinitialize_value_network",
]

DEFAULT_PAIRED_DELTA_FIELDS = [
    "final_exploitability",
    "intermediate_best_exploitability",
    "intermediate_exploitability_normalised_auc_nodes",
    "final_policy_value_error",
    "final_policy_loss",
    "last_intermediate_regret_loss_player_0",
    "last_intermediate_regret_loss_player_1",
    "last_intermediate_value_loss",
    "last_intermediate_value_test_loss",
    "peak_rss_mb",
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


def _parse_seeds(value: Optional[str], default_seeds: Sequence[int]) -> List[int]:
    if not value:
        return list(default_seeds)
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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), sort_keys=True))
        f.write("\n")


def _append_partial_result(run_dir: Path, result: Mapping[str, Any]) -> None:
    _append_jsonl(run_dir / "partial_variant_seed_summary.jsonl", result["summary"])
    for row in result["curves"]:
        _append_jsonl(run_dir / "partial_checkpoint_curves.jsonl", row)


def _write_failures(run_dir: Path, failures: Sequence[Mapping]) -> None:
    _write_json(run_dir / "failed_runs.json", list(failures))


def _configure_logging(run_dir: Path, logger_name: str, verbose: bool) -> logging.Logger:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    return logging.getLogger(logger_name)


def _build_base_config(args, default_config: Mapping) -> Dict:
    config = deepcopy(default_config)
    _apply_cli_overrides(config, args)
    return config


def _apply_cli_overrides(config: Dict, args) -> Dict:
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "check_exploitability_every": args.evaluation_interval,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "learning_rate": args.learning_rate,
        "memory_capacity": args.memory_capacity,
        "regret_replay_rare_history_quota": args.regret_replay_rare_history_quota,
        "regret_replay_weight_floor": args.regret_replay_weight_floor,
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
        "regret_target_fixed_scale": args.regret_target_fixed_scale,
        "regret_target_ema_decay": args.regret_target_ema_decay,
        "compute_exploitability": args.compute_exploitability,
        "save_final_checkpoints": args.save_final_checkpoints,
        "baseline_variant_id": args.baseline_variant_id,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _selected_variants(args, variants: Sequence[Mapping]) -> List[Dict]:
    selected_ids = parse_variant_ids(args.variant_ids, list(variants))
    lookup = variant_lookup(list(variants))
    unknown = [variant_id for variant_id in selected_ids if variant_id not in lookup]
    if unknown:
        raise ValueError(f"Unknown variant id(s): {unknown}")
    return [lookup[variant_id] for variant_id in selected_ids]


def _augment_summary(
    summary: Dict,
    config: Mapping,
    summary_hp_fields: Sequence[str],
    extra_summary_fields: Mapping[str, str],
) -> Dict:
    summary["variant_description"] = config.get("variant_description", "")
    for summary_key, config_key in extra_summary_fields.items():
        summary[summary_key] = json_safe(config.get(config_key))
    for key in summary_hp_fields:
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


def _aggregate_by_variant(rows: Sequence[Mapping], variants: Sequence[Mapping]) -> Dict:
    aggregate = {}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        aggregate[variant_id] = _numeric_summary(variant_rows)
    return aggregate


def _paired_rows(
    rows: Sequence[Mapping],
    baseline_variant_id: str,
    paired_delta_fields: Sequence[str],
) -> List[Dict]:
    by_variant_seed = {
        (str(row["variant_id"]), int(row["seed"])): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    variants = sorted({str(row["variant_id"]) for row in rows})
    paired = []
    for seed in seeds:
        baseline = by_variant_seed.get((baseline_variant_id, seed))
        if baseline is None:
            continue
        for variant_id in variants:
            if variant_id == baseline_variant_id:
                continue
            treatment = by_variant_seed.get((variant_id, seed))
            if treatment is None:
                continue
            row = {"seed": seed, "variant_id": variant_id}
            for field in paired_delta_fields:
                row[f"delta_{field}_vs_baseline"] = (
                    float(treatment.get(field, np.nan))
                    - float(baseline.get(field, np.nan))
                )
            paired.append(row)
    return paired


def _paired_summary(
    rows: Sequence[Mapping],
    paired_delta_fields: Sequence[str],
) -> Dict:
    variants = sorted({str(row["variant_id"]) for row in rows})
    summary = {}
    for variant_id in variants:
        variant_rows = [row for row in rows if row["variant_id"] == variant_id]
        summary[variant_id] = _numeric_summary(variant_rows)
        for field in paired_delta_fields:
            delta_key = f"delta_{field}_vs_baseline"
            values = np.asarray(
                [row.get(delta_key, np.nan) for row in variant_rows],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            summary[variant_id][f"fraction_improved_{field}"] = (
                float(np.mean(finite < 0.0)) if finite.size else np.nan
            )
    return summary


def _mean_by_checkpoint(rows: Sequence[Mapping], variant_id: str, y_key: str):
    variant_rows = [
        row for row in rows
        if row["variant_id"] == variant_id
        and not bool(row.get("is_final_policy_evaluation", False))
    ]
    checkpoint_indices = sorted({int(row["checkpoint_index"]) for row in variant_rows})
    xs, means, ses = [], [], []
    for checkpoint_index in checkpoint_indices:
        checkpoint_rows = [
            row for row in variant_rows
            if int(row["checkpoint_index"]) == checkpoint_index
        ]
        x_vals = np.asarray(
            [row.get("nodes_touched", np.nan) for row in checkpoint_rows],
            dtype=float,
        )
        y_vals = np.asarray(
            [row.get(y_key, np.nan) for row in checkpoint_rows],
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


def _variant_label(variant: Mapping) -> str:
    return str(variant.get("variant_label") or variant["variant_id"])


def _plot_final_exploitability(
    run_dir: Path,
    summary_rows: Sequence[Mapping],
    variants: Sequence[Mapping],
    plot_title_prefix: str,
) -> None:
    labels = [_variant_label(variant) for variant in variants]
    means, ses = [], []
    for variant in variants:
        variant_id = str(variant["variant_id"])
        values = np.asarray(
            [
                row.get("final_exploitability", np.nan)
                for row in summary_rows
                if row["variant_id"] == variant_id
            ],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        means.append(float(np.mean(finite)) if finite.size else np.nan)
        ses.append(
            float(np.std(finite, ddof=1) / np.sqrt(finite.size))
            if finite.size > 1
            else 0.0
        )

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=ses, capsize=4)
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, f"{plot_title_prefix}: final exploitability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_variant.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(
    run_dir: Path,
    curve_rows: Sequence[Mapping],
    variants: Sequence[Mapping],
    plot_title_prefix: str,
    extra_plot_specs: Sequence = (),
) -> None:
    plot_specs = [
        (
            "exploitability",
            "Exploitability (NashConv / 2)",
            f"{plot_title_prefix}: exploitability by nodes touched",
            "exploitability_by_nodes.png",
            NASH_EXPLOITABILITY_TARGET,
            NASH_EXPLOITABILITY_TARGET_LABEL,
        ),
        (
            "average_policy_value",
            "Average-policy value",
            f"{plot_title_prefix}: average-policy value by nodes touched",
            "average_policy_value_by_nodes.png",
            LEDUC_GAME_VALUE_PLAYER_0,
            AVERAGE_POLICY_VALUE_TARGET_LABEL,
        ),
        (
            "policy_value_error",
            "Absolute policy-value error",
            f"{plot_title_prefix}: policy-value error by nodes touched",
            "policy_value_error_by_nodes.png",
            None,
            None,
        ),
    ] + list(extra_plot_specs)
    for y_key, ylabel, title, filename, target, target_label in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for variant in variants:
            variant_id = str(variant["variant_id"])
            xs, means, ses = _mean_by_checkpoint(curve_rows, variant_id, y_key)
            if xs.size == 0:
                continue
            plotted = True
            ax.plot(xs, means, marker="o", linewidth=1.8, label=_variant_label(variant))
            if np.any(ses > 0):
                ax.fill_between(xs, means - ses, means + ses, alpha=0.15)
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
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def _build_arg_parser(
    *,
    description: str,
    output_root: str,
    baseline_variant_id: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default=output_root)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--variant-ids", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--memory-capacity", type=int, default=None)
    parser.add_argument("--regret-replay-rare-history-quota", type=int, default=None)
    parser.add_argument("--regret-replay-weight-floor", type=float, default=None)
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
    parser.add_argument("--regret-target-fixed-scale", type=float, default=None)
    parser.add_argument("--regret-target-ema-decay", type=float, default=None)
    parser.add_argument("--baseline-variant-id", default=baseline_variant_id)
    parser.add_argument("--compute-exploitability", type=_str2bool, default=None)
    parser.add_argument("--save-final-checkpoints", type=_str2bool, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument(
        "--disable-subprocess-isolation",
        action="store_true",
        help=(
            "Run all variant-seed trainings in the parent process. By default "
            "each independent training runs in a fresh Python worker so "
            "TensorFlow state is released between runs."
        ),
    )
    parser.add_argument("--worker-input-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _worker_stem(variant_id: str, seed: int) -> str:
    return f"{_safe_stem(str(variant_id))}_seed_{int(seed)}"


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


def _run_seed_variant_subprocess(
    seed: int,
    config: Dict[str, Any],
    run_dir: Path,
    *,
    worker_module: str,
) -> Dict[str, Any]:
    stem = _worker_stem(str(config["variant_id"]), seed)
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
        worker_module,
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


def _run_seed_variant(
    seed: int,
    config: Dict[str, Any],
    run_dir: Path,
    *,
    subprocess_isolation_enabled: bool,
    worker_module: str,
) -> Dict[str, Any]:
    if subprocess_isolation_enabled:
        return _run_seed_variant_subprocess(
            seed,
            config,
            run_dir,
            worker_module=worker_module,
        )
    return run_single_seed_variant(seed, config, export_dir=run_dir)


def run_variant_ablation(
    argv: Optional[List[str]],
    *,
    default_config: Mapping,
    default_seeds: Sequence[int],
    variants: Sequence[Mapping],
    baseline_variant_id: str,
    output_root: str,
    description: str,
    logger_name: str,
    progress_label: str,
    plot_title_prefix: str,
    worker_module: str,
    summary_hp_fields: Sequence[str] = DEFAULT_SUMMARY_HP_FIELDS,
    paired_delta_fields: Sequence[str] = DEFAULT_PAIRED_DELTA_FIELDS,
    extra_summary_fields: Mapping[str, str] = (),
    additional_paired_baseline_ids: Sequence[str] = (),
    extra_curve_plot_specs: Sequence = (),
) -> int:
    parser = _build_arg_parser(
        description=description,
        output_root=output_root,
        baseline_variant_id=baseline_variant_id,
    )
    args = parser.parse_args(argv)
    if args.worker_input_json or args.worker_output_json:
        if not args.worker_input_json or not args.worker_output_json:
            parser.error("--worker-input-json and --worker-output-json must be used together")
        return _run_worker(args.worker_input_json, args.worker_output_json)

    subprocess_isolation_enabled = not args.disable_subprocess_isolation
    base_config = _build_base_config(args, default_config)
    seeds = _parse_seeds(args.seeds, default_seeds)
    selected_variants = _selected_variants(args, variants)
    selected_ids = [str(variant["variant_id"]) for variant in selected_variants]
    active_baseline_id = str(base_config.get("baseline_variant_id", baseline_variant_id))
    if active_baseline_id not in selected_ids:
        active_baseline_id = selected_ids[0]
        base_config["baseline_variant_id"] = active_baseline_id

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    logger = _configure_logging(run_dir, logger_name, args.verbose)

    metadata = {
        "base_config": base_config,
        "seeds": seeds,
        "selected_variant_ids": selected_ids,
        "baseline_variant_id": active_baseline_id,
        "additional_paired_baseline_ids": list(additional_paired_baseline_ids),
        "available_variants": list(variants),
        "subprocess_isolation_enabled": bool(subprocess_isolation_enabled),
        "incremental_outputs": {
            "partial_variant_seed_summary_jsonl": "partial_variant_seed_summary.jsonl",
            "partial_checkpoint_curves_jsonl": "partial_checkpoint_curves.jsonl",
            "worker_results_dir": "worker_results",
            "worker_logs_dir": "worker_logs",
        },
    }
    _write_json(run_dir / "experiment_metadata.json", metadata)

    logger.info("Export directory: %s", run_dir.resolve())
    logger.info("Running seeds: %s", seeds)
    logger.info("Selected variants: %s", selected_ids)
    logger.info("Baseline variant: %s", active_baseline_id)
    logger.info("Subprocess isolation enabled: %s", subprocess_isolation_enabled)

    results = []
    failures = []
    total = len(seeds) * len(selected_variants)
    with tqdm(total=total, desc=progress_label) as progress:
        for seed in seeds:
            for variant in selected_variants:
                config = make_variant_config(base_config, variant)
                config = make_variant_config(_apply_cli_overrides(config, args), {})
                try:
                    result = _run_seed_variant(
                        seed,
                        config,
                        run_dir,
                        subprocess_isolation_enabled=subprocess_isolation_enabled,
                        worker_module=worker_module,
                    )
                    result["summary"] = _augment_summary(
                        result["summary"],
                        config,
                        summary_hp_fields,
                        dict(extra_summary_fields),
                    )
                    _append_partial_result(run_dir, result)
                    results.append(result)
                    logger.info(
                        "%s seed %s final exploitability %.6f",
                        config["variant_id"],
                        seed,
                        result["summary"]["final_exploitability"],
                    )
                except Exception as exc:  # pragma: no cover - operational robustness
                    failure = {
                        "variant_id": variant["variant_id"],
                        "seed": int(seed),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    _write_failures(run_dir, failures)
                    logger.error(
                        "Variant %s seed %s failed: %s",
                        variant["variant_id"],
                        seed,
                        exc,
                    )
                    if not args.continue_on_error:
                        break
                finally:
                    cleanup_tensorflow_memory()
                    progress.update(1)
            if failures and not args.continue_on_error:
                break

    if failures:
        _write_failures(run_dir, failures)

    if not results:
        logger.error("No variants completed successfully.")
        return 1

    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    paired_rows = _paired_rows(summary_rows, active_baseline_id, paired_delta_fields)
    aggregate = _aggregate_by_variant(summary_rows, selected_variants)
    paired_summary = _paired_summary(paired_rows, paired_delta_fields)
    additional_paired_summaries = {}
    for contrast_baseline_id in additional_paired_baseline_ids:
        contrast_baseline_id = str(contrast_baseline_id)
        if contrast_baseline_id not in selected_ids:
            continue
        contrast_rows = _paired_rows(
            summary_rows,
            contrast_baseline_id,
            paired_delta_fields,
        )
        contrast_summary = _paired_summary(contrast_rows, paired_delta_fields)
        safe_baseline_id = _safe_stem(contrast_baseline_id)
        _write_csv(
            run_dir / f"paired_differences_vs_{safe_baseline_id}.csv",
            contrast_rows,
        )
        _write_json(
            run_dir / f"paired_difference_summary_vs_{safe_baseline_id}.json",
            contrast_summary,
        )
        additional_paired_summaries[contrast_baseline_id] = contrast_summary

    _write_csv(run_dir / "variant_seed_summary.csv", summary_rows)
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows)
    _write_csv(run_dir / "paired_differences_vs_baseline.csv", paired_rows)
    _write_json(run_dir / "aggregate_summary.json", aggregate)
    _write_json(run_dir / "paired_difference_summary.json", paired_summary)
    _write_json(
        run_dir / "summary.json",
        {
            "variant_seed_summary": summary_rows,
            "aggregate_summary": aggregate,
            "paired_difference_summary": paired_summary,
            "additional_paired_difference_summaries": additional_paired_summaries,
            "failed_runs": failures,
        },
    )

    _plot_final_exploitability(
        run_dir,
        summary_rows,
        selected_variants,
        plot_title_prefix,
    )
    _plot_curves(
        run_dir,
        curve_rows,
        selected_variants,
        plot_title_prefix,
        extra_plot_specs=extra_curve_plot_specs,
    )

    logger.info("Saved ablation outputs to: %s", run_dir.resolve())
    return 0 if not failures else 2
