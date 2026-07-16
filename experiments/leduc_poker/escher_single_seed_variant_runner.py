"""Reusable CLI runner for one-seed ESCHER architecture sweeps."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import sys
import traceback
from typing import Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XDG_CACHE_HOME", str((Path("outputs") / ".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib_cache").resolve()))
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
)
from experiments.leduc_poker.escher_variant_config_utils import (  # noqa: E402
    make_variant_config,
    parse_variant_ids,
    variant_lookup,
)

_LOGGER = logging.getLogger("escher_poker.experiment")

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


def _build_base_config(args, default_config: Dict) -> Dict:
    config = deepcopy(default_config)
    overrides = {
        "experiment_name": args.experiment_name,
        "num_iterations": args.iterations,
        "check_exploitability_every": args.evaluation_interval,
        "num_traversals": args.traversals,
        "num_val_fn_traversals": args.value_traversals,
        "policy_network_train_steps": args.policy_network_train_steps,
        "regret_network_train_steps": args.regret_network_train_steps,
        "value_network_train_steps": args.value_network_train_steps,
        "batch_size_regret": args.batch_size_regret,
        "batch_size_value": args.batch_size_value,
        "batch_size_average_policy": args.batch_size_average_policy,
        "policy_network_layers": _parse_int_tuple(args.policy_network_layers),
        "regret_network_layers": _parse_int_tuple(args.regret_network_layers),
        "value_network_layers": _parse_int_tuple(args.value_network_layers),
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def _configure_logging(run_dir: Path, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stdout)
    file_handler = logging.FileHandler(run_dir / "experiment.log", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _add_config_columns(result: Dict, config: Dict) -> None:
    result["summary"]["variant_description"] = config.get("variant_description", "")
    for key in SUMMARY_HP_FIELDS:
        result["summary"][f"hp_{key}"] = json_safe(config.get(key))


def _plot_final_exploitability(run_dir: Path, summary_rows: List[Dict], title: str) -> None:
    labels = [row["variant_id"] for row in summary_rows]
    y = [row["final_exploitability"] for row in summary_rows]
    y_intermediate = [row["intermediate_final_exploitability"] for row in summary_rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(9, 0.85 * len(labels)), 5))
    ax.bar(x - 0.18, y, width=0.36, label="Recomputed final policy")
    ax.bar(x + 0.18, y_intermediate, width=0.36, label="Last intermediate checkpoint")
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "final_exploitability_by_variant.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_curves(run_dir: Path, curve_rows: List[Dict]) -> None:
    rows = [row for row in curve_rows if not bool(row.get("is_final_policy_evaluation", False))]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in sorted({row["variant_id"] for row in rows}):
        variant_rows = sorted(
            [row for row in rows if row["variant_id"] == variant_id],
            key=lambda row: row["iteration"],
        )
        ax.plot(
            [row["iteration"] for row in variant_rows],
            [row["exploitability"] for row in variant_rows],
            marker="o",
            linewidth=1.8,
            label=variant_id,
        )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, "Intermediate exploitability checkpoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "intermediate_exploitability_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for variant_id in sorted({row["variant_id"] for row in rows}):
        variant_rows = sorted(
            [row for row in rows if row["variant_id"] == variant_id],
            key=lambda row: row["iteration"],
        )
        ax.plot(
            [row["iteration"] for row in variant_rows],
            [row["average_policy_value"] for row in variant_rows],
            marker="o",
            linewidth=1.8,
            label=variant_id,
        )
    ax.axhline(
        LEDUC_GAME_VALUE_PLAYER_0,
        color="black",
        linestyle="--",
        linewidth=1,
        label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value")
    set_chart_title(ax, "Intermediate average-policy value checkpoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_iteration.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _build_arg_parser(description: str, output_root: str, default_seed: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-root", default=output_root)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--variant-ids", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--traversals", type=int, default=None)
    parser.add_argument("--value-traversals", type=int, default=None)
    parser.add_argument("--policy-network-train-steps", type=int, default=None)
    parser.add_argument("--regret-network-train-steps", type=int, default=None)
    parser.add_argument("--value-network-train-steps", type=int, default=None)
    parser.add_argument("--batch-size-regret", type=int, default=None)
    parser.add_argument("--batch-size-value", type=int, default=None)
    parser.add_argument("--batch-size-average-policy", type=int, default=None)
    parser.add_argument("--policy-network-layers", default=None)
    parser.add_argument("--regret-network-layers", default=None)
    parser.add_argument("--value-network-layers", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(
    config_module,
    argv: Optional[List[str]] = None,
    *,
    description: str,
    output_root: str,
    progress_label: str,
    final_plot_title: str,
    unknown_label: str,
) -> int:
    args = _build_arg_parser(
        description,
        output_root,
        int(config_module.DEFAULT_SEED),
    ).parse_args(argv)
    base_config = _build_base_config(args, config_module.DEFAULT_CONFIG)
    variants = list(config_module.VARIANTS)
    selected_ids = parse_variant_ids(args.variant_ids, variants)
    lookup = variant_lookup(variants)
    unknown = [variant_id for variant_id in selected_ids if variant_id not in lookup]
    if unknown:
        raise ValueError(f"Unknown {unknown_label} variant id(s): {unknown}")

    run_dir = create_run_dir(args.output_root, base_config["experiment_name"])
    _configure_logging(run_dir, args.verbose)

    selected_variants = [lookup[variant_id] for variant_id in selected_ids]
    metadata = {
        "base_config": base_config,
        "seed": int(args.seed),
        "selected_variant_ids": selected_ids,
        "available_variants": variants,
    }
    with open(run_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metadata), f, indent=2)

    _LOGGER.info("Export directory: %s", run_dir.resolve())
    _LOGGER.info("Running seed: %s", args.seed)
    _LOGGER.info("Selected variants: %s", selected_ids)

    results = []
    failures = []
    for variant in tqdm(selected_variants, desc=progress_label):
        config = make_variant_config(base_config, variant)
        try:
            result = run_single_seed_variant(args.seed, config, export_dir=run_dir)
            _add_config_columns(result, config)
            results.append(result)
            _LOGGER.info(
                "%s final exploitability %.6f",
                config["variant_id"],
                result["summary"]["final_exploitability"],
            )
        except Exception as exc:  # pragma: no cover - operational robustness
            failure = {
                "variant_id": variant["variant_id"],
                "seed": int(args.seed),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _LOGGER.error("Variant %s failed: %s", variant["variant_id"], exc)
            if not args.continue_on_error:
                break
        finally:
            cleanup_tensorflow_memory()

    if failures:
        with open(run_dir / "failed_runs.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(failures), f, indent=2)

    if not results:
        _LOGGER.error("No variants completed successfully.")
        return 1

    summary_rows = [result["summary"] for result in results]
    curve_rows = [row for result in results for row in result["curves"]]
    _write_csv(run_dir / "variant_summary.csv", summary_rows)
    _write_csv(run_dir / "checkpoint_curves.csv", curve_rows)

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            json_safe({
                "variant_summary": summary_rows,
                "failed_runs": failures,
            }),
            f,
            indent=2,
        )

    _plot_final_exploitability(run_dir, summary_rows, final_plot_title)
    _plot_curves(run_dir, curve_rows)
    _LOGGER.info("Saved sweep outputs to: %s", run_dir.resolve())
    return 0 if not failures else 2
