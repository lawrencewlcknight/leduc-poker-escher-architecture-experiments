"""Run Experiment 4 and combine its curves with saved Experiment 2 results."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib_cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from escher_poker.chart_titles import set_chart_title  # noqa: E402
from escher_poker.constants import (  # noqa: E402
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)
from experiments.leduc_poker.adaptive_residual_predictive_escher import run as exp3  # noqa: E402

from .config import (  # noqa: E402
    ADAPTIVE_CONFIG,
    ALGORITHM_ID,
    ALGORITHM_LABEL,
    DEFAULT_SEEDS,
    EXPERIMENT_2_NODE_TARGETS,
    EXPERIMENT_2_SOURCE,
    REFERENCE_CURVE_ROWS,
    REFERENCE_CURVES,
    REFERENCE_CURVES_SHA256,
)


LOGGER = logging.getLogger("adaptive_residual_predictive_escher_5x_nodes")

REFERENCE_ALGORITHM_IDS = (
    "escher_exp28",
    "vr_deep_dcfr_plus",
    "vr_deep_pdcfr_plus",
)
ALGORITHMS = {
    "escher_exp28": {"algorithm_label": "ESCHER (Experiment 28, 5x nodes)"},
    "vr_deep_dcfr_plus": {"algorithm_label": "VR-DeepDCFR+"},
    "vr_deep_pdcfr_plus": {"algorithm_label": "VR-DeepPDCFR+"},
    ALGORITHM_ID: {"algorithm_label": ALGORITHM_LABEL},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reference_curves(path: Path) -> List[Dict[str, Any]]:
    """Load and validate the immutable Experiment 2 checkpoint curves."""

    digest = _sha256(path)
    if digest != REFERENCE_CURVES_SHA256:
        raise ValueError(
            "Experiment 2 reference checksum mismatch: "
            f"expected {REFERENCE_CURVES_SHA256}, found {digest}"
        )
    rows = exp3._load_reference_curves(path)
    if len(rows) != REFERENCE_CURVE_ROWS:
        raise ValueError(
            f"Expected {REFERENCE_CURVE_ROWS} Experiment 2 rows, found {len(rows)}"
        )
    if {int(row["seed"]) for row in rows} != set(DEFAULT_SEEDS):
        raise ValueError("Experiment 2 reference curves must contain seeds 0, 1 and 2")

    for row in rows:
        row["is_initial_policy_evaluation"] = exp3._parse_bool(
            row.get("is_initial_policy_evaluation", False)
        )
        row["result_source"] = "saved_experiment_2"

    for seed, target in EXPERIMENT_2_NODE_TARGETS.items():
        escher_rows = [
            row
            for row in rows
            if row["algorithm_id"] == "escher_exp28" and int(row["seed"]) == seed
        ]
        final_rows = [row for row in escher_rows if row["is_final_policy_evaluation"]]
        if len(final_rows) != 1 or int(final_rows[0]["nodes_touched"]) != target:
            raise ValueError(
                f"Experiment 2 seed {seed} ESCHER endpoint does not match {target}"
            )
    return rows


def _run_adaptive(seed: int, config: Dict[str, Any], target_nodes: int):
    """Run the unchanged Experiment 3 architecture at the longer horizon."""

    result = exp3._run_adaptive(seed, config, target_nodes)
    result["summary"]["result_source"] = "experiment_4_new_run"
    for row in result["curves"]:
        row["result_source"] = "experiment_4_new_run"
    return result


def _run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = _run_adaptive(
        int(payload["seed"]),
        payload["config"],
        int(payload["target_nodes_touched"]),
    )
    exp3._write_json(output_path, result)
    return 0


def _run_subprocess(run_dir, seed, config, target_nodes):
    stem = f"{ALGORITHM_ID}_5x_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    output_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    exp3._write_json(
        input_path,
        {"seed": seed, "config": config, "target_nodes_touched": target_nodes},
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run",
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
            env={**exp3.os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"{stem} failed; see {log_path}")
    with open(output_path, encoding="utf-8") as handle:
        return json.load(handle)


def _reference_final_rows(reference_rows):
    final_rows = []
    for algorithm_id in REFERENCE_ALGORITHM_IDS:
        for seed in DEFAULT_SEEDS:
            candidates = [
                row
                for row in reference_rows
                if row["algorithm_id"] == algorithm_id and int(row["seed"]) == seed
            ]
            explicit = [row for row in candidates if row["is_final_policy_evaluation"]]
            if candidates:
                final_rows.append((explicit or candidates)[-1])
    return final_rows


def _reference_summaries(reference_rows):
    return [
        {
            "algorithm_id": row["algorithm_id"],
            "algorithm_label": row["algorithm_label"],
            "seed": int(row["seed"]),
            "final_exploitability": float(row["exploitability"]),
            "final_policy_value": float(row["average_policy_value"]),
            "final_policy_value_error": float(row["policy_value_error"]),
            "final_nodes_touched": float(row["nodes_touched"]),
            "final_wall_clock_seconds": float(row["wall_clock_seconds"]),
            "result_source": "saved_experiment_2",
        }
        for row in _reference_final_rows(reference_rows)
    ]


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
            field: exp3._stats(float(row.get(field, np.nan)) for row in rows)
            for field in sorted(numeric_fields)
            if field != "seed"
        }
    return result


def _paired_differences(summary_rows):
    indexed = {(row["algorithm_id"], int(row["seed"])): row for row in summary_rows}
    rows = []
    for baseline_id in REFERENCE_ALGORITHM_IDS:
        for seed in DEFAULT_SEEDS:
            baseline = indexed.get((baseline_id, seed))
            candidate = indexed.get((ALGORITHM_ID, seed))
            if baseline is None or candidate is None:
                continue
            rows.append(
                {
                    "baseline_algorithm_id": baseline_id,
                    "baseline_algorithm_label": ALGORITHMS[baseline_id][
                        "algorithm_label"
                    ],
                    "seed": seed,
                    "exploitability_difference": (
                        candidate["final_exploitability"]
                        - baseline["final_exploitability"]
                    ),
                    "nodes_difference": (
                        candidate["final_nodes_touched"]
                        - baseline["final_nodes_touched"]
                    ),
                }
            )
    return rows


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
        x, mean, se = exp3._mean_curve(curve_rows, algorithm_id)
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2.2,
            color=colors[algorithm_id],
            label=spec["algorithm_label"],
        )
        ax.fill_between(x, mean - se, mean + se, color=colors[algorithm_id], alpha=0.14)
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        color="black",
        linestyle="--",
        linewidth=1,
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 4 adaptive predictive ESCHER vs Experiment 2")
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
        stats = exp3._stats(values)
        labels.append(spec["algorithm_label"])
        means.append(stats["mean"])
        ses.append(stats["se"])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(np.arange(len(labels)), means, yerr=ses, capsize=5)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Final exploitability (NashConv / 2)")
    set_chart_title(ax, "Experiment 4 and Experiment 2: final exploitability")
    fig.tight_layout()
    fig.savefig(
        run_dir / "combined_final_exploitability.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _parse_seeds(value):
    if value is None:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/adaptive_residual_predictive_escher_5x_nodes",
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
    if any(seed not in EXPERIMENT_2_NODE_TARGETS for seed in seeds):
        raise ValueError("Experiment 4 supports paired Experiment 2 seeds 0, 1 and 2")
    config = deepcopy(ADAPTIVE_CONFIG)
    exp3._apply_overrides(args, config)
    reference_rows = _load_reference_curves(args.reference_curves)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.output_root)
        / f"adaptive_residual_predictive_escher_5x_nodes_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    targets = {
        seed: int(args.target_nodes or EXPERIMENT_2_NODE_TARGETS[seed])
        for seed in seeds
    }
    metadata = {
        "experiment_id": 4,
        "seeds": seeds,
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "adaptive_config": config,
        "paired_node_targets": targets,
        "experiment_2_source": EXPERIMENT_2_SOURCE,
        "reference_curves_file": str(args.reference_curves),
        "reference_curves_sha256": _sha256(args.reference_curves),
        "protocol": {
            "architecture": (
                "The adaptive architecture and all learning settings are unchanged "
                "from Experiment 3; only the stopping horizon changes."
            ),
            "baseline_reuse": (
                "All ESCHER and VR curves are immutable saved Experiment 2 results; "
                "only the adaptive algorithm is trained."
            ),
            "node_matching": (
                "The adaptive run stops after the first complete outer iteration "
                "crossing the paired Experiment 2 ESCHER node total."
            ),
        },
    }
    exp3._write_json(run_dir / "experiment_metadata.json", metadata)

    results, failures = [], []
    for seed in seeds:
        try:
            LOGGER.info("Running adaptive solver seed %s, target %s", seed, targets[seed])
            result = _run_subprocess(run_dir, seed, config, targets[seed])
            results.append(result)
            exp3._write_json(run_dir / "partial_results.json", results)
        except Exception as exc:  # pragma: no cover - operational path
            failures.append(
                {"seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
            )
            exp3._write_json(run_dir / "failed_runs.json", failures)
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

    exp3._write_csv(run_dir / "adaptive_5x_seed_summary.csv", new_summaries)
    exp3._write_csv(run_dir / "adaptive_5x_checkpoint_curves.csv", new_curves)
    exp3._write_csv(run_dir / "combined_checkpoint_curves.csv", combined_curves)
    exp3._write_csv(run_dir / "combined_seed_summary.csv", combined_summaries)
    exp3._write_csv(run_dir / "paired_differences.csv", paired)
    exp3._write_json(run_dir / "aggregate_summary.json", aggregate)
    exp3._write_json(
        run_dir / "summary.json",
        {
            "adaptive_5x_seed_summary": new_summaries,
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
