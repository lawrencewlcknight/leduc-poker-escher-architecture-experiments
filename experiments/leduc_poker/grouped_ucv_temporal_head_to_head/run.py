"""Replay saved Experiment 35 neural policies; never retrain or refit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from .config import (
    CANDIDATE_ID, EXPECTED_CONFIG, EXPERIMENT_ID, EXPERIMENT_NAME,
    EXPLOITABILITY_TOLERANCE, HOURS, SEEDS, SMOKE_HOURS, SMOKE_SEEDS,
    checkpoint_id, source_task,
)

LOG = logging.getLogger(__name__)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    # Shared conversion removes NaNs in one-seed smoke statistics.
    from escher_poker.json_utils import json_safe
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_inventory(source, seeds, hours):
    """Require every requested checkpoint exactly once; never drop failed seeds."""
    rows = read_csv(Path(source) / "analysis/snapshot_inventory.csv")
    chosen = {}
    wanted = {(seed, checkpoint_id(hour)) for seed in seeds for hour in hours}
    for row in rows:
        key = (int(row["seed"]), row["checkpoint_id"])
        if key not in wanted:
            continue
        if key in chosen:
            raise ValueError(f"Duplicate snapshot: {key}")
        expected_name = f"{CANDIDATE_ID}_seed_{key[0]}_{key[1]}.pkl"
        if row["candidate_id"] != CANDIDATE_ID or row["filename"] != expected_name:
            raise ValueError(f"Wrong candidate or unsafe filename: {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid checksum: {key}")
        chosen[key] = row
    if set(chosen) != wanted:
        raise ValueError(f"Missing required checkpoints: {sorted(wanted - set(chosen))}")
    return [chosen[(seed, checkpoint_id(hour))] for seed in seeds for hour in hours]


def fetch_source(source, source_uri, seeds, hours):
    """Download only small analysis files and playable snapshots, never reservoirs."""
    if not re.fullmatch(r"gs://[A-Za-z0-9._-]+/[A-Za-z0-9/_-]+", source_uri):
        raise ValueError("Source must be a gs://bucket/run-prefix without wildcards")
    source = Path(source)
    (source / "analysis").mkdir(parents=True, exist_ok=True)
    (source / "snapshots").mkdir(exist_ok=True)
    for name in ("snapshot_inventory.csv", "checkpoint_policy_metrics.csv",
                 "aggregate_summary.json"):
        subprocess.run(["gcloud", "storage", "cp", f"{source_uri}/analysis/{name}",
                        str(source / "analysis" / name)], check=True)
    for seed in seeds:
        rows = [r for r in selected_inventory(source, seeds, hours)
                if int(r["seed"]) == seed]
        missing = [r for r in rows if not (source / "snapshots" / r["filename"]).is_file()
                   or digest(source / "snapshots" / r["filename"]) != r["sha256"]]
        if missing:
            subprocess.run(
                ["gcloud", "storage", "cp"] +
                [f"{source_uri}/workers/{source_task(seed)}/snapshots/{r['filename']}"
                 for r in missing] + [str(source / "snapshots") + "/"], check=True)
    write_json(source / "source_origin.json", {"source_uri": source_uri})


def validate_snapshot(path, row):
    """Check the hash before unpickling our own trusted experiment artefacts."""
    from escher_poker.policy_snapshots import load_pickle
    if digest(path) != row["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {path}")
    payload = load_pickle(path)
    for key in ("candidate_id", "checkpoint_id", "repository_commit"):
        if payload.get(key) != row[key]:
            raise ValueError(f"Snapshot {key} mismatch: {path}")
    for key in ("seed", "nodes_touched", "completed_iteration"):
        if int(payload[key]) != int(row[key]):
            raise ValueError(f"Snapshot {key} mismatch: {path}")
    hour = float(row["checkpoint_target_active_hours"])
    if (payload.get("game") != "leduc_poker"
            or payload.get("framework") != "pytorch"
            or payload.get("checkpoint_type") != "active_time"
            or payload.get("checkpoint_target_active_hours") != hour
            or abs(float(payload["active_seconds"]) - float(row["active_seconds"])) > 1e-6
            or float(payload["active_seconds"]) < 3600 * hour):
        raise ValueError(f"Invalid game, framework or time provenance: {path}")
    config = payload["frozen_config"]
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"Not the confirmed grouped candidate: {key}={config.get(key)!r}")
    if payload["policy_network_layers"] != [136, 136, 136]:
        raise ValueError("Wrong saved policy-network architecture")
    return payload


def source_records(source, seeds, hours):
    rows = selected_inventory(source, seeds, hours)
    contract = json.loads((Path(source) / "analysis/aggregate_summary.json").read_text())["contract"]
    if (contract["experiment_id"] != 35 or contract["candidate_id"] != CANDIDATE_ID
            or tuple(contract["production_seeds"]) != SEEDS):
        raise ValueError("Source aggregate is not the five-seed grouped confirmation")
    for key, expected in EXPECTED_CONFIG.items():
        if contract["candidate_config"].get(key) != expected:
            raise ValueError(f"Source aggregate configuration mismatch: {key}")
    references = {}
    for record in read_csv(Path(source) / "analysis/checkpoint_policy_metrics.csv"):
        if record["policy_id"] == CANDIDATE_ID:
            key = (int(record["seed"]), record["checkpoint_id"])
            if key in references:
                raise ValueError(f"Duplicate source diagnostic: {key}")
            references[key] = record
    configs = []
    for row in rows:
        # Accept the compact download layout or the original workers layout.
        path = Path(source) / "snapshots" / row["filename"]
        if not path.exists():
            path = Path(source) / "workers" / source_task(int(row["seed"])) / "snapshots" / row["filename"]
        payload = validate_snapshot(path, row)
        configs.append(payload["frozen_config"])
        ref = references[(int(row["seed"]), row["checkpoint_id"])]
        if (int(ref["nodes_touched"]) != int(row["nodes_touched"])
                or int(ref["completed_iteration"]) != int(row["completed_iteration"])):
            raise ValueError("Source diagnostics disagree with snapshot metadata")
        row["local_path"] = str(path.resolve())
        row["source_exploitability"] = float(ref["exploitability"])
    if any(config != configs[0] for config in configs):
        raise ValueError("Source snapshots mix training configurations")
    for seed in seeds:
        trajectory = [r for r in rows if int(r["seed"]) == seed]
        for earlier, later in zip(trajectory, trajectory[1:]):
            if any(float(later[k]) <= float(earlier[k])
                   for k in ("active_seconds", "nodes_touched", "completed_iteration")):
                raise ValueError("Checkpoint trajectory is not strictly increasing")
    return rows


def evaluate_seed(rows):
    import numpy as np
    import pyspiel
    from open_spiel.python import policy
    from open_spiel.python.algorithms import exploitability
    from escher_poker.checkpoint_analysis import exact_seat_averaged_value_for_a
    from escher_poker.policy_snapshots import LoadedESCHERPolicy

    game = pyspiel.load_game("leduc_poker")
    policies, metrics, pairs = {}, [], []
    for row in rows:
        hour = int(float(row["checkpoint_target_active_hours"]))
        neural = LoadedESCHERPolicy(game, row["local_path"])
        tab = policy.tabular_policy_from_callable(game, neural.action_probabilities)
        probabilities = tab.action_probability_array
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError("Invalid policy probabilities")
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        exp = float(exploitability.nash_conv(game, tab) / 2)
        if abs(exp - row["source_exploitability"]) > EXPLOITABILITY_TOLERANCE:
            raise ValueError(f"Reloaded exploitability differs from source at {hour}h: "
                             f"{exp} vs {row['source_exploitability']}")
        self_ev = exact_seat_averaged_value_for_a(game, tab, tab)["A_EV_seat_averaged"]
        if abs(self_ev) > 1e-10:
            raise ValueError("Two-seat self-play EV must be zero")
        policies[hour] = tab
        metrics.append({
            "seed": int(row["seed"]), "checkpoint": hour,
            "actual_active_hours": float(row["active_seconds"]) / 3600,
            "nodes_touched": int(row["nodes_touched"]),
            "completed_iteration": int(row["completed_iteration"]),
            "exploitability": exp, "source_exploitability": row["source_exploitability"],
            "reload_absolute_error": abs(exp - row["source_exploitability"]),
            "self_play_seat_averaged_ev": self_ev,
        })
        LOG.info("Validated seed %s, %sh: exploitability %.8f", row["seed"], hour, exp)
    for earlier, later in itertools.combinations(policies, 2):
        values = exact_seat_averaged_value_for_a(game, policies[later], policies[earlier])
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Non-finite exact matchup")
        pairs.append({"seed": int(rows[0]["seed"]), "checkpoint_a": later,
                      "checkpoint_b": earlier, **values})
    return {"metrics": metrics, "pairs": pairs}


def validate_result(result, rows, hours):
    """Reject incomplete, duplicated or non-finite cached evaluations."""
    seed = int(rows[0]["seed"])
    metrics, pairs = result["metrics"], result["pairs"]
    if (len(metrics) != len(hours) or {r["checkpoint"] for r in metrics} != set(hours)
            or any(r["seed"] != seed for r in metrics + pairs)):
        raise ValueError("Incomplete or mixed-seed checkpoint results")
    expected = {(b, a) for a, b in itertools.combinations(hours, 2)}
    if (len(pairs) != len(expected)
            or {(r["checkpoint_a"], r["checkpoint_b"]) for r in pairs} != expected):
        raise ValueError("Incomplete or duplicated pairwise results")
    references = {int(float(r["checkpoint_target_active_hours"])): r for r in rows}
    for row in metrics:
        exp = row["exploitability"]
        if (not math.isfinite(exp)
                or abs(exp - references[row["checkpoint"]]["source_exploitability"]) > EXPLOITABILITY_TOLERANCE):
            raise ValueError("Cached exploitability differs from source")
    for row in pairs:
        values = [row[k] for k in ("A_EV_as_player0", "A_EV_as_player1", "A_EV_seat_averaged")]
        if (not all(math.isfinite(v) for v in values)
                or abs(values[2] - (values[0] + values[1]) / 2) > 1e-12):
            raise ValueError("Invalid cached two-seat expected value")


def analyse(results, hours, output):
    import numpy as np
    from experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.statistics import (
        build_inference_tables,
    )
    from .plots import plot_results
    output.mkdir(parents=True, exist_ok=True)
    metrics = [r for result in results for r in result["metrics"]]
    pairs = [r for result in results for r in result["pairs"]]
    seed_rows, inference, pair_inference = build_inference_tables(pairs, hours)
    for index, row in enumerate(inference):
        row["role"] = "primary" if index == 0 else "secondary_unadjusted"
    for name, rows in (
        ("checkpoint_metrics", metrics), ("head_to_head_pairwise", pairs),
        ("seed_summary", seed_rows), ("head_to_head_inference_summary", inference),
        ("head_to_head_pairwise_inference", pair_inference),
    ):
        write_csv(output / f"{name}.csv", rows)
    late_hours = tuple(h for h in hours if h >= 24)
    if len(late_hours) >= 2:
        late_seed, late_stats, late_pairs = build_inference_tables(pairs, late_hours)
        write_csv(output / "late_window_seed_summary.csv", late_seed)
        write_csv(output / "late_window_inference_summary.csv", late_stats)
        write_csv(output / "late_window_pairwise_inference.csv", late_pairs)
    aggregates = []
    for hour in hours:
        group = [r for r in metrics if r["checkpoint"] == hour]
        values = np.array([r["exploitability"] for r in group])
        aggregates.append({
            "checkpoint_target_active_hours": hour, "n_seeds": len(group),
            "mean_actual_active_hours": float(np.mean([r["actual_active_hours"] for r in group])),
            "mean_nodes_touched": float(np.mean([r["nodes_touched"] for r in group])),
            "mean_exploitability": float(values.mean()),
            "std_exploitability": float(values.std(ddof=1)) if len(values) > 1 else None,
            "sem_exploitability": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
        })
    write_csv(output / "checkpoint_summary.csv", aggregates)
    strengths, best = [], []
    for seed in sorted({r["seed"] for r in metrics}):
        candidates = []
        for hour in hours:
            own = next(r for r in metrics if r["seed"] == seed and r["checkpoint"] == hour)
            values = [
                r["A_EV_seat_averaged"] if r["checkpoint_a"] == hour else -r["A_EV_seat_averaged"]
                for r in pairs if r["seed"] == seed
                and hour in (r["checkpoint_a"], r["checkpoint_b"])
            ]
            candidates.append({**own, "mean_ev_vs_all_other_checkpoints": float(np.mean(values))})
        strengths.extend(candidates)
        best.append({
            "seed": seed, "final_checkpoint": hours[-1],
            "lowest_exploitability_checkpoint": min(candidates, key=lambda r: r["exploitability"])["checkpoint"],
            "strongest_round_robin_checkpoint": max(candidates, key=lambda r: r["mean_ev_vs_all_other_checkpoints"])["checkpoint"],
        })
    write_csv(output / "checkpoint_strength.csv", strengths)
    write_csv(output / "best_checkpoint_summary.csv", best)
    plot_results(metrics, pairs, pair_inference, hours, output)
    summary = {
        "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID, "seeds": sorted({r["seed"] for r in metrics}),
        "checkpoint_target_hours": list(hours),
        "num_snapshots": len(metrics), "num_unordered_same_seed_pairs": len(pairs),
        "num_seat_assignments": 2 * len(pairs), "inferential_unit": "training_seed",
        "primary": inference[0], "checkpoint_summary": aggregates,
        "late_window": "24--36h; secondary descriptive sensitivity analysis",
        "multiplicity": "Holm correction across all checkpoint-pair tests; secondary "
                        "aggregate and late-window tests are exploratory",
        "evidence_status": "retrospective evaluation of existing Experiment 35 policies, "
                           "not a new independently trained confirmation",
        "status": "complete",
    }
    write_json(output / "aggregate_summary.json", summary)
    return summary


def run(source, output, *, smoke=False):
    import torch
    torch.set_num_threads(1)
    seeds, hours = (SMOKE_SEEDS, SMOKE_HOURS) if smoke else (SEEDS, HOURS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS.json").exists():
        raise FileExistsError("Completed output exists; choose a new output root")
    rows = source_records(source, seeds, hours)
    origin = Path(source) / "source_origin.json"
    metadata = {
        "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID, "seeds": seeds, "hours": hours, "smoke": smoke,
        "source_origin": json.loads(origin.read_text()) if origin.exists() else str(Path(source).resolve()),
        "source_files": {name: digest(Path(source) / "analysis" / name) for name in
                         ("aggregate_summary.json", "snapshot_inventory.csv", "checkpoint_policy_metrics.csv")},
        "snapshots": [{k: v for k, v in r.items() if k != "local_path"} for r in rows],
        "expected_config": EXPECTED_CONFIG, "command": sys.argv,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "open_spiel", "scipy")},
        "training_performed": False, "policy_fitting_performed": False,
    }
    write_json(output / "experiment_metadata.json", metadata)
    started, results = time.perf_counter(), []
    # Each immutable per-seed result is resumable only for identical inputs and evaluator.
    for seed in seeds:
        seed_rows = [r for r in rows if int(r["seed"]) == seed]
        repository = Path(__file__).resolve().parents[3]
        identity = {"snapshots": [r["sha256"] for r in seed_rows],
                    "reference": metadata["source_files"], "hours": list(hours),
                    "evaluator_sha256": digest(Path(__file__)),
                    "shared_code": {name: digest(repository / name) for name in (
                        "escher_poker/policy_snapshots.py",
                        "escher_poker/checkpoint_analysis.py", "vr_deep_cfr/solver.py",
                    )}}
        path = output / "seeds" / f"seed_{seed}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["identity"] != identity:
                raise ValueError(f"Resume inputs changed for seed {seed}; use a new output root")
            result = saved["result"]
            LOG.info("Reusing completed seed %s", seed)
        else:
            result = evaluate_seed(seed_rows)
            validate_result(result, seed_rows, hours)
            write_json(path, {"identity": identity, "result": result})
        validate_result(result, seed_rows, hours)
        results.append(result)
    summary = analyse(results, hours, output / "analysis")
    write_json(output / "SUCCESS.json", {
        "status": "complete", "smoke": smoke, "num_seeds": len(seeds),
        "evaluation_seconds": time.perf_counter() - started,
        "summary": "analysis/aggregate_summary.json",
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "run", "smoke"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-uri")
    parser.add_argument("--smoke-only", action="store_true", help="Fetch only three real policies")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "fetch":
        if not args.source_uri:
            parser.error("fetch requires --source-uri")
        seeds, hours = (SMOKE_SEEDS, SMOKE_HOURS) if args.smoke_only else (SEEDS, HOURS)
        fetch_source(args.source_root, args.source_uri.rstrip("/"), seeds, hours)
    else:
        if not args.output_root:
            parser.error("run/smoke requires --output-root")
        summary = run(args.source_root, args.output_root, smoke=args.action == "smoke")
        print(json.dumps({k: summary[k] for k in
                          ("status", "num_snapshots", "num_unordered_same_seed_pairs")}, indent=2))


if __name__ == "__main__":
    main()
