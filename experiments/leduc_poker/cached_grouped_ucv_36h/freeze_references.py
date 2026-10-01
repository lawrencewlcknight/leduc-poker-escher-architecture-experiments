"""Reproducibly freeze small historical analyses; never load training states.

Usage: python -m ...freeze_references --exp35-analysis DIR --sdcfr-analysis DIR
The resulting checksum must be explicitly reviewed and pinned in config.py.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def build(exp35, sdcfr):
    sources = {}
    def read(root, name):
        path = root / name
        sources[str(path.name)] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    rows = read(exp35, "all_algorithm_checkpoint_policy_metrics.csv")
    endpoints = read(exp35, "checkpoint_policy_metrics.csv")
    selected = [r for r in endpoints if r["policy_id"] == "grouped_wide_ucv"]
    # Existing performance file contains the two-hour historical UCV/Deep CFR cohorts.
    output = []
    for row in rows:
        output.append({"algorithm_id": row["algorithm_id"], "label": row["algorithm_label"],
            "seed": int(row["seed"]), "checkpoint_id": row["checkpoint_id"],
            "hours": float(row["checkpoint_target_active_hours"]),
            "actual_hours": float(row["actual_active_hours"]),
            "nodes": int(row["nodes_touched"]), "iteration": int(row["completed_iteration"]),
            "exploitability": float(row["exploitability"])})
    node_endpoints = []
    for row in selected:
        if row["checkpoint_type"] == "nodes":
            node_endpoints.append({"algorithm_id": "grouped_wide_ucv", "seed": int(row["seed"]),
                "nodes": int(row["nodes_touched"]), "iteration": int(row["completed_iteration"]),
                "actual_hours": float(row["actual_active_hours"]),
                "exploitability": float(row["exploitability"])})
    for row in read(sdcfr, "checkpoint_seed_metrics.csv"):
        h = float(row["scheduled_training_hours"]) if row["scheduled_training_hours"] else None
        is_time = h in range(2, 37, 2) and row["checkpoint_kind"] == "scheduled_time"
        is_node = row["is_node_15m_endpoint"].lower() == "true"
        if not is_time and not is_node:
            continue
        record = {"algorithm_id": "sd_cfr", "label": "SD-CFR (uniform)",
            "seed": int(row["seed"]), "checkpoint_id": f"time_{int(h):02d}h" if is_time else "node_15m",
            "hours": h, "actual_hours": float(row["training_hours"]),
            "nodes": int(row["nodes_touched"]), "iteration": int(row["iteration"]),
            "exploitability": float(row["sd_cfr_uniform_exploitability"])}
        (output if is_time else node_endpoints).append(record)
    for alg in {r["algorithm_id"] for r in output}:
        subset = [r for r in output if r["algorithm_id"] == alg]
        if len(subset) != 90 or len({(r["seed"], r["hours"]) for r in subset}) != 90:
            raise ValueError(f"Incomplete/duplicated reference: {alg}")
    milestones = {str(r["seed"]): r for r in output
                  if r["algorithm_id"] == "grouped_wide_ucv" and r["hours"] == 36}
    return {"source_files": sources,
        "source_roots": {
            "exp35": "gs://clever-overview-399515-leduc-poker-dream-results/exp35-confirm-20260916-011231/analysis",
            "sdcfr": "gs://clever-overview-399515-leduc-poker-results/exp29-sdcfr36h-20260920-233546/analysis"},
        "trajectories": output, "node_endpoints": node_endpoints, "milestones": milestones}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp35-analysis", type=Path, required=True)
    p.add_argument("--sdcfr-analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("frozen_reference.json"))
    a = p.parse_args()
    a.output.write_text(json.dumps(build(a.exp35_analysis, a.sdcfr_analysis), indent=2, sort_keys=True) + "\n")
    print(hashlib.sha256(a.output.read_bytes()).hexdigest())
