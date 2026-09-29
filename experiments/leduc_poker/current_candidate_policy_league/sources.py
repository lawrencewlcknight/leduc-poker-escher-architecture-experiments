"""Explicit, frozen source selection. Never select by observed performance."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

EXPERIMENT_ID = 46
DEEP_REPO_REF = "1669e5af4cbc88c648626148fd9c395c2e5d4583"
COMMON_SEEDS = (1234, 2025, 31415, 27182, 16180)
UCV_SEEDS = (470892, 385626, 145871, 902492, 318362)
DEEP_LONG_SEEDS = (104729, 130363, 155921, 181081, 205759)
LABELS = {"deep_cfr": "Deep CFR", "sd_cfr": "SD-CFR", "dream": "DREAM",
          "escher": "ESCHER", "vr_deep_dcfr_plus": "VR-DeepDCFR+",
          "vr_deep_pdcfr_plus": "VR-DeepPDCFR+", "ucv": "UCV-ESCHER (grouped)"}
DREAM_BUCKET = "gs://clever-overview-399515-leduc-poker-dream-results"
OLD_RUN = "leduc-escher-arch-exp17-six-algorithm-h2h-20260811-220853"
OLD = f"{DREAM_BUCKET}/{OLD_RUN}/outputs/cloud/{OLD_RUN}/six_algorithm_final_policy_head_to_head_20260811_221252"
ROOTS = {
    "old": OLD,
    "ucv": f"{DREAM_BUCKET}/exp35-confirm-20260916-011231",
    "deep_long": f"{DREAM_BUCKET}/exp21-36h-20260830-141641",
    "dream": f"{DREAM_BUCKET}/drm46-paper36h-20260920-214138",
    "escher": "gs://clever-overview-399515-leduc-poker-escher-results/exp49-paper36h-20260920-194746",
    "sd_cfr": "gs://clever-overview-399515-leduc-poker-results/exp29-sdcfr36h-20260920-233546",
}
SEEDS = {a: COMMON_SEEDS for a in LABELS}
SEEDS["ucv"] = UCV_SEEDS
ENDPOINTS = {"node_15m": tuple(LABELS),
             "time_36h": ("deep_cfr", "sd_cfr", "dream", "escher", "ucv")}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Unsafe source path: {value}")
    return str(p)


def download(uri, root):
    """Pin object generation and retain its server metadata beside local bytes."""
    root = Path(root)
    name = hashlib.sha256(uri.encode()).hexdigest()[:20] + "_" + uri.rsplit("/", 1)[-1]
    path = root / "objects" / name
    sidecar = path.with_name(path.name + ".origin.json")
    if path.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text())
        if saved["uri"] != uri or saved["sha256"] != sha256(path):
            raise ValueError(f"Cached source changed: {path}")
        return path
    meta = json.loads(subprocess.check_output(
        ["gcloud", "storage", "objects", "describe", uri, "--format=json"], text=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid sliced-download temporary-file/checksum failures seen on macOS.
    # Keep both gcloud checksum validation and the original archive SHA-256 check.
    env = dict(os.environ, CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD="0")
    for attempt in range(3):
        try:
            subprocess.run(["gcloud", "storage", "cp", uri + "#" + str(meta["generation"]), str(path)],
                           check=True, env=env)
            break
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    write_json(sidecar, {"uri": uri, "object_metadata": meta, "sha256": sha256(path)})
    return path


def one(rows, **criteria):
    matches = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source row for {criteria}, got {len(matches)}")
    return matches[0]


def truth(value):
    return str(value).lower() in ("true", "1")


def build_records(root, *, smoke=False):
    """Fetch small metadata first; return audited endpoints with exact artifact URIs."""
    root = Path(root)
    def table(uri):
        return read_csv(download(uri, root))
    records = []
    def add(a, seed, endpoint, uri, ref, *, nodes, iteration, seconds=None,
            expected_sha=None, source, checkpoint, cohort="historical_common"):
        records.append(dict(algorithm=a, seed=int(seed), endpoint=endpoint, uri=uri,
                            source_experiment=source, source_checkpoint=checkpoint,
                            source_exploitability=float(ref), nodes_touched=int(nodes),
                            iteration=int(iteration), active_seconds=seconds,
                            expected_sha256=expected_sha, cohort=cohort))
    old = table(OLD + "/snapshot_inventory.csv")
    old_metrics = table(OLD + "/final_policy_metrics.csv")
    for a in ("deep_cfr", "vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"):
        for seed in COMMON_SEEDS[:1] if smoke else COMMON_SEEDS:
            r = one(old, algorithm_id=a, seed=seed)
            ref = one(old_metrics, algorithm_id=a, seed=seed)
            rel = "snapshots/" + r["path"].split("/snapshots/", 1)[1]
            add(a, seed, "node_15m", OLD + "/" + safe_relative(rel), ref["exploitability"],
                nodes=r["nodes_touched"], iteration=r["checkpoint"], expected_sha=r["sha256"],
                source="architecture17 (retained Deep CFR / author VR configurations)",
                checkpoint=Path(r["path"]).name)
    for a, key, seeds, source in (("ucv", "ucv", UCV_SEEDS, "architecture35"),
                                ("deep_cfr", "deep_long", DEEP_LONG_SEEDS, "architecture21")):
        inventory = table(ROOTS[key] + "/analysis/snapshot_inventory.csv")
        metrics = table(ROOTS[key] + "/analysis/checkpoint_policy_metrics.csv")
        for seed in seeds[:1] if smoke else seeds:
            for ep in (("node_15m", "time_36h") if a == "ucv" else ("time_36h",)):
                r = one(inventory, seed=seed, checkpoint_id=ep,
                        **({"candidate_id": "grouped_wide_ucv"} if a == "ucv" else {"algorithm_id": a}))
                ref = one(metrics, seed=seed, checkpoint_id=ep,
                          **({"policy_id": "grouped_wide_ucv"} if a == "ucv" else {"algorithm_id": a}))
                rel = "workers/" + r["path"].split("/workers/", 1)[1]
                add(a, seed, ep, ROOTS[key] + "/" + safe_relative(rel), ref["exploitability"],
                    nodes=r["nodes_touched"], iteration=r["completed_iteration"],
                    seconds=float(r["active_seconds"] if a == "ucv" else r["wall_clock_seconds"]),
                    expected_sha=r["sha256"], source=source, checkpoint=ep,
                    cohort="ucv_fresh" if a == "ucv" else "deep_long")
    for a, filename, field, source in (
            ("dream", "checkpoint_seed_metrics.csv", "neural_exploitability", "dream46"),
            ("escher", "checkpoint_metrics.csv", "neural_exploitability", "escher49"),
            ("sd_cfr", "checkpoint_seed_metrics.csv", "sd_cfr_uniform_exploitability", "deep29")):
        metrics = table(ROOTS[a] + "/analysis/" + filename)
        for index, seed in enumerate(COMMON_SEEDS):
            if smoke and index:
                break
            selected = [r for r in metrics if int(r["seed"]) == seed]
            for ep, flag in (("node_15m", "is_node_15m_endpoint"), ("time_36h", "is_final_endpoint")):
                r = one([r for r in selected if truth(r.get(flag))], seed=seed)
                base = f"{ROOTS[a]}/evaluation/task_{index:03d}_seed_{seed}/"
                expected_sha = None
                if a == "sd_cfr":
                    base = f"{ROOTS[a]}/training/task_{index:03d}_seed_{seed}/"
                    result = json.loads(download(base + f"seed_{seed}/training_result.json", root).read_text())
                    if result["experiment_id"] != 29 or int(result["seed"]) != seed:
                        raise ValueError("Wrong SD-CFR training result")
                    uri = base + safe_relative(result["archive_path"])
                    expected_sha = result["archive_sha256"]
                else:
                    uri = base + safe_relative(r["weights_path"])
                add(a, seed, ep, uri, r[field], nodes=r["nodes_touched"], iteration=r["iteration"],
                    seconds=float(r["active_training_seconds"]), source=source,
                    checkpoint=r["checkpoint_id"], expected_sha=expected_sha)
    validate_records(records, smoke=smoke)
    write_json(root / ("smoke_manifest.json" if smoke else "manifest.json"), records)
    return records


def validate_records(records, *, smoke=False):
    expected = set()
    for ep, algorithms in ENDPOINTS.items():
        for a in algorithms:
            seeds = DEEP_LONG_SEEDS if a == "deep_cfr" and ep == "time_36h" else SEEDS[a]
            expected.update((ep, a, seed) for seed in (seeds[:1] if smoke else seeds))
    observed = [(r["endpoint"], r["algorithm"], r["seed"]) for r in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Incomplete, duplicated or unexpected league policies")
    for r in records:
        if r["endpoint"] == "node_15m":
            # Historical Deep CFR stopped slightly below the boundary. Report, do not hide it.
            low = 14_800_000 if r["algorithm"] == "deep_cfr" else 15_000_000
            if not low <= r["nodes_touched"] <= 15_500_000:
                raise ValueError(f"Unexpected node-budget discrepancy: {r}")
        elif not 36 * 3600 <= r["active_seconds"] <= 38 * 3600:
            raise ValueError(f"Unexpected time-budget discrepancy: {r}")


def artifact(record, root):
    path = download(record["uri"], root)
    if record["expected_sha256"] and sha256(path) != record["expected_sha256"]:
        raise ValueError(f"Source archive hash differs from original manifest: {record['uri']}")
    return path
