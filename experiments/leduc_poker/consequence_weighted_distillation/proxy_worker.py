"""Build proxy diagnostics for one frozen Experiment 29 trajectory."""

from __future__ import annotations

import gc
from pathlib import Path

import pyspiel

from experiments.leduc_poker.causal_rare_state_audit.audit import extract_promoted_source
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    read_json,
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    read_training_state,
)

from .common import (
    peak_rss_mb,
    read_csv,
    repository_commit,
    source_path,
    validate_source_manifest,
)
from .config import EXPERIMENT_NAME, GAME_NAME, SOURCE_CHECKPOINTS, contract_manifest
from .proxy import build_proxy_rows


def run_proxy_worker(
    *,
    seed: int,
    source_root: Path,
    audit_root: Path,
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    audit_paths = list(Path(audit_root).rglob("information_set_audit.csv"))
    matching_audits = []
    for path in audit_paths:
        rows = read_csv(path)
        if rows and int(rows[0]["source_seed"]) == int(seed):
            matching_audits.append((path, rows))
    if len(matching_audits) != 1:
        raise FileNotFoundError(
            f"Expected one Experiment 30 audit for seed {seed}, found {len(matching_audits)}"
        )
    audit_path, audit_rows = matching_audits[0]
    audit_manifest_path = audit_path.parent / "worker_result.json"
    if audit_manifest_path.is_file():
        audit_manifest = read_json(audit_manifest_path)
        if (
            audit_manifest.get("status") != "complete"
            or audit_manifest.get("experiment_name") != "causal_rare_state_audit"
            or int(audit_manifest.get("source_seed", -1)) != int(seed)
            or audit_manifest.get("artifacts", {}).get("information_set_audit")
            != audit_path.name
        ):
            raise ValueError("Experiment 30 audit manifest is incompatible")
    elif not smoke:
        raise FileNotFoundError("Experiment 30 worker manifest is required")
    game = pyspiel.load_game(GAME_NAME)
    proxy_rows, metric_rows, sources = [], [], []
    for checkpoint_id in SOURCE_CHECKPOINTS:
        path, role = source_path(
            source_root, seed=seed, checkpoint_id=checkpoint_id, smoke=smoke
        )
        provenance = validate_source_manifest(path)
        payload = read_training_state(path)
        source = extract_promoted_source(
            payload,
            seed=seed,
            checkpoint_id=checkpoint_id,
            source_checkpoint_id=role,
        )
        rows, metrics = build_proxy_rows(source, game, audit_rows)
        proxy_rows.extend(rows)
        metric_rows.extend(metrics)
        sources.append(
            {
                "checkpoint_id": checkpoint_id,
                "source_checkpoint_id": role,
                **provenance,
                "size_bytes": int(path.stat().st_size),
            }
        )
        del payload, source
        gc.collect()
    write_csv(worker_dir / "proxy_information_sets.csv", proxy_rows)
    write_csv(worker_dir / "proxy_repair_metrics.csv", metric_rows)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "stage": "proxy",
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": repository_commit(),
        "contract": contract_manifest(),
        "source_states": sources,
        "audit_path": str(audit_path.resolve()),
        "audit_manifest": str(audit_manifest_path.resolve()),
        "audit_sha256": sha256(audit_path),
        "peak_rss_mb": peak_rss_mb(),
        "artifacts": {
            "proxy_information_sets": "proxy_information_sets.csv",
            "proxy_repair_metrics": "proxy_repair_metrics.csv",
        },
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(worker_dir / "SUCCESS.json", {"status": "complete", "source_seed": seed})
    return result
