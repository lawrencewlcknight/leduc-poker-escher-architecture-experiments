"""Run one source-trajectory worker for Experiment 30."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
import resource
import subprocess
import sys
import time

import pyspiel

from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_ID as SOURCE_CANDIDATE_ID,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    read_training_state,
)

from .audit import audit_source, extract_promoted_source
from .config import (
    EXPERIMENT_NAME,
    GAME_NAME,
    SOURCE_CHECKPOINTS,
    contract_manifest,
)


LOGGER = logging.getLogger(__name__)


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _source_role(checkpoint_id: str, *, smoke: bool) -> str:
    if not smoke:
        return checkpoint_id
    return "smoke_time_02" if checkpoint_id == "time_24h" else "smoke_time_03"


def _source_path(
    source_root: Path,
    *,
    seed: int,
    checkpoint_id: str,
    smoke: bool,
) -> tuple[Path, str]:
    role = _source_role(checkpoint_id, smoke=smoke)
    filename = f"{SOURCE_CANDIDATE_ID}_seed_{seed}_{role}.pt"
    matches = list(Path(source_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Experiment 29 source state named {filename}, "
            f"found {len(matches)}"
        )
    return matches[0], role


def run_worker(
    *,
    seed: int,
    source_root: Path,
    worker_dir: Path,
    smoke: bool,
) -> dict:
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    game = pyspiel.load_game(GAME_NAME)
    source_records = []
    summaries = []
    information_rows = []
    repair_rows = []

    for checkpoint_id in SOURCE_CHECKPOINTS:
        started = time.perf_counter()
        source_path, source_checkpoint_id = _source_path(
            source_root,
            seed=seed,
            checkpoint_id=checkpoint_id,
            smoke=smoke,
        )
        source_hash = sha256(source_path)
        payload = read_training_state(source_path)
        source = extract_promoted_source(
            payload,
            seed=seed,
            checkpoint_id=checkpoint_id,
            source_checkpoint_id=source_checkpoint_id,
        )
        audit = audit_source(source, game, smoke=smoke)
        elapsed_seconds = time.perf_counter() - started
        audit.source_summary["audit_wall_clock_seconds"] = elapsed_seconds
        summaries.append(audit.source_summary)
        information_rows.extend(audit.information_sets)
        repair_rows.extend(audit.repair_curves)
        source_records.append(
            {
                "checkpoint_id": checkpoint_id,
                "source_checkpoint_id": source_checkpoint_id,
                "source_path": str(source_path.resolve()),
                "source_sha256": source_hash,
                "source_size_bytes": int(source_path.stat().st_size),
                "source_repository_commit": str(payload["repository_commit"]),
                "audit_wall_clock_seconds": elapsed_seconds,
            }
        )
        LOGGER.info(
            "Audited source seed %s checkpoint %s: %s information sets, "
            "%s repair-curve rows in %.1f seconds",
            seed,
            checkpoint_id,
            len(audit.information_sets),
            len(audit.repair_curves),
            elapsed_seconds,
        )
        del payload, source, audit
        gc.collect()

    write_csv(worker_dir / "source_summary.csv", summaries)
    write_csv(worker_dir / "information_set_audit.csv", information_rows)
    write_csv(worker_dir / "repair_curves.csv", repair_rows)
    result = {
        "schema_version": 1,
        "experiment_name": EXPERIMENT_NAME,
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": _repository_commit(),
        "contract": contract_manifest(),
        "source_states": source_records,
        "peak_rss_mb": _peak_rss_mb(),
        "artifacts": {
            "source_summary": "source_summary.csv",
            "information_set_audit": "information_set_audit.csv",
            "repair_curves": "repair_curves.csv",
        },
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(
        worker_dir / "SUCCESS.json",
        {
            "status": "complete",
            "source_seed": int(seed),
            "source_states": source_records,
        },
    )
    return result


__all__ = ["run_worker"]
