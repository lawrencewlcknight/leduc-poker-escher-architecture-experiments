"""Shared Experiment 33 source and artifact helpers."""

from __future__ import annotations

import csv
from pathlib import Path
import resource
import subprocess
import sys

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    CANDIDATE_ID as SOURCE_CANDIDATE_ID,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    read_json,
    sha256,
)


def repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def source_role(checkpoint_id: str, *, smoke: bool) -> str:
    if not smoke:
        return checkpoint_id
    return "smoke_time_02" if checkpoint_id == "time_24h" else "smoke_time_03"


def source_path(
    source_root: Path, *, seed: int, checkpoint_id: str, smoke: bool
) -> tuple[Path, str]:
    role = source_role(checkpoint_id, smoke=smoke)
    filename = f"{SOURCE_CANDIDATE_ID}_seed_{seed}_{role}.pt"
    matches = list(Path(source_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source named {filename}, found {len(matches)}")
    return matches[0], role


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_source_manifest(path: Path) -> dict:
    """Require the archived Experiment 29 manifest and validate the state hash."""
    manifests = list(path.parents[1].glob("worker_result.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"Expected one source worker manifest for {path}")
    manifest = read_json(manifests[0])
    rows = [
        row for row in manifest.get("training_states", ())
        if row.get("filename") == path.name
    ]
    if len(rows) != 1:
        raise ValueError(f"Source manifest does not identify {path.name}")
    observed = sha256(path)
    if rows[0].get("sha256") != observed:
        raise ValueError(f"Source checksum differs for {path.name}")
    return {
        "source_manifest": str(manifests[0].resolve()),
        "source_repository_commit": manifest.get("repository_commit", "unknown"),
        "sha256": observed,
    }
