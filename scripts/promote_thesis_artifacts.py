#!/usr/bin/env python3
"""Promote lightweight thesis artifacts from experiment outputs into the repo.

Full experiment outputs remain scratch data under ``outputs/`` or downloaded cloud
folders. This script copies only selected thesis-facing artifacts into a tracked
destination tree:

    thesis_artifacts/<experiment_name>/<run_directory_name>/
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


DEFAULT_INCLUDE_PATTERNS = [
    "*.png",
    "*.csv",
    "aggregate_summary.json",
    "paired_difference_summary.json",
    "paired_aggregate_summary.json",
    "best_checkpoint_summary.json",
    "experiment_metadata.json",
]

DEFAULT_EXCLUDE_PATTERNS = [
    "*.pt",
    "*.pth",
    "*.npz",
    "*.log",
    "failed_seeds.json",
    "failed_runs.json",
    "checkpoints/*",
    "snapshots/*",
    "traces/*",
]

MANIFEST_NAME = "promotion_manifest.json"


def parse_globs(value: str | None) -> list[str]:
    """Parse a comma-separated glob list."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def json_safe(value: Any) -> Any:
    """Convert paths and nested containers to JSON-serialisable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def path_matches(path: Path, patterns: Iterable[str]) -> bool:
    """Return true when ``path`` matches any glob by relative path or basename."""
    rel = path.as_posix()
    name = path.name
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if fnmatch.fnmatch(rel, normalized):
            return True
        if "/" not in normalized and fnmatch.fnmatch(name, normalized):
            return True
    return False


def discover_run_dirs(source: Path) -> list[Path]:
    """Find run directories by locating ``experiment_metadata.json`` files."""
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source}")

    direct_metadata = source / "experiment_metadata.json"
    if direct_metadata.is_file():
        return [source]

    run_dirs = {
        metadata.parent
        for metadata in source.rglob("experiment_metadata.json")
        if metadata.is_file()
    }
    return sorted(run_dirs)


def load_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "experiment_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a JSON object: {metadata_path}")
    return metadata


def infer_experiment_name(metadata: dict[str, Any], run_dir: Path) -> str:
    """Infer experiment name from metadata, falling back to the run directory."""
    experiment_config = metadata.get("experiment_config")
    if isinstance(experiment_config, dict):
        experiment_name = experiment_config.get("experiment_name")
        if experiment_name:
            return str(experiment_name)

    # Some current ESCHER runners write equivalent config under these keys. They
    # are fallbacks only; ``experiment_config.experiment_name`` is preferred.
    for key in ("experiment_name", "base_config", "baseline_config"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict) and value.get("experiment_name"):
            return str(value["experiment_name"])

    return run_dir.name


def selected_files(
    run_dir: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> tuple[list[Path], list[dict[str, str]]]:
    """Return selected files and skipped-file reasons for one run directory."""
    selected: list[Path] = []
    skipped: list[dict[str, str]] = []

    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        rel = path.relative_to(run_dir)
        if rel.name == MANIFEST_NAME:
            skipped.append({"path": rel.as_posix(), "reason": "manifest"})
            continue
        if path_matches(rel, exclude_patterns):
            skipped.append({"path": rel.as_posix(), "reason": "excluded"})
            continue
        if path_matches(rel, include_patterns):
            selected.append(rel)
        else:
            skipped.append({"path": rel.as_posix(), "reason": "not_included"})

    return selected, skipped


def promote_run(
    run_dir: Path,
    dest_root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    *,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Promote selected artifacts for a single discovered run directory."""
    metadata = load_metadata(run_dir)
    experiment_name = infer_experiment_name(metadata, run_dir)
    destination = dest_root / experiment_name / run_dir.name
    selected, skipped = selected_files(run_dir, include_patterns, exclude_patterns)

    copied_files: list[dict[str, Any]] = []
    selected_manifest_entries: list[dict[str, Any]] = []
    overwritten_files: list[str] = []
    existing_skipped: list[dict[str, str]] = []

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)

    for rel in selected:
        source_path = run_dir / rel
        dest_path = destination / rel
        exists = dest_path.exists()
        entry = {
            "relative_path": rel.as_posix(),
            "source_path": source_path.resolve(),
            "destination_path": dest_path.resolve(),
            "would_overwrite": bool(exists),
        }
        selected_manifest_entries.append(entry)

        if exists and not overwrite:
            existing_skipped.append({
                "path": rel.as_posix(),
                "reason": "destination_exists",
            })
            continue

        if exists and overwrite:
            overwritten_files.append(rel.as_posix())

        if not dry_run:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            copied_files.append(entry)

    skipped.extend(existing_skipped)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_directory": run_dir.resolve(),
        "destination_run_directory": destination.resolve(),
        "experiment_name": experiment_name,
        "run_directory_name": run_dir.name,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "dry_run": bool(dry_run),
        "overwrite": bool(overwrite),
        "selected_files": selected_manifest_entries,
        "copied_files": copied_files,
        "skipped_files": skipped,
        "overwritten_files": overwritten_files,
    }

    if not dry_run:
        with open(destination / MANIFEST_NAME, "w", encoding="utf-8") as f:
            json.dump(json_safe(manifest), f, indent=2)

    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Promote lightweight thesis artifacts from one or more experiment "
            "output directories into thesis_artifacts/."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help=(
            "Run directory, downloaded cloud job directory, or parent directory "
            "containing experiment run directories."
        ),
    )
    parser.add_argument("--dest", default="thesis_artifacts")
    parser.add_argument("--include", default=None, help="Comma-separated extra include globs.")
    parser.add_argument("--exclude", default=None, help="Comma-separated extra exclude globs.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dest_root = Path(args.dest)
    include_patterns = DEFAULT_INCLUDE_PATTERNS + parse_globs(args.include)
    exclude_patterns = DEFAULT_EXCLUDE_PATTERNS + parse_globs(args.exclude)

    manifests: list[dict[str, Any]] = []
    seen_run_dirs: set[Path] = set()
    for source in args.sources:
        for run_dir in discover_run_dirs(Path(source)):
            resolved = run_dir.resolve()
            if resolved in seen_run_dirs:
                continue
            seen_run_dirs.add(resolved)
            manifests.append(
                promote_run(
                    run_dir,
                    dest_root,
                    include_patterns,
                    exclude_patterns,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            )

    summary = {
        "runs_discovered": len(manifests),
        "dry_run": bool(args.dry_run),
        "destination_root": dest_root.resolve(),
        "runs": [
            {
                "source": manifest["source_run_directory"],
                "destination": manifest["destination_run_directory"],
                "selected_count": len(manifest["selected_files"]),
                "copied_count": len(manifest["copied_files"]),
                "skipped_count": len(manifest["skipped_files"]),
            }
            for manifest in manifests
        ],
    }
    print(json.dumps(json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
