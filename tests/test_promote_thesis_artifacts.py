"""Tests for promoting lightweight thesis artifacts from experiment outputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "promote_thesis_artifacts.py"


def load_promoter():
    spec = importlib.util.spec_from_file_location("promote_thesis_artifacts", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_file(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_metadata(run_dir: Path, experiment_name: str = "example_experiment") -> None:
    write_file(
        run_dir / "experiment_metadata.json",
        json.dumps({"experiment_config": {"experiment_name": experiment_name}}),
    )


def test_selected_files_include_thesis_artifacts(tmp_path):
    promoter = load_promoter()
    run_dir = tmp_path / "outputs" / "example_run_20260517_120000"
    write_metadata(run_dir)
    write_file(run_dir / "plot.png")
    write_file(run_dir / "table.csv")
    write_file(run_dir / "aggregate_summary.json", "{}")
    write_file(run_dir / "paired_difference_summary.json", "{}")
    write_file(run_dir / "notes.txt")

    selected, skipped = promoter.selected_files(
        run_dir,
        promoter.DEFAULT_INCLUDE_PATTERNS,
        promoter.DEFAULT_EXCLUDE_PATTERNS,
    )

    selected_names = {path.as_posix() for path in selected}
    assert {
        "plot.png",
        "table.csv",
        "aggregate_summary.json",
        "paired_difference_summary.json",
        "experiment_metadata.json",
    } <= selected_names
    assert "notes.txt" not in selected_names
    assert {"path": "notes.txt", "reason": "not_included"} in skipped


def test_selected_files_exclude_heavy_artifacts(tmp_path):
    promoter = load_promoter()
    run_dir = tmp_path / "outputs" / "example_run_20260517_120000"
    write_metadata(run_dir)
    write_file(run_dir / "checkpoint.pt")
    write_file(run_dir / "weights.pth")
    write_file(run_dir / "curves.npz")
    write_file(run_dir / "experiment.log")
    write_file(run_dir / "failed_seeds.json", "{}")
    write_file(run_dir / "checkpoints" / "model.csv")
    write_file(run_dir / "snapshots" / "policy.png")
    write_file(run_dir / "traces" / "trace.csv")

    selected, skipped = promoter.selected_files(
        run_dir,
        promoter.DEFAULT_INCLUDE_PATTERNS,
        promoter.DEFAULT_EXCLUDE_PATTERNS,
    )

    selected_names = {path.as_posix() for path in selected}
    assert selected_names == {"experiment_metadata.json"}
    skipped_excluded = {
        row["path"]
        for row in skipped
        if row["reason"] == "excluded"
    }
    assert {
        "checkpoint.pt",
        "weights.pth",
        "curves.npz",
        "experiment.log",
        "failed_seeds.json",
        "checkpoints/model.csv",
        "snapshots/policy.png",
        "traces/trace.csv",
    } <= skipped_excluded


def test_downloaded_job_directory_is_searched_recursively(tmp_path):
    promoter = load_promoter()
    job_dir = tmp_path / "cloud_outputs" / "JOB_NAME"
    run_dir = (
        job_dir
        / "outputs"
        / "cloud"
        / "escher-architecture-exp28"
        / "leduc_poker_escher_candidate_architecture_multiseed_20260711_120000"
    )
    write_metadata(
        run_dir,
        experiment_name="leduc_poker_escher_candidate_architecture_multiseed",
    )
    write_file(run_dir / "plot.png")
    write_file(run_dir / "seed_summary.csv")
    write_file(run_dir / "curves.npz")

    dest = tmp_path / "thesis_artifacts"
    discovered = promoter.discover_run_dirs(job_dir)
    assert discovered == [run_dir]

    manifest = promoter.promote_run(
        run_dir,
        dest,
        promoter.DEFAULT_INCLUDE_PATTERNS,
        promoter.DEFAULT_EXCLUDE_PATTERNS,
        overwrite=False,
        dry_run=False,
    )

    destination_run = (
        dest
        / "leduc_poker_escher_candidate_architecture_multiseed"
        / "leduc_poker_escher_candidate_architecture_multiseed_20260711_120000"
    )
    assert (destination_run / "plot.png").is_file()
    assert (destination_run / "seed_summary.csv").is_file()
    assert (destination_run / "experiment_metadata.json").is_file()
    assert not (destination_run / "curves.npz").exists()
    assert (destination_run / "promotion_manifest.json").is_file()
    assert manifest["destination_run_directory"] == destination_run.resolve()
