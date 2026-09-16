"""Per-seed post-training worker shared by Experiments 36--39."""

from __future__ import annotations

import gc
from pathlib import Path
import resource
import subprocess
import sys
import time

import pyspiel
import torch

from experiments.leduc_poker.causal_rare_state_audit.audit import (
    extract_promoted_source,
)
from experiments.leduc_poker.four_algorithm_heldout_benchmark.common import (
    sha256,
    write_csv,
    write_json,
)
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    read_training_state,
)

from .config import (
    GAME_NAME,
    SOURCE_CHECKPOINT,
    contract_manifest,
    evaluation_steps,
    method_config,
)
from .core import clone_state, exact_metrics, information_sets, make_model, model_table
from .methods import METHOD_RUNNERS


def repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def source_path(source_root: Path, *, seed: int, smoke: bool) -> tuple[Path, str]:
    role = "smoke_time_03" if smoke else SOURCE_CHECKPOINT
    filename = f"promoted_ucv_cross_entropy_seed_{seed}_{role}.pt"
    matches = list(Path(source_root).rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Experiment 29 source named {filename}; found {len(matches)}"
        )
    return matches[0], role


def _save_policy(path: Path, *, method_id: str, arm_id: str, seed: int,
                 source, state, selection: str, update: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "type": "post_training_policy",
            "method_id": method_id,
            "arm_id": arm_id,
            "source_seed": int(seed),
            "source_checkpoint": SOURCE_CHECKPOINT,
            "selection": selection,
            "update": int(update),
            "input_size": int(source.infostates.shape[1]),
            "output_size": int(source.policies.shape[1]),
            "network_layers": list(source.network_layers),
            "model": state,
        },
        path,
    )
    return {
        "arm_id": arm_id,
        "selection": selection,
        "update": int(update),
        "relative_path": str(path),
        "sha256": sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def run_worker(*, method_id: str, seed: int, source_root: Path,
               worker_dir: Path, smoke: bool) -> dict:
    started = time.perf_counter()
    worker_dir = Path(worker_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=True)
    config = method_config(method_id, smoke=smoke)
    schedule = set(evaluation_steps(method_id, smoke=smoke))
    path, source_role = source_path(source_root, seed=seed, smoke=smoke)
    payload = read_training_state(path)
    source = extract_promoted_source(
        payload, seed=seed, checkpoint_id=SOURCE_CHECKPOINT,
        source_checkpoint_id=source_role,
    )
    game = pyspiel.load_game(GAME_NAME)
    rows = information_sets(game)
    blueprint_model = make_model(source)
    blueprint_table = model_table(game, blueprint_model)
    metrics_rows, policies = [], []

    for arm_index, (arm_id, arm) in enumerate(config["arms"].items()):
        model = make_model(source)
        best = {"exploitability": float("inf"), "state": None, "update": -1}
        latest_state = None
        latest_update = -1

        def record(active_model, update, loss, diagnostics):
            nonlocal latest_state, latest_update
            if int(update) not in schedule:
                return
            table = model_table(game, active_model)
            metrics = exact_metrics(
                game, table, blueprint_table, source.exact_table
            )
            row = {
                "method_id": method_id,
                "experiment_id": config["experiment_id"],
                "source_seed": int(seed),
                "arm_id": arm_id,
                "arm_index": int(arm_index),
                "update": int(update),
                "training_loss": float(loss),
                **{key: float(value) for key, value in diagnostics.items()},
                **metrics,
            }
            metrics_rows.append(row)
            latest_state = clone_state(active_model)
            latest_update = int(update)
            if metrics["exploitability"] < best["exploitability"]:
                best.update(
                    exploitability=float(metrics["exploitability"]),
                    state=clone_state(active_model), update=int(update),
                )

        METHOD_RUNNERS[method_id](
            game=game, source=source, model=model, rows=rows,
            blueprint_table=blueprint_table, config=config, arm=arm,
            record=record,
        )
        if best["state"] is None or latest_state is None:
            raise RuntimeError(f"Method {method_id} arm {arm_id} produced no evaluations")
        for selection, state, update in (
            ("best_development", best["state"], best["update"]),
            ("final", latest_state, latest_update),
        ):
            record_path = worker_dir / "policies" / (
                f"{method_id}_{arm_id}_{selection}_seed_{seed}.pt"
            )
            policy_record = _save_policy(
                record_path, method_id=method_id, arm_id=arm_id, seed=seed,
                source=source, state=state, selection=selection, update=update,
            )
            policy_record["relative_path"] = str(record_path.relative_to(worker_dir))
            policies.append(policy_record)

    write_csv(worker_dir / "fine_tuning_metrics.csv", metrics_rows)
    result = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "method_id": method_id,
        "source_seed": int(seed),
        "smoke": bool(smoke),
        "repository_commit": repository_commit(),
        "contract": contract_manifest(method_id),
        "source": {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "size_bytes": int(path.stat().st_size),
            "repository_commit": str(payload["repository_commit"]),
        },
        "policies": policies,
        "peak_rss_mb": peak_rss_mb(),
        "wall_clock_seconds": time.perf_counter() - started,
        "artifacts": {"metrics": "fine_tuning_metrics.csv"},
        "status": "complete",
    }
    write_json(worker_dir / "worker_result.json", result)
    write_json(
        worker_dir / "SUCCESS.json",
        {"status": "complete", "method_id": method_id, "source_seed": seed},
    )
    del payload, source
    gc.collect()
    return result


__all__ = ["run_worker"]

