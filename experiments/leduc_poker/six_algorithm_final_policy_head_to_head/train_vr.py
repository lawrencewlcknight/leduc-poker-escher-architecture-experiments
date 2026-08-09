"""Train the two missing VR-Deep arms and save final playable policies."""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Mapping, Sequence

import numpy as np

from escher_poker.constants import LEDUC_GAME_VALUE_PLAYER_0
from vr_deep_cfr.logger import Logger
from vr_deep_cfr.policy_snapshots import save_policy_snapshot, snapshot_filename

from .config import ALGORITHMS, VR_ALGORITHMS
from .io_utils import write_csv, write_json
from .policies import read_snapshot_metadata


LOGGER = logging.getLogger(__name__)


def _solver(algorithm_id: str, seed: int, config: Mapping[str, object], target_nodes: int):
    from vr_deep_cfr import VRDeepDCFRPlus, VRDeepPDCFRPlus

    spec = ALGORITHMS[algorithm_id]
    solver_class = {
        "VRDeepDCFRPlus": VRDeepDCFRPlus,
        "VRDeepPDCFRPlus": VRDeepPDCFRPlus,
    }[spec["class_name"]]
    excluded = {
        "max_num_iterations",
        "preserve_evaluation_rng",
        "evaluate_initial_policy",
        "early_evaluation_node_thresholds",
    }
    kwargs = {key: value for key, value in config.items() if key not in excluded}
    kwargs.update(
        num_episodes=(
            2 * int(config["num_traversals"]) * int(config["max_num_iterations"])
        ),
        alpha=float(spec["alpha"]),
        gamma=float(spec["gamma"]),
        seed=int(seed),
        logger=Logger(verbose=False),
    )
    if spec["reinitialize_imm_regret_networks"] is not None:
        kwargs["reinitialize_imm_regret_networks"] = bool(
            spec["reinitialize_imm_regret_networks"]
        )
    solver = solver_class(**kwargs)
    solver.target_nodes_touched = int(target_nodes)
    solver.max_num_iterations = int(config["max_num_iterations"])
    solver.preserve_evaluation_rng = bool(config["preserve_evaluation_rng"])
    solver.evaluate_initial_policy = bool(config.get("evaluate_initial_policy", False))
    solver.early_evaluation_node_thresholds = tuple(
        int(value) for value in config.get("early_evaluation_node_thresholds", ())
    )
    return solver


def train_vr_seed(
    *,
    algorithm_id: str,
    seed: int,
    config: Mapping[str, object],
    target_nodes: int,
    run_dir: Path,
) -> dict:
    """Train one author-parameterised VR policy to the common node target."""
    import torch

    solver = _solver(algorithm_id, seed, config, target_nodes)
    try:
        raw_rows = solver.solve()
        if int(solver.nodes_touched) < int(target_nodes):
            raise RuntimeError(
                f"{algorithm_id} seed {seed} stopped at {solver.nodes_touched} nodes"
            )
        snapshot_path = (
            run_dir
            / "snapshots"
            / algorithm_id
            / snapshot_filename(algorithm_id, seed)
        )
        save_policy_snapshot(
            solver,
            snapshot_path,
            algorithm_id=algorithm_id,
            algorithm_label=ALGORITHMS[algorithm_id]["algorithm_label"],
            seed=seed,
            config=dict(config),
        )
        saved = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        for name, tensor in solver.ave_policy_trainer.model.state_dict().items():
            if not torch.equal(tensor.detach().cpu(), saved["policy_state_dict"][name]):
                raise RuntimeError(f"Saved final policy differs at tensor {name}")

        curves = []
        for index, raw in enumerate(raw_rows):
            policy_value = float(raw["average_policy_value"])
            curves.append(
                {
                    "algorithm_id": algorithm_id,
                    "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
                    "seed": int(seed),
                    "checkpoint_index": int(index),
                    "iteration": int(raw["iteration"]),
                    "episode": int(raw["episode"]),
                    "nodes_touched": int(raw["nodes_touched"]),
                    "wall_clock_seconds": float(raw["wall_clock_seconds"]),
                    "exploitability": float(raw["exp"]),
                    "average_policy_value": policy_value,
                    "policy_value_error": abs(
                        policy_value - LEDUC_GAME_VALUE_PLAYER_0
                    ),
                    "checkpoint_kind": str(raw.get("checkpoint_kind", "")),
                    "checkpoint_target_nodes": raw.get("checkpoint_target_nodes"),
                }
            )
        final = curves[-1]
        return {
            "algorithm_id": algorithm_id,
            "algorithm_label": ALGORITHMS[algorithm_id]["algorithm_label"],
            "seed": int(seed),
            "target_nodes": int(target_nodes),
            "final_nodes_touched": int(final["nodes_touched"]),
            "final_iteration": int(final["iteration"]),
            "final_wall_clock_seconds": float(final["wall_clock_seconds"]),
            "final_exploitability": float(final["exploitability"]),
            "final_policy_value": float(final["average_policy_value"]),
            "snapshot": read_snapshot_metadata(algorithm_id, snapshot_path),
            "curves": curves,
        }
    finally:
        del solver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_worker(input_path: Path, output_path: Path) -> int:
    with open(input_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = train_vr_seed(
        algorithm_id=str(payload["algorithm_id"]),
        seed=int(payload["seed"]),
        config=payload["config"],
        target_nodes=int(payload["target_nodes"]),
        run_dir=Path(payload["run_dir"]),
    )
    write_json(output_path, result)
    return 0


def _run_subprocess(
    algorithm_id: str,
    seed: int,
    config: Mapping[str, object],
    target_nodes: int,
    run_dir: Path,
) -> dict:
    stem = f"{algorithm_id}_seed_{seed}"
    input_path = run_dir / "worker_inputs" / f"{stem}.json"
    result_path = run_dir / "worker_results" / f"{stem}.json"
    log_path = run_dir / "worker_logs" / f"{stem}.log"
    snapshot_path = run_dir / "snapshots" / algorithm_id / snapshot_filename(
        algorithm_id, seed
    )
    if result_path.exists() and snapshot_path.exists():
        with open(result_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached["snapshot"]["sha256"] == read_snapshot_metadata(
            algorithm_id, snapshot_path
        )["sha256"]:
            LOGGER.info("Reusing completed worker %s", stem)
            return cached

    write_json(
        input_path,
        {
            "algorithm_id": algorithm_id,
            "seed": int(seed),
            "config": dict(config),
            "target_nodes": int(target_nodes),
            "run_dir": str(run_dir.resolve()),
        },
    )
    command = [
        sys.executable,
        "-m",
        "experiments.leduc_poker.six_algorithm_final_policy_head_to_head.run",
        "--worker-input-json",
        str(input_path),
        "--worker-output-json",
        str(result_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if completed.returncode:
        raise RuntimeError(f"Worker {stem} failed; see {log_path}")
    with open(result_path, encoding="utf-8") as handle:
        return json.load(handle)


def train_all_vr(
    *,
    seeds: Sequence[int],
    config: Mapping[str, object],
    target_nodes: int,
    run_dir: Path,
    continue_on_error: bool = False,
) -> dict:
    results = []
    failures = []
    for algorithm_id in VR_ALGORITHMS:
        for seed in seeds:
            LOGGER.info("Training %s seed %s", algorithm_id, seed)
            try:
                results.append(
                    _run_subprocess(
                        algorithm_id,
                        int(seed),
                        config,
                        target_nodes,
                        run_dir,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "algorithm_id": algorithm_id,
                        "seed": int(seed),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if not continue_on_error:
                    raise

    summaries = [{key: value for key, value in row.items() if key != "curves"} for row in results]
    curves = [curve for result in results for curve in result["curves"]]
    write_csv(run_dir / "vr_training_summary.csv", summaries)
    write_csv(run_dir / "vr_training_curves.csv", curves)
    write_json(run_dir / "vr_training_failures.json", failures)
    return {"results": results, "failures": failures}


__all__ = ["run_worker", "train_all_vr", "train_vr_seed"]
