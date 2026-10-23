"""Verified historical inputs; reconstruct inference and optimizer state only."""
from copy import deepcopy
import gc
import json
from pathlib import Path
import subprocess
import torch

from experiments.leduc_poker.grouped_wide_policy_confirmation.config import CANDIDATE_CONFIG, PRODUCTION_SEEDS
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _make_solver
from experiments.leduc_poker.frozen_critic_target_efficiency.run import make_smoke
from experiments.critic_target_cache_cli import source_paths
from experiments.critic_target_cache_benchmark import sha256
from .config import tasks


def spec(seed, checkpoint):
    return {"manifest": "SUCCESS.json", "game": "leduc", "checkpoint": checkpoint}


def validate_source(payload, seed, checkpoint):
    expected = {"type": "experiment_35_full_training_state", "schema_version": 1,
                "experiment_name": "grouped_wide_policy_confirmation", "variant_id": "grouped_wide_ucv",
                "seed": seed, "checkpoint_id": checkpoint}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Source mismatch: {key}")
    if (seed, checkpoint) not in tasks() or int(payload["solver"]["num_iteration"]) < 1:
        raise ValueError("Source is outside the frozen experiment schedule")
    for key, expected_value in CANDIDATE_CONFIG.items():
        if json.dumps(payload["config"].get(key)) != json.dumps(expected_value):
            raise ValueError(f"Historical learning configuration mismatch: {key}")
    if len(payload["regret_trainers"]) != 2 or len(payload["q_ensemble"]["members"]) != 2 or payload["calibration"] is None:
        raise ValueError("Missing native learning components")


def reconstruct(payload):
    config = deepcopy(payload["config"])
    # No historical replay is used or allocated a second time. New regret panels
    # are separate immutable arrays. Architecture and fit hyperparameters remain.
    for key in ("advantage_buffer_size", "ave_policy_buffer_size", "calibration_buffer_size"):
        config[key] = 1
    config["baseline_buffer_size"] = 2
    config.update(evaluate_initial_policy=False, early_evaluation_node_thresholds=(), evaluation_frequency=10**9)
    solver = _make_solver(payload["seed"], config)
    for trainer, state in zip(solver.regret_trainers, payload["regret_trainers"]):
        trainer.model.load_state_dict(state["model"])
        trainer.target_model.load_state_dict(state["target_model"])
        trainer.optimizer.load_state_dict(deepcopy(state["optimizer"]))
    for member, state in zip(solver.q_value_trainer.members, payload["q_ensemble"]["members"]):
        member.model.load_state_dict(state["model"])
        member.target_model.load_state_dict(state["target_model"])
        member.target_version = state["target_version"]
    calibration = payload["calibration"]
    solver.calibration_trainer.model.load_state_dict(calibration["model"])
    solver.calibration_trainer.target_model.load_state_dict(calibration["target_model"])
    solver.calibration_trainer.target_version = calibration["target_version"]
    solver.num_iteration = int(payload["solver"]["num_iteration"]) + 1
    solver.episode = int(payload["solver"]["episode"])
    return solver


def load_source(directory, seed, checkpoint, smoke=False):
    if smoke:
        if directory is not None:
            raise ValueError("Synthetic smoke does not load historical states")
        solver = make_smoke(seed)
        source_iteration = solver.num_iteration
        solver.num_iteration += 1
        return solver, {"synthetic_smoke": True, "seed": seed, "checkpoint": checkpoint,
                        "source_iteration": source_iteration}
    if directory is None:
        raise ValueError("Production requires a verified source worker directory")
    path, checksum = source_paths(directory, spec(seed, checkpoint), seed)
    if sha256(path) != checksum:
        raise ValueError("Source training-state checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_source(payload, seed, checkpoint)
    identity = {"seed": seed, "checkpoint": checkpoint, "source_iteration": payload["solver"]["num_iteration"],
                "source_commit": payload["repository_commit"], "training_state_sha256": checksum,
                "receipt_sha256": sha256(Path(directory)/"SUCCESS.json"), "source_config": payload["config"]}
    solver = reconstruct(payload)
    del payload
    gc.collect()
    return solver, identity


def fetch_source(bucket, run_id, index, directory):
    seed, checkpoint = tasks()[index]
    # Selected numeric seed order differs from the historical array order.
    source_index = tuple(PRODUCTION_SEEDS).index(seed)
    worker = f"task_{source_index:03d}_grouped_wide_ucv_seed_{seed}"
    remote = f"{bucket.rstrip('/')}/{run_id}/workers/{worker}"
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", f"{remote}/SUCCESS.json", str(directory/"SUCCESS.json")], check=True)
    path, checksum = source_paths(directory, spec(seed, checkpoint), seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or sha256(path) != checksum:
        subprocess.run(["gcloud", "storage", "cp", f"{remote}/{path.relative_to(directory.resolve())}", str(path)], check=True)
