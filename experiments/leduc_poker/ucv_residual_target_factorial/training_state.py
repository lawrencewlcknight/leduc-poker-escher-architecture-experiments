"""Exact continuation checkpoints for Experiment 25 workers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1
STATE_TYPE = "experiment_25_full_training_state"


def _cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _reservoir_state(buffer) -> dict[str, Any]:
    size = min(int(buffer.cur_id), int(buffer.buffer_size))
    return {
        "cur_id": int(buffer.cur_id),
        "size": size,
        "infostate": np.asarray(buffer.infostate_buf[:size]).copy(),
        "q_value": np.asarray(buffer.q_value_buf[:size]).copy(),
        "q_value_mask": np.asarray(buffer.q_value_mask_buf[:size]).copy(),
        "iteration": np.asarray(buffer.iteration_buf[:size]).copy(),
    }


def _load_reservoir_state(buffer, state: Mapping[str, Any]) -> None:
    size = int(state["size"])
    if size > int(buffer.buffer_size):
        raise ValueError("Saved reservoir exceeds configured capacity")
    buffer.infostate_buf[:size] = state["infostate"]
    buffer.q_value_buf[:size] = state["q_value"]
    buffer.q_value_mask_buf[:size] = state["q_value_mask"]
    buffer.iteration_buf[:size] = state["iteration"]
    buffer.cur_id = int(state["cur_id"])


def _circular_state(buffer) -> dict[str, Any]:
    size = int(buffer.size)
    occupied = int(buffer.buffer_size) if size == int(buffer.buffer_size) else size
    return {
        "cur_id": int(buffer.cur_id),
        "size": size,
        "history": np.asarray(buffer.history_buf[:occupied]).copy(),
        "action": np.asarray(buffer.action_buf[:occupied]).copy(),
        "next_history": np.asarray(buffer.next_history_buf[:occupied]).copy(),
        "next_state": np.asarray(buffer.next_state_buf[:occupied]).copy(),
        "next_legal_actions_mask": np.asarray(
            buffer.next_legal_actions_mask_buf[:occupied]
        ).copy(),
        "next_player": np.asarray(buffer.next_player_buf[:occupied]).copy(),
        "done": np.asarray(buffer.done_buf[:occupied]).copy(),
        "reward": np.asarray(buffer.reward_buf[:occupied]).copy(),
    }


def _load_circular_state(buffer, state: Mapping[str, Any]) -> None:
    size = int(state["size"])
    occupied = len(state["action"])
    if occupied > int(buffer.buffer_size) or size > int(buffer.buffer_size):
        raise ValueError("Saved circular replay exceeds configured capacity")
    buffer.history_buf[:occupied] = state["history"]
    buffer.action_buf[:occupied] = state["action"]
    buffer.next_history_buf[:occupied] = state["next_history"]
    buffer.next_state_buf[:occupied] = state["next_state"]
    buffer.next_legal_actions_mask_buf[:occupied] = state[
        "next_legal_actions_mask"
    ]
    buffer.next_player_buf[:occupied] = state["next_player"]
    buffer.done_buf[:occupied] = state["done"]
    buffer.reward_buf[:occupied] = state["reward"]
    buffer.cur_id = int(state["cur_id"])
    buffer.size = size


def _calibration_state(buffer) -> dict[str, Any]:
    size = int(buffer.size)
    occupied = int(buffer.capacity) if size == int(buffer.capacity) else size
    return {
        "cursor": int(buffer.cursor),
        "size": size,
        "features": np.asarray(buffer.features[:occupied]).copy(),
        "targets": np.asarray(buffer.targets[:occupied]).copy(),
    }


def _load_calibration_state(buffer, state: Mapping[str, Any]) -> None:
    size = int(state["size"])
    occupied = len(state["targets"])
    if occupied > int(buffer.capacity) or size > int(buffer.capacity):
        raise ValueError("Saved calibration replay exceeds configured capacity")
    buffer.features[:occupied] = state["features"]
    buffer.targets[:occupied] = state["targets"]
    buffer.cursor = int(state["cursor"])
    buffer.size = size


def _trainer_state(trainer, *, include_buffer: bool) -> dict[str, Any]:
    state = {
        "model": _cpu_state_dict(trainer.model),
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
    }
    if hasattr(trainer, "target_model"):
        state["target_model"] = _cpu_state_dict(trainer.target_model)
    if include_buffer:
        state["buffer"] = _reservoir_state(trainer.buffer)
    if hasattr(trainer, "prediction_gate"):
        state["prediction_gate"] = float(trainer.prediction_gate)
    return state


def _load_trainer_state(trainer, state: Mapping[str, Any]) -> None:
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    if "target_model" in state:
        trainer.target_model.load_state_dict(state["target_model"])
    if "buffer" in state:
        _load_reservoir_state(trainer.buffer, state["buffer"])
    if "prediction_gate" in state and hasattr(trainer, "set_prediction_gate"):
        trainer.set_prediction_gate(float(state["prediction_gate"]))


def build_training_state(
    solver,
    *,
    variant_id: str,
    seed: int,
    checkpoint_id: str,
    active_seconds: float,
    repository_commit: str,
    config: Mapping[str, Any],
    captured_snapshots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    calibration = solver.calibration_trainer
    rng_state = solver._checkpoint_resume_rng_state
    if rng_state is None:
        rng_state = solver._capture_rng_state()
    return {
        "schema_version": SCHEMA_VERSION,
        "type": STATE_TYPE,
        "experiment_name": "ucv_residual_target_factorial",
        "variant_id": str(variant_id),
        "seed": int(seed),
        "checkpoint_id": str(checkpoint_id),
        "active_seconds": float(active_seconds),
        "repository_commit": str(repository_commit),
        "config": dict(config),
        "captured_snapshots": [dict(row) for row in captured_snapshots],
        "solver": {
            "num_iteration": int(solver.num_iteration),
            "episode": int(solver.episode),
            "nodes_touched": int(solver.nodes_touched),
            "checkpoint_rows": list(solver.checkpoint_rows),
            "cumulative_experience_collection_seconds": float(
                solver._cumulative_experience_collection_seconds
            ),
            "architecture_stats": dict(solver._architecture_stats),
            "minimum_sample_probability": float(
                solver._minimum_sample_probability
            ),
            "current_regret_policy_learning_rate": float(
                solver.current_regret_policy_learning_rate
            ),
            "q_oracle_diagnostic_rows": list(solver.q_oracle_diagnostic_rows),
            "cumulative_factorial_diagnostic_seconds": float(
                solver._cumulative_factorial_diagnostic_seconds
            ),
        },
        "rng": deepcopy(rng_state),
        "regret_trainers": [
            _trainer_state(trainer, include_buffer=False)
            for trainer in solver.regret_trainers
        ],
        "average_policy_trainer": _trainer_state(
            solver.ave_policy_trainer, include_buffer=True
        ),
        "q_ensemble": {
            "active_fold": int(solver.q_value_trainer.active_fold),
            "active_add_count": int(solver.q_value_trainer._active_add_count),
            "members": [
                {
                    "model": _cpu_state_dict(member.model),
                    "target_model": _cpu_state_dict(member.target_model),
                    "optimizer": deepcopy(member.optimizer.state_dict()),
                    "buffer": _circular_state(member.buffer),
                    "target_version": int(member.target_version),
                    "target_history": [
                        {name: value.detach().cpu().clone() for name, value in row.items()}
                        for row in member.target_history
                    ],
                    "last_target_update_l2": float(member.last_target_update_l2),
                }
                for member in solver.q_value_trainer.members
            ],
        },
        "calibration": (
            None
            if calibration is None
            else {
                "model": _cpu_state_dict(calibration.model),
                "target_model": _cpu_state_dict(calibration.target_model),
                "optimizer": deepcopy(calibration.optimizer.state_dict()),
                "buffer": _calibration_state(calibration.buffer),
                "target_version": int(calibration.target_version),
            }
        ),
        "gate_controller": {
            "gates": np.asarray(solver.gate_controller.gates).copy(),
            "prediction_mse": np.asarray(solver.gate_controller.prediction_mse).copy(),
            "zero_mse": np.asarray(solver.gate_controller.zero_mse).copy(),
            "relative_skill": np.asarray(solver.gate_controller.relative_skill).copy(),
        },
        "exact_average_strategy": {
            "numerators": {
                key: np.asarray(value).copy()
                for key, value in solver.exact_average_strategy.numerators.items()
            },
            "denominators": dict(solver.exact_average_strategy.denominators),
        },
        "online_diagnostics": {
            "information_action": solver._information_action,
            "iteration_information_action": solver._iteration_information_action,
            "beta_histogram": np.asarray(solver._beta_histogram).copy(),
        },
        "logger": {
            "pending": dict(solver.logger._pending),
            "history": list(solver.logger.history),
        },
    }


def save_training_state(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def read_training_state(path: Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def restore_training_state(
    solver,
    payload: Mapping[str, Any],
    *,
    variant_id: str,
    seed: int,
    repository_commit: str,
    config: Mapping[str, Any],
) -> list[dict]:
    if payload.get("type") != STATE_TYPE or int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("Unsupported Experiment 25 training-state schema")
    if payload.get("variant_id") != variant_id or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("Training state belongs to a different task")
    if payload.get("repository_commit") != repository_commit:
        raise ValueError("Training state was produced by a different commit")
    if payload.get("config") != dict(config):
        raise ValueError("Training-state configuration differs from the worker contract")

    core = payload["solver"]
    solver.num_iteration = int(core["num_iteration"])
    solver.episode = int(core["episode"])
    solver.nodes_touched = int(core["nodes_touched"])
    solver.checkpoint_rows = list(core["checkpoint_rows"])
    solver._cumulative_experience_collection_seconds = float(
        core["cumulative_experience_collection_seconds"]
    )
    solver._architecture_stats = dict(core["architecture_stats"])
    solver._minimum_sample_probability = float(core["minimum_sample_probability"])
    solver.current_regret_policy_learning_rate = float(
        core["current_regret_policy_learning_rate"]
    )
    solver.q_oracle_diagnostic_rows = list(core["q_oracle_diagnostic_rows"])
    solver._cumulative_factorial_diagnostic_seconds = float(
        core["cumulative_factorial_diagnostic_seconds"]
    )
    solver._resume_elapsed_seconds = float(payload["active_seconds"])

    for trainer, state in zip(solver.regret_trainers, payload["regret_trainers"]):
        _load_trainer_state(trainer, state)
    _load_trainer_state(solver.ave_policy_trainer, payload["average_policy_trainer"])

    ensemble_state = payload["q_ensemble"]
    ensemble = solver.q_value_trainer
    ensemble.active_fold = int(ensemble_state["active_fold"])
    ensemble._active_add_count = int(ensemble_state["active_add_count"])
    if len(ensemble.members) != len(ensemble_state["members"]):
        raise ValueError("Critic ensemble size differs from saved state")
    for member, state in zip(ensemble.members, ensemble_state["members"]):
        member.model.load_state_dict(state["model"])
        member.target_model.load_state_dict(state["target_model"])
        member.optimizer.load_state_dict(state["optimizer"])
        _load_circular_state(member.buffer, state["buffer"])
        member.target_version = int(state["target_version"])
        member.target_history.clear()
        member.target_history.extend(state["target_history"])
        member.last_target_update_l2 = float(state["last_target_update_l2"])

    calibration_state = payload["calibration"]
    calibration = solver.calibration_trainer
    if (calibration_state is None) != (calibration is None):
        raise ValueError("Calibration presence differs from saved state")
    if calibration is not None:
        calibration.model.load_state_dict(calibration_state["model"])
        calibration.target_model.load_state_dict(calibration_state["target_model"])
        calibration.optimizer.load_state_dict(calibration_state["optimizer"])
        _load_calibration_state(calibration.buffer, calibration_state["buffer"])
        calibration.target_version = int(calibration_state["target_version"])

    gate = payload["gate_controller"]
    solver.gate_controller.gates = np.asarray(gate["gates"]).copy()
    solver.gate_controller.prediction_mse = np.asarray(gate["prediction_mse"]).copy()
    solver.gate_controller.zero_mse = np.asarray(gate["zero_mse"]).copy()
    solver.gate_controller.relative_skill = np.asarray(gate["relative_skill"]).copy()

    exact = payload["exact_average_strategy"]
    solver.exact_average_strategy.numerators = {
        key: np.asarray(value).copy() for key, value in exact["numerators"].items()
    }
    solver.exact_average_strategy.denominators = dict(exact["denominators"])

    diagnostics = payload["online_diagnostics"]
    solver._information_action = diagnostics["information_action"]
    solver._iteration_information_action = diagnostics["iteration_information_action"]
    solver._beta_histogram = np.asarray(diagnostics["beta_histogram"]).copy()
    solver.logger._pending = dict(payload["logger"]["pending"])
    solver.logger.history = list(payload["logger"]["history"])

    # The checkpoint was taken inside evaluation, before the solver restored
    # its evaluation-isolated RNG. Restore that pre-evaluation state exactly.
    solver._restore_rng_state(payload["rng"])
    return [dict(row) for row in payload["captured_snapshots"]]


__all__ = [
    "SCHEMA_VERSION",
    "STATE_TYPE",
    "build_training_state",
    "read_training_state",
    "restore_training_state",
    "save_training_state",
]
