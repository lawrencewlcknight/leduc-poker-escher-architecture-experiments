"""Extend the audited full-state format with both critic roles and RNG streams."""
from copy import deepcopy
from .solver import cpu_weights
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    build_training_state, restore_training_state)

STATE_TYPE = "experiment_50_full_training_state"


def pack(solver, arm, seed, config, identity, progress):
    payload = build_training_state(solver, variant_id=arm, seed=seed,
        checkpoint_id=f"iteration_{solver.num_iteration}", active_seconds=0.,
        repository_commit=identity, config=config, captured_snapshots=[],
        state_type=STATE_TYPE, experiment_name="critic_averaging_mechanism")
    payload["mechanism"] = {"streams": deepcopy(solver.streams),
        "fit_diagnostics": [deepcopy(m.fit_diagnostics) for m in solver.q_value_trainer.members],
        "collection_models": [cpu_weights(m.collection_model) for m in solver.q_value_trainer.members],
        "component_seconds": dict(solver.component_seconds), "progress": deepcopy(progress)}
    return payload


def unpack(solver, payload, arm, seed, config, identity):
    restore_training_state(solver, payload, variant_id=arm, seed=seed,
        repository_commit=identity, config=config, expected_state_type=STATE_TYPE)
    extra = payload["mechanism"]
    solver.streams.clear(); solver.streams.update(deepcopy(extra["streams"]))
    for member, weights in zip(solver.q_value_trainer.members, extra["collection_models"]):
        member.collection_model.load_state_dict(weights)
        member.collection_model.eval()
    for member, rows in zip(solver.q_value_trainer.members, extra["fit_diagnostics"]):
        member.fit_diagnostics = deepcopy(rows)
    solver.component_seconds = dict(extra["component_seconds"])
    return deepcopy(extra["progress"])
