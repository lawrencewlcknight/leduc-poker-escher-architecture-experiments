"""Full continuation including calibration, all component RNGs and update counts."""
from copy import deepcopy
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    build_training_state, restore_training_state)

STATE_TYPE = "experiment_51_full_training_state"


def pack(solver, arm, seed, config, identity, progress):
    payload = build_training_state(solver, variant_id=arm, seed=seed,
        checkpoint_id=f"iteration_{solver.num_iteration}", active_seconds=0.,
        repository_commit=identity, config=config, captured_snapshots=[],
        state_type=STATE_TYPE, experiment_name="critic_update_budget")
    payload["mechanism"] = {"streams": deepcopy(solver.streams),
        "fit_diagnostics": [deepcopy(m.fit_diagnostics) for m in solver.q_value_trainer.members],
        "updates": [m.total_updates for m in solver.q_value_trainer.members],
        "policy_fit_count":solver.policy_fit_count,"last_policy_fit_iteration":solver.last_policy_fit_iteration,
        "component_seconds": dict(solver.component_seconds), "progress": deepcopy(progress)}
    return payload


def unpack(solver, payload, arm, seed, config, identity):
    restore_training_state(solver, payload, variant_id=arm, seed=seed,
        repository_commit=identity, config=config, expected_state_type=STATE_TYPE)
    extra = payload["mechanism"]
    solver.streams.clear(); solver.streams.update(deepcopy(extra["streams"]))
    for member, count in zip(solver.q_value_trainer.members, extra["updates"]):
        member.total_updates=count
    solver.policy_fit_count=extra["policy_fit_count"]
    solver.last_policy_fit_iteration=extra["last_policy_fit_iteration"]
    for member, rows in zip(solver.q_value_trainer.members, extra["fit_diagnostics"]):
        member.fit_diagnostics = deepcopy(rows)
    solver.component_seconds = dict(extra["component_seconds"])
    return deepcopy(extra["progress"])
