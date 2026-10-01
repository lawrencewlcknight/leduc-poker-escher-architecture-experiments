"""Fail closed on complete-iteration differences before submitting five long jobs."""
from copy import deepcopy
import gc
import math
import numpy as np
import torch

from experiments.leduc_poker.ucv_residual_target_factorial.training_state import (
    build_training_state, _reservoir_state, _trainer_state,
)
from experiments.critic_target_cache_benchmark import rng_hash, target_audit
from .worker import make_solver, configuration


def _state(solver):
    state = build_training_state(solver, variant_id="smoke", seed=0, checkpoint_id="smoke",
        active_seconds=0., repository_commit="smoke", config={}, captured_snapshots=[])
    # Wall clocks/logs are deliberately not algorithm state.
    for key in ("checkpoint_rows", "cumulative_experience_collection_seconds",
                "cumulative_factorial_diagnostic_seconds"):
        state["solver"].pop(key)
    state.pop("logger")
    state["regret_replay"] = [_reservoir_state(t.buffer) for t in solver.regret_trainers]
    state["legacy_fit"] = _trainer_state(solver.paired_legacy_policy_trainer, include_buffer=False)
    state["actual_rng"] = solver._capture_rng_state()
    state["sampler_rng"] = [rng_hash(m) for m in solver.q_value_trainer.members]
    state["checkpoint_quality"] = [{k: r[k] for k in (
        "iteration", "nodes_touched", "exp", "neural_average_exploitability",
        "exact_average_exploitability", "legacy_average_exploitability",
        "empirical_reservoir_exploitability")} for r in solver.checkpoint_rows]
    return deepcopy(state)


def assert_exact(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0, equal_nan=True, msg=path)
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right, equal_nan=True):
            raise AssertionError(f"Nonidentical array: {path}")
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"State keys differ: {path}")
        for key in left:
            assert_exact(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"State lengths differ: {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f"{path}/{i}")
    elif isinstance(left, float) and math.isnan(left):
        if not math.isnan(right):
            raise AssertionError(path)
    elif left != right:
        raise AssertionError(f"Nonidentical value: {path}: {left!r} != {right!r}")


def equivalence_smoke(threads=8):
    torch.set_num_threads(threads)
    outputs = []
    for cached in (False, True):
        c = configuration(True)
        c.update(cache_frozen_critic_targets=cached, max_num_iterations=3, num_traversals=16,
                 baseline_batch_size=32, baseline_network_train_steps=3,
                 early_evaluation_node_thresholds=(10,))
        solver = make_solver(0, c)
        solver.solve()
        # Additional production-sized fitting block, including a partial cache
        # chunk. Deterministic replay replication is only a smoke fixture.
        for member in solver.q_value_trainer.members:
            count = len(member.buffer)
            for name in ("history", "next_history", "next_state", "reward", "action",
                         "next_legal_actions_mask", "next_player", "done"):
                data = getattr(member.buffer, name + "_buf")[:count]
                setattr(member.buffer, name + "_buf", data[np.arange(2051) % count].copy())
            member.buffer.size = member.buffer.buffer_size = 2051
            member.buffer.cur_id = 0
            member.batch_size = 2048
            if not target_audit(member, solver.num_iteration, 7919)["allclose"]:
                raise AssertionError("Production-shaped target audit failed")
            member.train_model(solver.num_iteration)
        outputs.append(_state(solver))
        del solver
        gc.collect()
    assert_exact(*outputs)
    return {"passed": True, "bitwise_equal": True, "threads": threads,
            "completed_iterations_per_arm": 3, "integration_critic_batch_size": 32,
            "additional_critic_fit_batch_size": 2048, "additional_critic_fit_rows": 2051,
            "checked": ["regret/critic/calibration/policy weights and optimizers", "target histories",
                        "replay data and cursors", "global and sampling RNG", "exact averages",
                        "gate state", "node counts", "playable policy evaluation values"],
            "scope": "small complete-iteration integration test, not a long-run equivalence theorem"}
