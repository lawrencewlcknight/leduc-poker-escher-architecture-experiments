"""Leduc architecture Experiment 47: correctness-gated frozen target caching."""
from copy import deepcopy

from experiments.critic_target_cache_cli import main
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _make_solver, _smoke_overrides
from experiments.leduc_poker.grouped_wide_policy_confirmation.config import CANDIDATE_CONFIG

SPEC = {
    "experiment_id": 47,
    "experiment_name": "frozen_critic_target_efficiency",
    "game": "leduc",
    "seeds": [470892, 385626, 145871],
    "source_run_default": "exp35-confirm-20260916-011231",
    "source_experiment": "grouped_wide_policy_confirmation",
    "source_algorithm": "grouped_wide_ucv",
    "checkpoint": "time_36h",
    "state_type": "experiment_35_full_training_state",
    "manifest": "SUCCESS.json",
    "arms": ["recompute", "cache"],
    "timing_repeats": 3,
    "production_critic_updates": 10000,
    "production_batch_size": 2048,
    "production_threads": 8,
}


def make_smoke(seed):
    config = deepcopy(CANDIDATE_CONFIG)
    _smoke_overrides(config)
    config.update(evaluation_frequency=10**9, evaluate_initial_policy=False, early_evaluation_node_thresholds=())
    solver = _make_solver(seed, config)
    for _ in range(2):
        solver.iteration()
    return solver


if __name__ == "__main__":
    main(SPEC, _make_solver, make_smoke)
