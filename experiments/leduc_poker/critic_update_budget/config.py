"""Frozen three-seed, four-arm, iteration-matched Leduc budget study."""
from copy import deepcopy
import hashlib

from experiments.leduc_poker.cached_grouped_ucv_36h.config import CANDIDATE_CONFIG

EXPERIMENT_ID = 51
ARMS = {"A": 10000, "B": 2500, "C": 500, "D": 0}
SEEDS = tuple(int.from_bytes(hashlib.sha256(f"ucv-critic-budget-v1-{i}".encode()).digest()[:4], "big") for i in range(3))
PILOT_SEED = 720511
ITERATIONS = 600
# Exploratory practical-loss threshold in NashConv/2 units, not a power claim.
NONINFERIORITY_MARGIN = 0.005


def schedule(smoke=False):
    if smoke:
        return {"end": 6, "evaluate": [2,4,5,6], "late": [4,5,6],
                "diagnose": [6], "save": [2,4,6], "snapshots": [4,6]}
    return {"end": ITERATIONS, "evaluate": list(range(50,601,50)),
            "late": [450,500,550,600], "diagnose": [100,300,600],
            "save": list(range(100,601,100)), "snapshots": [300,600]}


def configuration(arm, smoke=False):
    config = deepcopy(CANDIDATE_CONFIG)
    config.update(max_num_iterations=schedule(smoke)["end"],
                  baseline_network_train_steps=ARMS[arm], zero_critic_control=arm=="D",
                  evaluate_initial_policy=False, early_evaluation_node_thresholds=(),
                  evaluation_frequency=1, preserve_evaluation_rng=True)
    if smoke:
        from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _smoke_overrides
        _smoke_overrides(config)
        config.update(max_num_iterations=6, baseline_network_train_steps={"A":4,"B":2,"C":1,"D":0}[arm],
                      evaluate_initial_policy=False, early_evaluation_node_thresholds=())
    return config


def tasks(smoke=False):
    return [(arm, seed) for seed in ((PILOT_SEED,) if smoke else SEEDS) for arm in ARMS]


def contract(smoke=False):
    return {"experiment":EXPERIMENT_ID, "arms":ARMS, "seeds":[PILOT_SEED] if smoke else SEEDS,
            "schedule":schedule(smoke), "smoke":smoke,
            "primary":"late-window exact-average exploitability; B-A, C-A and D-A",
            "inference":"exploratory paired training seeds; three simultaneous 98.333% t intervals",
            "noninferiority_margin":NONINFERIORITY_MARGIN,
            "baseline":"Experiment 35 learning configuration with Experiment 48 target caching",
            "sampler":"native residual-standard-deviation sampler with 0.2 uniform floor",
            "calibration":"native learned residual calibration, including in zero-control arm",
            "policy_fitting":"native grouped-wide reset fit every completed iteration",
            "zero_control":"identically zero Q values; no critic updates/cache/history refresh; calibration retained",
            "scope":"fixed iterations/trajectories, not equal elapsed training time"}
