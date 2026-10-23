"""Prespecified, iteration-matched Leduc mechanism experiment (not a league)."""
from copy import deepcopy
import hashlib

from experiments.leduc_poker.grouped_wide_policy_confirmation.config import CANDIDATE_CONFIG

EXPERIMENT_ID = 50
ARMS = {"A": (1, 1), "B": (4, 1), "C": (1, 4), "D": (4, 4)}

# Retain the first three prespecified seed blocks; no outcome-based selection.
SEEDS = tuple(int.from_bytes(hashlib.sha256(f"ucv-critic-roles-confirm-v1-{i}".encode()).digest()[:4], "big") for i in range(3))
PILOT_SEED = 720501
ITERATIONS = 600
MINIMUM_WORTHWHILE_EFFECT = 0.005


def schedule(smoke=False):
    return {"end": 6, "evaluate": [2, 4, 5, 6], "late": [4, 5, 6],
            "diagnose": [6], "fit": [3, 6], "save": [2, 4, 6]} if smoke else {
        "end": ITERATIONS, "evaluate": list(range(50, 601, 50)),
        "late": [450, 500, 550, 600], "diagnose": [100, 300, 600],
        "fit": [300, 600], "save": list(range(100, 601, 100))}


def configuration(arm, smoke=False):
    td, collection = ARMS[arm]
    c = deepcopy(CANDIDATE_CONFIG)
    c.update(max_num_iterations=schedule(smoke)["end"],
             critic_target_average_window=4, critic_td_target_window=td,
             critic_collection_control_window=collection,
             use_residual_calibration=False,
             # Base-class validation requires this when calibration is absent.
             # The explicit overridden proposal below uses proposal_uniform_mass.
             sampling_uniform_floor_mass=1.0, proposal_uniform_mass=0.2,
             evaluation_frequency=10**9, evaluate_initial_policy=False,
             early_evaluation_node_thresholds=(), preserve_evaluation_rng=True)
    if smoke:
        from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _smoke_overrides
        _smoke_overrides(c)
        c.update(max_num_iterations=schedule(True)["end"], evaluation_frequency=10**9,
                 evaluate_initial_policy=False, early_evaluation_node_thresholds=())
    return c


def tasks(smoke=False):
    return [(arm, seed) for seed in ((PILOT_SEED,) if smoke else SEEDS) for arm in ARMS]


def contract(smoke=False):
    return {"experiment": EXPERIMENT_ID, "arms": ARMS, "seeds": [PILOT_SEED] if smoke else SEEDS,
            "schedule": schedule(smoke), "minimum_worthwhile_effect": MINIMUM_WORTHWHILE_EFFECT,
            "primary": "paired D-A late-window exact-average exploitability",
            "proposal": "0.2 uniform + 0.8 current strategy at traverser; opponent on-policy",
            "baseline": "Leduc 35 with per-fit target cache from 47/48",
            "scope": "fixed-proposal derivative; not production adaptive-sampler confirmation",
            "smoke": smoke}
