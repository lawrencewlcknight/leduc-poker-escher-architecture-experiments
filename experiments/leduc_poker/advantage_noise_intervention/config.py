"""Prespecified isolated-fit experiment; never a model-free training algorithm."""
import hashlib
from experiments.leduc_poker.grouped_wide_policy_confirmation.config import PRODUCTION_SEEDS

EXPERIMENT_ID = 52
# First three numerically sorted source seeds; no selection on outcomes.
SEEDS = tuple(sorted(PRODUCTION_SEEDS)[:3])
CHECKPOINTS = ("time_24h", "time_36h")
GAMMAS = (1., .5, 0.)
REPETITIONS = 8
TRAVERSALS = 10000
FIT_STEPS = 750
BATCH_SIZE = 2048
SMOKE_SEED = 90252


def tasks(smoke=False):
    return [(SMOKE_SEED, "smoke")] if smoke else [(s, c) for s in SEEDS for c in CHECKPOINTS]


def derived_seed(seed, checkpoint, repetition, player, purpose):
    text = f"ucv-noise-v1/{seed}/{checkpoint}/{repetition}/{player}/{purpose}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def contract(smoke=False, pilot=False):
    if smoke and pilot:
        raise ValueError("Pilot and smoke are distinct")
    return {"experiment": EXPERIMENT_ID,
            "tasks": tasks(smoke), "gammas": GAMMAS,
            "repetitions": 2 if smoke else (1 if pilot else REPETITIONS),
            "traversals_per_player": 16 if smoke else TRAVERSALS,
            "fit_steps": 3 if smoke else FIT_STEPS,
            "batch_size": 8 if smoke else BATCH_SIZE,
            "smoke": smoke, "pilot": pilot,
            "primary": "collection-reach-weighted exact-mean regret MSE; gamma 0 minus 1",
            "secondary": "gamma .5 minus 1; policy TV; current-profile exploitability",
            "sensitivity_eta": "0.01 times reach-weighted RMS information-set mean advantage",
            "sensitive_group": "top ceil(N/4) supported infosets by symmetric coordinate-perturbation TV",
            "scope": "one frozen-player fit; both collections precede fitting; no sequential CFR update",
            "source_selection": "first three numerically sorted Experiment 35 seeds; 24h and 36h",
            "uncertainty": "paired repetitions conditional on saved states; source seeds remain n=3"}
