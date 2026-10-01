"""Frozen Experiment 48 contract. Do not modify historical experiments."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from experiments.leduc_poker.grouped_wide_policy_confirmation import config as base

EXPERIMENT_ID = 48
EXPERIMENT_NAME = "cached_grouped_ucv_36h"
# Preserve the playable-policy/diagnostic schema; use a distinct analysis ID.
CANDIDATE_ID = base.CANDIDATE_ID
ALGORITHM_ID = "cached_grouped_wide_ucv"
LABEL = "UCV-ESCHER (grouped policy, cached critics)"
PRODUCTION_SEEDS = base.PRODUCTION_SEEDS
SMOKE_SEEDS = (0,)
THREADS = 8
CANDIDATE_CONFIG = deepcopy(base.CANDIDATE_CONFIG)
# Engineering safety ceiling only; the stopping rule is 36 active hours.
CANDIDATE_CONFIG["max_num_iterations"] = 2000
CANDIDATE_CONFIG["cache_frozen_critic_targets"] = True
REFERENCE_FILE = Path(__file__).with_name("frozen_reference.json")
REFERENCE_SHA256 = "e67cb6e474c89571c2ce6be0d890974536298ad04482efb2c8fb5c4b882bfbf8"


def checkpoint_schedule(smoke=False):
    return base.checkpoint_schedule(smoke=smoke)


def load_reference():
    data = REFERENCE_FILE.read_bytes()
    if hashlib.sha256(data).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Experiment 48 historical reference checksum mismatch")
    return json.loads(data)


def contract(smoke=False):
    return {
        "experiment_id": EXPERIMENT_ID, "experiment_name": EXPERIMENT_NAME,
        "candidate_id": CANDIDATE_ID, "algorithm_id": ALGORITHM_ID,
        "candidate_config": CANDIDATE_CONFIG,
        "production_seeds": list(SMOKE_SEEDS if smoke else PRODUCTION_SEEDS),
        "target_active_hours": 36, "checkpoint_interval_hours": 2,
        "target_nodes": 15_000_000, "threads": THREADS,
        "training_state_retention": "none", "reference_sha256": REFERENCE_SHA256,
        "evidence_status": "same-seed historical efficiency follow-up; not fresh confirmation",
        "clock": "Experiment 35 recorded-active clock, including candidate policy fitting; "
                 "subtract snapshot persistence, factorial diagnostics and paired legacy diagnostics",
        "only_learning_change": "rebuild frozen critic targets once per critic fitting call",
    }
