"""Frozen contract for Experiment 32."""

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    PRODUCTION_SEEDS,
)


EXPERIMENT_ID = 32
EXPERIMENT_NAME = "ucv_live_resolving"
SOURCE_EXPERIMENT_ID = 29
SOURCE_EXPERIMENT_31_ID = 31
SOURCE_CHECKPOINT_ID = "time_36h"
SMOKE_SEEDS = (0,)
RESOLVER_KIND = "ucv_external_sampling"
RESOLVER_KINDS = ("external_sampling", "ucv_external_sampling")
SEARCH_UPDATE_BUDGETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SMOKE_SEARCH_UPDATE_BUDGETS = (2, 4)
RESOLVER_REPLICATES = 8
SMOKE_RESOLVER_REPLICATES = 2
EXPLORATION = 0.05
RESOLVER_SEED_OFFSET = 32_000_000


def resolver_seeds(source_seed: int, *, smoke: bool = False) -> tuple[int, ...]:
    count = SMOKE_RESOLVER_REPLICATES if smoke else RESOLVER_REPLICATES
    return tuple(
        RESOLVER_SEED_OFFSET + int(source_seed) * 100 + replicate
        for replicate in range(count)
    )


def contract_manifest(*, smoke: bool = False) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_experiment_31_id": SOURCE_EXPERIMENT_31_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "production_seeds": list(PRODUCTION_SEEDS),
        "resolver_kind": RESOLVER_KIND,
        "resolver_arms": list(RESOLVER_KINDS),
        "search_update_budgets": list(
            SMOKE_SEARCH_UPDATE_BUDGETS if smoke else SEARCH_UPDATE_BUDGETS
        ),
        "resolver_replicates_per_blueprint": (
            SMOKE_RESOLVER_REPLICATES if smoke else RESOLVER_REPLICATES
        ),
        "exploration": EXPLORATION,
        "control_variate": (
            "pre-update running action-value baseline at sampled opponent nodes"
        ),
        "resolve_trigger": "start of Leduc second betting round",
        "belief_state": "all compatible private histories weighted by blueprint reach",
        "primary_outcomes": [
            "exact whole-game exploitability versus online nodes",
            "within-blueprint resolver variance",
        ],
        "safety_status": "range_conditioned_but_no_safe_resolving_gadget",
        "evidence_status": "post_selection_forward_search_development",
    }


__all__ = [
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPLORATION",
    "PRODUCTION_SEEDS",
    "RESOLVER_KIND",
    "RESOLVER_REPLICATES",
    "RESOLVER_KINDS",
    "SEARCH_UPDATE_BUDGETS",
    "SMOKE_SEARCH_UPDATE_BUDGETS",
    "SMOKE_SEEDS",
    "SOURCE_CHECKPOINT_ID",
    "contract_manifest",
    "resolver_seeds",
]
