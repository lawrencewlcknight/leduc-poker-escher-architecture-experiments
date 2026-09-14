"""Frozen contract for Experiment 31."""

from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.config import (
    PRODUCTION_SEEDS,
)


EXPERIMENT_ID = 31
EXPERIMENT_NAME = "tabular_cfr_live_resolving"
SOURCE_EXPERIMENT_ID = 29
SOURCE_CHECKPOINT_ID = "time_36h"
SMOKE_SEEDS = (0,)
RESOLVER_KIND = "cfr_plus"
SEARCH_UPDATE_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SMOKE_SEARCH_UPDATE_BUDGETS = (1, 2)
EXPLORATION = 0.0


def contract_manifest(*, smoke: bool = False) -> dict:
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_checkpoint_id": SOURCE_CHECKPOINT_ID,
        "production_seeds": list(PRODUCTION_SEEDS),
        "resolver_kind": RESOLVER_KIND,
        "search_update_budgets": list(
            SMOKE_SEARCH_UPDATE_BUDGETS if smoke else SEARCH_UPDATE_BUDGETS
        ),
        "resolver_replicates_per_blueprint": 1,
        "resolve_trigger": "start of Leduc second betting round",
        "belief_state": "all compatible private histories weighted by blueprint reach",
        "primary_outcome": "exact whole-game exploitability versus online nodes",
        "safety_status": "range_conditioned_but_no_safe_resolving_gadget",
        "evidence_status": "post_selection_forward_search_development",
    }


__all__ = [
    "EXPERIMENT_ID",
    "EXPERIMENT_NAME",
    "EXPLORATION",
    "PRODUCTION_SEEDS",
    "RESOLVER_KIND",
    "SEARCH_UPDATE_BUDGETS",
    "SMOKE_SEARCH_UPDATE_BUDGETS",
    "SMOKE_SEEDS",
    "SOURCE_CHECKPOINT_ID",
    "contract_manifest",
]
