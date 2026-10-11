"""Prespecified protocol. No settings are selected using endpoint exploitability."""

from dataclasses import asdict, dataclass

from experiments.leduc_poker.grouped_ucv_temporal_head_to_head.config import SEEDS

EXPERIMENT_ID = 49
SOURCE_URI = (
    "gs://clever-overview-399515-leduc-poker-dream-results/"
    "exp35-confirm-20260916-011231"
)
HOURS = (12, 36)
BUDGETS = (0.1, 0.5, 2.0, 5.0)
PRIMARY_BUDGET = 0.5
REPETITIONS = 8
FIXED_UPDATES = 4096
ARMS = ("cfr_plus_gadget", "rnad_gadget")


@dataclass(frozen=True)
class RNaDConfig:
    eta: float = 0.2
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gadget_learning_rate: float = 1e-3
    adam_beta1: float = 0.99
    adam_beta2: float = 0.999
    batch_size: int = 64
    logit_threshold: float = 2.0
    advantage_clip: float = 5.0
    trace_lambda: float = 0.95
    target_rate: float = 0.001
    critic_pretrain_hands: int = 4096
    critic_validation_hands: int = 1024
    critic_pretrain_updates: int = 1000


def protocol():
    return {
        "experiment_id": EXPERIMENT_ID, "source_uri": SOURCE_URI,
        "training_seeds": list(SEEDS), "hours": list(HOURS),
        "seconds_per_public_root_and_resolving_player": list(BUDGETS),
        "primary_seconds": PRIMARY_BUDGET, "nested_resolver_repetitions": REPETITIONS,
        "arms": list(ARMS), "ablation": "rnad_bayesian at 0.5s, 36h only",
        "fixed_updates": FIXED_UPDATES,
        "fixed_update_scope": "36h, both main arms, repetition 0 only",
        "rnad": asdict(RNaDConfig()), "utility_units": "native OpenSpiel Leduc chips",
        "trigger": "public start of betting round 2; one independent solve per seat",
        "deployment": "resolving seat only; blueprint round 1; all legal public roots",
        "gadget_bounds": "exact opponent counterfactual best-response values vs blueprint",
        "regularizer": "frozen original blueprint; no moving magnet",
        "cfr_initialization": "one unit of positive pseudo-regret per infoset, proportional to blueprint",
        "critic": "separate neural MC value fit; never the UCV training critic",
        "primary_metric": "exact full-game exploitability of each saved stitched resolver realization",
        "stochastic_scope": "average conditional-policy metrics across repeats; not exploitability of an online mixture",
        "selection": "no filtering, blending or stopping by exact exploitability",
        "latency": "includes root construction, bounds, copying, fitting and extraction; soft deadline",
        "panel": "five frozen Experiment 35 36h policies, equal weight; own seed identified",
        "inference_unit": "five training seeds; average the eight nested repeats first",
        "scope": "controlled Leduc optimizer audit, not a large-game implementation or safety proof",
    }
