"""Frozen contracts for the four Experiment 29 post-training studies."""

from __future__ import annotations

GAME_NAME = "leduc_poker"
SOURCE_CHECKPOINT = "time_36h"
# Keep the controller-side experiment contract importable using only the Python
# standard library.  These are the frozen Experiment 29 production seeds; do
# not import Experiment 29's training config here because that transitively
# imports NumPy and PyTorch on the deliberately minimal controller VM.
PRODUCTION_SEEDS = (104729, 130363, 155921, 181081, 205759)
SMOKE_SEEDS = (0,)

METHODS = {
    "best_response_guided_repair": {
        "experiment_id": 36,
        "experiment_name": "best_response_guided_supervised_repair",
        "short_name": "br-repair",
        "label": "Best-response-guided supervised repair",
        "arms": {
            "br_weight_1": {"learning_rate": 3e-4, "br_weight": 1.0},
            "br_weight_10": {"learning_rate": 3e-4, "br_weight": 10.0},
            "br_weight_100": {"learning_rate": 3e-4, "br_weight": 100.0},
        },
        "updates": 400,
        "refresh_interval": 20,
    },
    "kl_exploitability_descent": {
        "experiment_id": 37,
        "experiment_name": "kl_constrained_exploitability_descent",
        "short_name": "kl-ed",
        "label": "KL-constrained exploitability descent",
        "arms": {
            "kl_0_01": {"learning_rate": 1e-4, "kl_coefficient": 0.01},
            "kl_0_1": {"learning_rate": 1e-4, "kl_coefficient": 0.1},
            "kl_1_0": {"learning_rate": 1e-4, "kl_coefficient": 1.0},
        },
        "updates": 300,
        "refresh_interval": 10,
    },
    "neurd_fine_tuning": {
        "experiment_id": 38,
        "experiment_name": "neural_replicator_dynamics_fine_tuning",
        "short_name": "neurd",
        "label": "NeuRD fine-tuning",
        "arms": {
            "lr_1e_4": {"learning_rate": 1e-4},
            "lr_3e_4": {"learning_rate": 3e-4},
            "lr_1e_3": {"learning_rate": 1e-3},
        },
        "updates": 400,
        "refresh_interval": 1,
    },
    "ppo_self_play": {
        "experiment_id": 39,
        "experiment_name": "vanilla_ppo_self_play_fine_tuning",
        "short_name": "ppo",
        "label": "Vanilla PPO self-play",
        "arms": {
            "lr_1e_4": {"learning_rate": 1e-4},
            "lr_3e_4": {"learning_rate": 3e-4},
            "lr_1e_3": {"learning_rate": 1e-3},
        },
        "updates": 100,
        "rollout_episodes": 1024,
        "ppo_epochs": 4,
        "clip_ratio": 0.2,
        "entropy_coefficient": 0.01,
        "value_coefficient": 0.5,
    },
}


def method_config(method_id: str, *, smoke: bool = False) -> dict:
    if method_id not in METHODS:
        raise ValueError(f"Unknown post-training method {method_id!r}")
    result = {**METHODS[method_id]}
    result["arms"] = {
        arm: dict(values) for arm, values in METHODS[method_id]["arms"].items()
    }
    if smoke:
        result["updates"] = 2
        result["refresh_interval"] = 1
        result["rollout_episodes"] = 32
        result["ppo_epochs"] = 1
    return result


def evaluation_steps(method_id: str, *, smoke: bool = False) -> tuple[int, ...]:
    total = int(method_config(method_id, smoke=smoke)["updates"])
    if smoke:
        return (0, 1, total)
    candidates = (0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400)
    return tuple(value for value in candidates if value <= total)


def task_schedule(seeds=PRODUCTION_SEEDS) -> tuple[int, ...]:
    return tuple(int(seed) for seed in seeds)


def contract_manifest(method_id: str) -> dict:
    config = method_config(method_id)
    return {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "method_id": method_id,
        "method_label": config["label"],
        "source_experiment": 29,
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_seeds": list(PRODUCTION_SEEDS),
        "arms": config["arms"],
        "updates": config["updates"],
        "evidence_status": "post_selection_development_evidence",
        "selection_warning": (
            "Exact Leduc exploitability selects checkpoints and hyperparameters; "
            "any winner requires fresh-seed confirmation."
        ),
    }


__all__ = [
    "GAME_NAME", "METHODS", "PRODUCTION_SEEDS", "SMOKE_SEEDS",
    "SOURCE_CHECKPOINT", "contract_manifest", "evaluation_steps",
    "method_config", "task_schedule",
]
