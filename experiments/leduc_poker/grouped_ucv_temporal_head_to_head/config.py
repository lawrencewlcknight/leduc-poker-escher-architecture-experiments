"""Frozen evaluation contract; no solver or training is invoked."""

EXPERIMENT_ID = 45
EXPERIMENT_NAME = "grouped_ucv_temporal_head_to_head"
CANDIDATE_ID = "grouped_wide_ucv"
SOURCE_RUN_ID = "exp35-confirm-20260916-011231"
SEEDS = (470892, 385626, 145871, 902492, 318362)
HOURS = tuple(range(2, 37, 2))
SMOKE_SEEDS = (470892,)
SMOKE_HOURS = (2, 4, 36)
EXPLOITABILITY_TOLERANCE = 1e-5
EXPECTED_CONFIG = {
    "game_name": "leduc_poker",
    "fixed_control_variate_beta": 1.0,
    "q_ensemble_size": 2,
    "use_instantaneous_predictor": False,
    "critic_target_average_window": 4,
    "average_policy_loss": "grouped_soft_target_cross_entropy",
    "average_policy_network_layers": [136, 136, 136],
    "average_policy_learning_rate": 0.003,
    "average_policy_train_steps": 20000,
    "average_policy_reset_each_fit": True,
}


def checkpoint_id(hour):
    return f"time_{hour:02d}h"


def source_task(seed):
    return f"task_{SEEDS.index(seed):03d}_{CANDIDATE_ID}_seed_{seed}"
