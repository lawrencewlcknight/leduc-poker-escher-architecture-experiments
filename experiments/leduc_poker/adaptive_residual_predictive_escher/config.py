"""Configuration for Experiment 3 at Experiment 1 node budgets."""

from __future__ import annotations

from pathlib import Path


DEFAULT_SEEDS = [0, 1, 2]
ALGORITHM_ID = "adaptive_residual_predictive_escher"
ALGORITHM_LABEL = "Adaptive Residual Predictive ESCHER"

# Exact per-seed Experiment 1 ESCHER node totals. The new solver stops after
# the first complete outer iteration crossing its paired target, just as the
# Experiment 1 VR arms did.
EXPERIMENT_1_NODE_TARGETS = {
    0: 942_635,
    1: 939_834,
    2: 962_274,
}

REFERENCE_CURVES = Path(__file__).with_name("experiment1_checkpoint_curves.csv")

# Start from the paper-author settings used by the VR-DeepPDCFR+ arm in
# Experiment 1, changing only mechanisms that define the new architecture.
ADAPTIVE_CONFIG = {
    "game_name": "leduc_poker",
    "advantage_buffer_size": 1_000_000,
    "ave_policy_buffer_size": 1_000_000,
    "baseline_buffer_size": 1_000_000,
    "learning_rate": 1e-3,
    "num_traversals": 10_000,
    "advantage_network_train_steps": 750,
    "ave_policy_network_train_steps": 5_000,
    "baseline_network_train_steps": 10_000,
    "advantage_batch_size": 2_048,
    "ave_policy_batch_size": 2_048,
    "baseline_batch_size": 2_048,
    "num_layers": 3,
    "num_hiddens": 64,
    "reinitialize_advantage_networks": False,
    "reinitialize_imm_regret_networks": True,
    "use_regret_matching_argmax": True,
    "epsilon": 0.6,
    "fit_advantage": True,
    "use_baseline": True,
    "alpha": 2.3,
    "gamma": 2.0,
    "device": "cpu",
    "evaluation_frequency": 1,
    "max_num_iterations": 100,
    "preserve_evaluation_rng": True,
    "evaluate_initial_policy": True,
    "early_evaluation_node_thresholds": (10_000,),
    # New architecture: predictable residual calibration plus an asymptotic
    # floor. At iteration 1 the floor is 0.2; at iterations 3, 5 and 7 it is
    # 0.6, 0.7333 and 0.8. Residual uncertainty can raise it further.
    "lambda_start": 0.2,
    "lambda_schedule_half_life": 2.0,
    "lambda_schedule_power": 1.0,
    "lambda_residual_ema_decay": 0.99,
    "lambda_residual_scale": 0.25,
    "lambda_initial_residual": 1.0,
    "q_gradient_clip_norm": 10.0,
    "sampling_mode": "fixed_uniform",
}

EXPERIMENT_1_SOURCE = {
    "batch_job": (
        "projects/clever-overview-399515/locations/europe-west1/jobs/"
        "leduc-escher-arch-exp1-20260716-223327"
    ),
    "run_directory": "escher_vs_vr_deep_cfr_matched_nodes_20260716_213639",
    "algorithms": ["escher_exp28", "vr_deep_dcfr_plus", "vr_deep_pdcfr_plus"],
    "seeds": DEFAULT_SEEDS,
}
