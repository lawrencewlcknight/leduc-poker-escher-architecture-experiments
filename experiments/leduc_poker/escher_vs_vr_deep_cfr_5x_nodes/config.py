"""Configuration for the five-times-longer matched-node comparison."""

from __future__ import annotations

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXP28_DEFAULT_CONFIG,
)
from experiments.leduc_poker.escher_variant_config_utils import make_variant_config

DEFAULT_SEEDS = [0, 1, 2]
NODE_BUDGET_MULTIPLIER = 5
# ESCHER's solver executes ``num_iterations + 1`` training cycles. Experiment
# 28 uses 81 cycles (0..80), so 404 gives exactly 405 cycles: five times as many.
ESCHER_NUM_ITERATIONS = (
    NODE_BUDGET_MULTIPLIER * (int(EXP28_DEFAULT_CONFIG["num_iterations"]) + 1) - 1
)
EXPECTED_BATCH_RUNTIME_HOURS = 24
BATCH_TIMEOUT_SECONDS = 36 * 60 * 60

ALGORITHMS = {
    "escher_exp28": {
        "algorithm_id": "escher_exp28",
        "algorithm_label": "ESCHER (Experiment 28, 5x nodes)",
    },
    "vr_deep_dcfr_plus": {
        "algorithm_id": "vr_deep_dcfr_plus",
        "algorithm_label": "VR-DeepDCFR+",
        "class_name": "VRDeepDCFRPlus",
        "alpha": 2.0,
        "gamma": 2.0,
        "reinitialize_imm_regret_networks": None,
    },
    "vr_deep_pdcfr_plus": {
        "algorithm_id": "vr_deep_pdcfr_plus",
        "algorithm_label": "VR-DeepPDCFR+",
        "class_name": "VRDeepPDCFRPlus",
        "alpha": 2.3,
        "gamma": 2.0,
        "reinitialize_imm_regret_networks": True,
    },
}

ESCHER_CONFIG = make_variant_config(
    EXP28_DEFAULT_CONFIG,
    {
        "experiment_name": "leduc_escher_vs_vr_deep_cfr_5x_nodes",
        "variant_id": ALGORITHMS["escher_exp28"]["algorithm_id"],
        "variant_label": ALGORITHMS["escher_exp28"]["algorithm_label"],
        "variant_description": (
            "Experiment 28 architecture and training settings, extended from 81 "
            "to 405 training cycles to target five times the Experiment 1 nodes."
        ),
        "num_iterations": ESCHER_NUM_ITERATIONS,
        "evaluate_initial_policy": True,
    },
)

# Table 2 settings from the paper. The released YAML uses a 150,000 advantage
# buffer and 1,000 history-value steps; those discrepancies are recorded in the
# experiment metadata and README rather than silently substituted here.
VR_PAPER_CONFIG = {
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
    "use_regret_matching_argmax": True,
    "epsilon": 0.6,
    "fit_advantage": True,
    "use_baseline": True,
    "device": "cpu",
    # Retain one evaluation per outer iteration, plus the explicitly scheduled
    # zero-node and approximately 10k-node early checkpoints below.
    "evaluation_frequency": 1,
    # Safety cap only; matched ESCHER nodes are the primary stopping condition.
    "max_num_iterations": 100,
    "preserve_evaluation_rng": True,
    "evaluate_initial_policy": True,
    "early_evaluation_node_thresholds": (10_000,),
}

UPSTREAM = {
    "repository": "https://github.com/rpSebastian/DeepPDCFR",
    "commit": "9f156c9fcdac7f8c9bd0debf94c9432d222858d3",
    "retrieved": "2026-07-16",
    "paper": "https://doi.org/10.1609/aaai.v40i20.38780",
    "extended_paper": "https://arxiv.org/abs/2511.08174",
}
