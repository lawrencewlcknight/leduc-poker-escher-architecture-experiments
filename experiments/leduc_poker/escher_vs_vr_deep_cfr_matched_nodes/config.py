"""Configuration for the paired, matched-node algorithm comparison."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXP28_DEFAULT_CONFIG,
)

DEFAULT_SEEDS = [0, 1, 2]

ALGORITHMS = {
    "escher_exp28": {
        "algorithm_id": "escher_exp28",
        "algorithm_label": "ESCHER (Experiment 28)",
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

ESCHER_CONFIG = deepcopy(EXP28_DEFAULT_CONFIG)
ESCHER_CONFIG.update(
    experiment_name="leduc_escher_vs_vr_deep_cfr_matched_nodes",
    variant_id=ALGORITHMS["escher_exp28"]["algorithm_id"],
    variant_label=ALGORITHMS["escher_exp28"]["algorithm_label"],
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
    # One evaluation per outer iteration is about one point per 100k--150k
    # sampled nodes in Leduc, close to Experiment 28's checkpoint spacing.
    "evaluation_frequency": 1,
    # Safety cap only; matched ESCHER nodes are the primary stopping condition.
    "max_num_iterations": 100,
    "preserve_evaluation_rng": True,
}

UPSTREAM = {
    "repository": "https://github.com/rpSebastian/DeepPDCFR",
    "commit": "9f156c9fcdac7f8c9bd0debf94c9432d222858d3",
    "retrieved": "2026-07-16",
    "paper": "https://doi.org/10.1609/aaai.v40i20.38780",
    "extended_paper": "https://arxiv.org/abs/2511.08174",
}
