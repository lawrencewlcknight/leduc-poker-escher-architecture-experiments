"""Base configuration helper for new ESCHER architecture experiments."""

from __future__ import annotations

from copy import deepcopy

from experiments.leduc_poker.escher_candidate_architecture_multiseed.config import (
    DEFAULT_CONFIG as EXPERIMENT_28_CONFIG,
)

DEFAULT_SEED = 1234


def make_default_config(experiment_name: str):
    """Return an Experiment 28-equivalent config under a new run name."""
    config = deepcopy(EXPERIMENT_28_CONFIG)
    config["experiment_name"] = experiment_name
    return config
