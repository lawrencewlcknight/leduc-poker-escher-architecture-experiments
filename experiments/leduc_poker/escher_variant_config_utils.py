"""Shared helpers for one-seed ESCHER architecture-variant experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


def parse_variant_ids(value: str | None, variants: List[Dict]) -> List[str]:
    if not value:
        return [variant["variant_id"] for variant in variants]
    return [item.strip() for item in value.split(",") if item.strip()]


def variant_lookup(variants: List[Dict]) -> Dict[str, Dict]:
    return {variant["variant_id"]: dict(variant) for variant in variants}


def make_variant_config(base_config: Dict, variant: Dict) -> Dict:
    config = deepcopy(base_config)
    config.update(variant)

    interval = int(config["check_exploitability_every"])
    intermediate_events = (
        len(range(0, int(config["num_iterations"]) + 1, interval))
        if bool(config["compute_exploitability"])
        else 0
    )
    final_events = 1
    config["intermediate_policy_training_events_expected"] = int(intermediate_events)
    config["final_policy_training_events_expected"] = int(final_events)
    config["total_policy_training_events_expected"] = int(intermediate_events + final_events)
    config["policy_gradient_steps_expected"] = int(
        config["total_policy_training_events_expected"]
        * int(config["policy_network_train_steps"])
    )
    return config
