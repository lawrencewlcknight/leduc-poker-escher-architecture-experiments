"""Exact policy-surgery primitives for the causal rare-state audit."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata

from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import (
    build_policy_table,
    exact_exploitability,
)
from experiments.leduc_poker.average_policy_redistillation.distill import (
    SourceData,
    empirical_table,
    evaluate_model_state,
)

from .config import (
    RANKING_LABELS,
    RANKING_ORDER,
    REPAIR_FRACTIONS,
    SOURCE_CANDIDATE_ID,
    random_ranking_replicates,
)


SOURCE_STATE_TYPE = "experiment_29_full_training_state"


def _tensor_key(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _policy_key(state) -> tuple[int, str]:
    player = int(state.current_player())
    return player, str(state.information_state_string(player))


def extract_promoted_source(
    payload: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_id: str,
    source_checkpoint_id: str,
) -> SourceData:
    """Convert one Experiment 29 continuation state into common source data."""

    if payload.get("type") != SOURCE_STATE_TYPE:
        raise ValueError("Source is not an Experiment 29 continuation state")
    if payload.get("variant_id") != SOURCE_CANDIDATE_ID:
        raise ValueError("Source is not the promoted UCV candidate")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError("Source seed differs from the audit worker contract")
    if payload.get("checkpoint_id") != source_checkpoint_id:
        raise ValueError("Source checkpoint differs from the audit worker contract")

    config = payload["config"]
    buffer = payload["average_policy_trainer"]["buffer"]
    size = int(buffer["size"])
    if size <= 0:
        raise ValueError("Average-policy reservoir is empty")

    exact = payload["exact_average_strategy"]
    exact_table = {}
    for key, numerator in exact["numerators"].items():
        denominator = float(exact["denominators"][key])
        if denominator > 0.0:
            exact_table[key] = np.asarray(numerator, dtype=np.float64) / denominator

    return SourceData(
        checkpoint_id=checkpoint_id,
        seed=int(seed),
        iteration=int(payload["solver"]["num_iteration"]),
        gamma=float(config["gamma"]),
        network_layers=tuple(
            int(config["num_hiddens"]) for _ in range(int(config["num_layers"]))
        ),
        learning_rate=float(config["learning_rate"]),
        batch_size=int(config["ave_policy_batch_size"]),
        train_steps=int(config["ave_policy_network_train_steps"]),
        infostates=np.asarray(buffer["infostate"][:size], dtype=np.float32),
        policies=np.asarray(buffer["q_value"][:size], dtype=np.float32),
        legal_masks=np.asarray(buffer["q_value_mask"][:size], dtype=np.float32),
        iterations=np.asarray(buffer["iteration"][:size], dtype=np.float32).reshape(-1),
        exact_table=exact_table,
        exact_denominators={
            key: float(value) for key, value in exact["denominators"].items()
        },
        source_model_state={
            name: tensor.detach().cpu().clone()
            for name, tensor in payload["average_policy_trainer"]["model"].items()
        },
    )


def _infoset_catalog(game) -> dict:
    catalog: dict[tuple[int, str], dict[str, Any]] = {}

    def walk(state) -> None:
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                walk(state.child(action))
            return

        key = _policy_key(state)
        tensor = np.asarray(state.information_state_tensor(), dtype=np.float32)
        legal_actions = tuple(int(action) for action in state.legal_actions())
        row = catalog.get(key)
        if row is None:
            catalog[key] = {
                "tensor": tensor,
                "legal_actions": legal_actions,
                "representative_history": " ".join(
                    str(int(action)) for action in state.history()
                ),
                "decision_depth": int(len(state.history())),
                "num_histories": 1,
            }
        else:
            if (
                row["legal_actions"] != legal_actions
                or not np.array_equal(row["tensor"], tensor)
            ):
                raise ValueError(f"Inconsistent representation inside infoset {key!r}")
            row["num_histories"] += 1

        for action in legal_actions:
            walk(state.child(action))

    walk(game.new_initial_state())
    return catalog


def _full_exact_table(source: SourceData, catalog: Mapping) -> dict:
    action_size = int(source.policies.shape[1])
    table = {}
    for key, row in catalog.items():
        legal = row["legal_actions"]
        target = source.exact_table.get(key)
        values = np.zeros(action_size, dtype=np.float64)
        if target is None:
            values[list(legal)] = 1.0 / float(len(legal))
        else:
            supplied = np.asarray(target, dtype=np.float64)
            values[list(legal)] = np.maximum(supplied[list(legal)], 0.0)
            mass = float(np.sum(values))
            if not np.isfinite(mass) or mass <= 0.0:
                values[list(legal)] = 1.0 / float(len(legal))
            else:
                values /= mass
        table[key] = values
    return table


def _reservoir_masses(source: SourceData, catalog: Mapping) -> dict:
    by_tensor = {}
    for key, row in catalog.items():
        tensor = _tensor_key(row["tensor"])
        previous = by_tensor.get(tensor)
        if previous is not None and previous != key:
            raise ValueError(
                "Information-state tensor collision prevents a causal rarity audit"
            )
        by_tensor[tensor] = key

    unique, inverse, raw_counts = np.unique(
        source.infostates, axis=0, return_inverse=True, return_counts=True
    )
    iteration_weights = np.power(
        np.maximum(source.iterations, 0.0), source.gamma, dtype=np.float64
    )
    weighted_counts = np.bincount(
        inverse, weights=iteration_weights, minlength=len(unique)
    )
    result = {
        key: {"reservoir_rows": 0, "reservoir_iteration_weight": 0.0}
        for key in catalog
    }
    for index, tensor in enumerate(unique):
        key = by_tensor.get(_tensor_key(tensor))
        if key is None:
            raise ValueError("Reservoir contains an unknown information-state tensor")
        result[key] = {
            "reservoir_rows": int(raw_counts[index]),
            "reservoir_iteration_weight": float(weighted_counts[index]),
        }
    return result


def _normalised_entropy(values: np.ndarray, legal_actions: tuple[int, ...]) -> float:
    probabilities = np.asarray(values, dtype=np.float64)[list(legal_actions)]
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    maximum = math.log(float(len(legal_actions)))
    return entropy / maximum if maximum > 0.0 else 0.0


def _policy_error(
    target: np.ndarray,
    fitted: np.ndarray,
    legal_actions: tuple[int, ...],
) -> tuple[float, float, float]:
    legal = list(legal_actions)
    target_legal = np.asarray(target, dtype=np.float64)[legal]
    fitted_legal = np.asarray(fitted, dtype=np.float64)[legal]
    kl = float(
        np.sum(
            np.where(
                target_legal > 0.0,
                target_legal
                * (
                    np.log(np.clip(target_legal, 1e-12, 1.0))
                    - np.log(np.clip(fitted_legal, 1e-12, 1.0))
                ),
                0.0,
            )
        )
    )
    difference = np.abs(np.asarray(target) - np.asarray(fitted))
    return float(np.sum(difference)), float(np.max(difference)), kl


def _percentiles(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values
    return rankdata(values, method="average") / float(len(values))


def _repair_counts(num_information_sets: int) -> tuple[tuple[float, int], ...]:
    result = []
    for fraction in REPAIR_FRACTIONS:
        if fraction <= 0.0:
            count = 0
        elif fraction >= 1.0:
            count = num_information_sets
        else:
            count = max(1, int(math.ceil(float(fraction) * num_information_sets)))
        if result and count == result[-1][1]:
            continue
        result.append((float(count) / float(num_information_sets), count))
    return tuple(result)


def _ranked_keys(rows: list[dict], ranking_id: str) -> list[tuple[int, str]]:
    if ranking_id == "rarest_first":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["rarity_score"]),
                -float(row["kl_exact_to_neural"]),
                row["player"],
                row["information_set"],
            ),
        )
    elif ranking_id == "largest_policy_error":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["kl_exact_to_neural"]),
                -float(row["single_repair_gain"]),
                row["player"],
                row["information_set"],
            ),
        )
    elif ranking_id == "largest_single_repair_gain":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["single_repair_gain"]),
                -float(row["kl_exact_to_neural"]),
                row["player"],
                row["information_set"],
            ),
        )
    elif ranking_id == "rare_error_impact":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["rare_error_impact_score"]),
                -float(row["single_repair_gain"]),
                row["player"],
                row["information_set"],
            ),
        )
    else:
        raise ValueError(f"Unsupported deterministic ranking {ranking_id}")
    return [(int(row["player"]), str(row["information_set"])) for row in ordered]


@dataclass(frozen=True)
class AuditResult:
    source_summary: dict
    information_sets: list[dict]
    repair_curves: list[dict]


def audit_source(
    source: SourceData,
    game,
    *,
    smoke: bool,
) -> AuditResult:
    """Run exact single-state interventions and cumulative policy repairs."""

    catalog = _infoset_catalog(game)
    exact_table = _full_exact_table(source, catalog)
    neural_value, neural_table = evaluate_model_state(
        source, game, source.source_model_state
    )
    empirical_policy, empirical_value = empirical_table(source, game)
    exact_value = exact_exploitability(game, exact_table)
    masses = _reservoir_masses(source, catalog)

    total_rows = max(sum(row["reservoir_rows"] for row in masses.values()), 1)
    total_iteration_weight = max(
        sum(row["reservoir_iteration_weight"] for row in masses.values()), 1e-30
    )
    total_exact_weight = max(
        sum(float(source.exact_denominators.get(key, 0.0)) for key in catalog),
        1e-30,
    )
    positive_mass = [
        row["reservoir_iteration_weight"] / total_iteration_weight
        for row in masses.values()
        if row["reservoir_iteration_weight"] > 0.0
    ]
    zero_floor = (
        min(positive_mass) * 0.5
        if positive_mass
        else 1.0 / float(max(len(catalog), 1))
    )

    information_rows = []
    for key, catalog_row in sorted(catalog.items()):
        target = np.asarray(exact_table[key], dtype=np.float64)
        fitted = np.asarray(neural_table[key], dtype=np.float64)
        l1, maximum_error, kl = _policy_error(
            target, fitted, catalog_row["legal_actions"]
        )
        mass = masses[key]
        empirical_mass = (
            float(mass["reservoir_iteration_weight"]) / total_iteration_weight
        )
        hybrid = dict(neural_table)
        hybrid[key] = target
        repaired_value = exact_exploitability(game, hybrid)
        gain = float(neural_value - repaired_value)
        information_rows.append(
            {
                "source_seed": int(source.seed),
                "checkpoint_id": source.checkpoint_id,
                "player": int(key[0]),
                "information_set": str(key[1]),
                "representative_history": catalog_row["representative_history"],
                "decision_depth": int(catalog_row["decision_depth"]),
                "num_histories": int(catalog_row["num_histories"]),
                "legal_actions": " ".join(
                    str(action) for action in catalog_row["legal_actions"]
                ),
                "target_observed_during_exact_averaging": key in source.exact_table,
                "reservoir_rows": int(mass["reservoir_rows"]),
                "reservoir_row_fraction": float(mass["reservoir_rows"]) / total_rows,
                "reservoir_iteration_weight": float(
                    mass["reservoir_iteration_weight"]
                ),
                "reservoir_iteration_weight_fraction": empirical_mass,
                "exact_average_weight": float(
                    source.exact_denominators.get(key, 0.0)
                ),
                "exact_average_weight_fraction": float(
                    source.exact_denominators.get(key, 0.0)
                )
                / total_exact_weight,
                "exact_policy": " ".join(f"{value:.12g}" for value in target),
                "neural_policy": " ".join(f"{value:.12g}" for value in fitted),
                "exact_policy_entropy_normalised": _normalised_entropy(
                    target, catalog_row["legal_actions"]
                ),
                "l1_exact_to_neural": l1,
                "max_abs_exact_to_neural": maximum_error,
                "kl_exact_to_neural": kl,
                "single_repair_exploitability": repaired_value,
                "single_repair_gain": gain,
                "single_repair_gain_positive": max(gain, 0.0),
                "rarity_score": -math.log10(max(empirical_mass, zero_floor)),
            }
        )

    rarity = _percentiles([row["rarity_score"] for row in information_rows])
    error = _percentiles(
        [row["kl_exact_to_neural"] for row in information_rows]
    )
    impact = _percentiles(
        [row["single_repair_gain_positive"] for row in information_rows]
    )
    for index, row in enumerate(information_rows):
        row["rarity_percentile"] = float(rarity[index])
        row["policy_error_percentile"] = float(error[index])
        row["strategic_impact_percentile"] = float(impact[index])
        row["rare_error_impact_score"] = float(
            max(rarity[index] * error[index] * impact[index], 0.0) ** (1.0 / 3.0)
        )

    repair_rows = []
    count_schedule = _repair_counts(len(information_rows))
    denominator = float(neural_value - exact_value)

    def record_curve(
        ranking_id: str,
        ordered_keys: list[tuple[int, str]],
        *,
        replicate: int,
    ) -> None:
        row_by_key = {
            (int(row["player"]), str(row["information_set"])): row
            for row in information_rows
        }
        for actual_fraction, count in count_schedule:
            selected = ordered_keys[:count]
            hybrid = dict(neural_table)
            for key in selected:
                hybrid[key] = exact_table[key]
            value = exact_exploitability(game, hybrid)
            recovered = (
                float((neural_value - value) / denominator)
                if abs(denominator) > 1e-15
                else float("nan")
            )
            repair_rows.append(
                {
                    "source_seed": int(source.seed),
                    "checkpoint_id": source.checkpoint_id,
                    "ranking_id": ranking_id,
                    "ranking_label": RANKING_LABELS[ranking_id],
                    "ranking_replicate": int(replicate),
                    "num_information_sets": len(information_rows),
                    "num_repaired": int(count),
                    "information_set_fraction_repaired": actual_fraction,
                    "reservoir_iteration_weight_fraction_repaired": float(
                        sum(
                            row_by_key[key][
                                "reservoir_iteration_weight_fraction"
                            ]
                            for key in selected
                        )
                    ),
                    "exact_average_weight_fraction_repaired": float(
                        sum(
                            row_by_key[key]["exact_average_weight_fraction"]
                            for key in selected
                        )
                    ),
                    "hybrid_exploitability": float(value),
                    "neural_exploitability": float(neural_value),
                    "exact_average_exploitability": float(exact_value),
                    "absolute_gap_recovered": float(neural_value - value),
                    "fraction_distillation_gap_recovered": recovered,
                }
            )

    for ranking_id in RANKING_ORDER:
        if ranking_id == "random":
            for replicate in range(random_ranking_replicates(smoke=smoke)):
                rng = np.random.default_rng(
                    int(source.seed) * 10_000
                    + (24 if source.checkpoint_id == "time_24h" else 36) * 100
                    + replicate
                )
                ordered = [
                    (int(row["player"]), str(row["information_set"]))
                    for row in information_rows
                ]
                rng.shuffle(ordered)
                record_curve(ranking_id, ordered, replicate=replicate)
        else:
            record_curve(
                ranking_id,
                _ranked_keys(information_rows, ranking_id),
                replicate=0,
            )

    complete_rows = [
        row
        for row in repair_rows
        if row["num_repaired"] == len(information_rows)
    ]
    if not complete_rows or any(
        not np.isclose(
            row["hybrid_exploitability"], exact_value, atol=1e-12, rtol=0.0
        )
        for row in complete_rows
    ):
        raise RuntimeError("Complete policy repair did not reproduce exact average")

    source_summary = {
        "source_seed": int(source.seed),
        "checkpoint_id": source.checkpoint_id,
        "iteration": int(source.iteration),
        "reservoir_size": int(len(source.infostates)),
        "num_information_sets": int(len(information_rows)),
        "num_information_sets_without_reservoir_rows": int(
            sum(row["reservoir_rows"] == 0 for row in information_rows)
        ),
        "neural_exploitability": float(neural_value),
        "exact_average_exploitability": float(exact_value),
        "empirical_reservoir_exploitability": float(empirical_value),
        "neural_minus_exact_distillation_gap": denominator,
        "empirical_minus_exact_gap": float(empirical_value - exact_value),
        "sum_positive_single_repair_gain": float(
            sum(row["single_repair_gain_positive"] for row in information_rows)
        ),
        "max_single_repair_gain": float(
            max(row["single_repair_gain"] for row in information_rows)
        ),
    }
    del empirical_policy
    return AuditResult(source_summary, information_rows, repair_rows)


__all__ = [
    "AuditResult",
    "SOURCE_STATE_TYPE",
    "audit_source",
    "extract_promoted_source",
]
