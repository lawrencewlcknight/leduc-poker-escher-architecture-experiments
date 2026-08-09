"""Seed-level inference for Experiment 17 exact head-to-head effects."""

from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


def exact_sign_flip_p(values: Iterable[float], *, two_sided: bool) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    observed = float(np.mean(values))
    direction = 1.0 if observed >= 0 else -1.0
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(float(np.mean(values * np.asarray(signs))))
    permutation = np.asarray(permuted)
    tolerance = 1e-15
    if two_sided:
        return float(np.mean(np.abs(permutation) >= abs(observed) - tolerance))
    return float(
        np.mean(direction * permutation >= abs(observed) - tolerance)
    )


def summary(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    n = int(array.size)
    if not n:
        return {
            "n_seeds": 0,
            "mean_ev": float("nan"),
            "standard_deviation": float("nan"),
            "standard_error": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "two_sided_exact_sign_flip_p": float("nan"),
            "exploratory_directional_exact_p": float("nan"),
        }
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if n > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(n)
    if n > 1:
        margin = float(stats.t.ppf(0.975, n - 1) * standard_error)
    else:
        margin = 0.0
    return {
        "n_seeds": n,
        "mean_ev": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
        "positive_seed_fraction": float(np.mean(array > 0)),
        "equivalent_zero_seed_fraction": float(np.mean(array == 0)),
        "two_sided_exact_sign_flip_p": exact_sign_flip_p(array, two_sided=True),
        "exploratory_directional_exact_p": exact_sign_flip_p(
            array, two_sided=False
        ),
    }


def holm_adjust(rows: list[dict], p_key: str, output_key: str) -> None:
    finite = [
        (index, float(row[p_key]))
        for index, row in enumerate(rows)
        if np.isfinite(float(row[p_key]))
    ]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    adjusted = {}
    total = len(finite)
    for rank, (index, p_value) in enumerate(finite):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[index] = running
    for index, row in enumerate(rows):
        row[output_key] = adjusted.get(index, float("nan"))


__all__ = ["exact_sign_flip_p", "holm_adjust", "summary"]
