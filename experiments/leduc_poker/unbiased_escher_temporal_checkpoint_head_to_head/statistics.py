"""Seed-level inference for exact temporal head-to-head values."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats


def exact_one_sided_sign_flip_p(values: Sequence[float]) -> float:
    """Return the exact randomisation p-value for a positive mean."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    observed = float(np.mean(arr))
    null_means = [
        float(np.mean(arr * np.asarray(signs, dtype=np.float64)))
        for signs in itertools.product((-1.0, 1.0), repeat=arr.size)
    ]
    return float(np.mean(np.asarray(null_means) >= observed - 1e-15))


def _summary(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else float("nan")
    std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    sem = float(std / math.sqrt(n)) if n > 1 else float("nan")
    if n > 1:
        margin = float(stats.t.ppf(0.975, df=n - 1) * sem)
        ci_low, ci_high = mean - margin, mean + margin
    else:
        ci_low = ci_high = float("nan")
    return {
        "n_seeds": n,
        "mean_ev": mean,
        "sample_std": std,
        "standard_error": sem,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "exact_one_sided_sign_flip_p": exact_one_sided_sign_flip_p(arr),
        "positive_seed_fraction": float(np.mean(arr > 0.0)) if n else float("nan"),
    }


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(p))
    ordered = finite_indices[np.argsort(p[finite_indices])]
    running = 0.0
    family_size = len(ordered)
    for rank, original_index in enumerate(ordered):
        candidate = min(1.0, float((family_size - rank) * p[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def build_inference_tables(
    pairwise_rows: Sequence[Mapping[str, object]],
    checkpoint_schedule: Sequence[int],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Build per-seed effects, headline summaries and pairwise tests."""
    schedule = [int(value) for value in checkpoint_schedule]
    order = {checkpoint: index for index, checkpoint in enumerate(schedule)}
    later_by_seed: Dict[str, List[float]] = defaultdict(list)
    adjacent_by_seed: Dict[str, List[float]] = defaultdict(list)
    final_first_by_seed: Dict[str, float] = {}
    pair_values: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(dict)

    for row in pairwise_rows:
        seed = str(row["seed"])
        checkpoint_a = int(row["checkpoint_a"])
        checkpoint_b = int(row["checkpoint_b"])
        if checkpoint_a not in order or checkpoint_b not in order:
            continue
        if order[checkpoint_a] <= order[checkpoint_b]:
            continue
        ev = float(row["A_EV_seat_averaged"])
        later_by_seed[seed].append(ev)
        if order[checkpoint_a] == order[checkpoint_b] + 1:
            adjacent_by_seed[seed].append(ev)
        pair_values[(checkpoint_a, checkpoint_b)][seed] = ev
        if checkpoint_a == schedule[-1] and checkpoint_b == schedule[0]:
            final_first_by_seed[seed] = ev

    seed_rows = []
    for seed in sorted(later_by_seed, key=lambda value: int(value)):
        values = np.asarray(later_by_seed[seed], dtype=np.float64)
        adjacent_values = np.asarray(adjacent_by_seed[seed], dtype=np.float64)
        seed_rows.append(
            {
                "seed": int(seed),
                "num_later_vs_earlier_pairs": int(values.size),
                "mean_later_vs_earlier_ev": float(np.mean(values)),
                "minimum_later_vs_earlier_ev": float(np.min(values)),
                "maximum_later_vs_earlier_ev": float(np.max(values)),
                "positive_pair_fraction": float(np.mean(values > 0.0)),
                "mean_adjacent_checkpoint_ev": float(np.mean(adjacent_values)),
                "positive_adjacent_fraction": float(
                    np.mean(adjacent_values > 0.0)
                ),
                "final_vs_first_ev": float(
                    final_first_by_seed.get(seed, float("nan"))
                ),
            }
        )

    summary_rows = [
        {
            "estimand": "seed_mean_ev_later_vs_all_earlier_checkpoints",
            **_summary([row["mean_later_vs_earlier_ev"] for row in seed_rows]),
        },
        {
            "estimand": "seed_mean_ev_vs_immediately_previous_checkpoint",
            **_summary([row["mean_adjacent_checkpoint_ev"] for row in seed_rows]),
        },
        {
            "estimand": "final_checkpoint_ev_vs_first_checkpoint",
            **_summary([row["final_vs_first_ev"] for row in seed_rows]),
        },
    ]

    pair_rows: List[dict] = []
    for later, earlier in sorted(
        pair_values,
        key=lambda pair: (order[pair[0]], order[pair[1]]),
    ):
        by_seed = pair_values[(later, earlier)]
        values = [
            by_seed[seed] for seed in sorted(by_seed, key=lambda value: int(value))
        ]
        pair_rows.append(
            {
                "later_checkpoint": int(later),
                "earlier_checkpoint": int(earlier),
                **_summary(values),
            }
        )
    adjusted = _holm_adjust(
        [row["exact_one_sided_sign_flip_p"] for row in pair_rows]
    )
    for row, adjusted_p in zip(pair_rows, adjusted):
        row["holm_adjusted_p"] = float(adjusted_p)

    return seed_rows, summary_rows, pair_rows


__all__ = ["build_inference_tables", "exact_one_sided_sign_flip_p"]
