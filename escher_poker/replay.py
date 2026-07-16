"""Replay buffers used by ESCHER experiments."""

from __future__ import annotations

import heapq
import random
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np


RESERVOIR = "reservoir"
ALL_SAMPLES = "all_samples"
INFOSET_STRATIFIED = "infoset_stratified"
RARE_HISTORY_QUOTA = "rare_history_quota"
COUNTERFACTUAL_REACH_WEIGHTED = "counterfactual_reach_weighted"

VALID_REGRET_REPLAY_MODES = {
    RESERVOIR,
    ALL_SAMPLES,
    INFOSET_STRATIFIED,
    RARE_HISTORY_QUOTA,
    COUNTERFACTUAL_REACH_WEIGHTED,
}


def _distribution_diagnostics(keys, stored_count, stream_count):
    counts = np.asarray(
        list(Counter(key for key in keys if key is not None).values()),
        dtype=np.float64,
    )
    return {
        "stored_count": int(stored_count),
        "stream_count": int(stream_count),
        "retention_fraction": (
            float(stored_count / stream_count) if stream_count else 0.0
        ),
        "unique_infosets": int(counts.size),
        "samples_per_infoset_min": float(np.min(counts)) if counts.size else 0.0,
        "samples_per_infoset_mean": float(np.mean(counts)) if counts.size else 0.0,
        "samples_per_infoset_max": float(np.max(counts)) if counts.size else 0.0,
        "samples_per_infoset_cv": (
            float(np.std(counts) / np.mean(counts))
            if counts.size and np.mean(counts) > 0.0
            else 0.0
        ),
    }


class ReservoirBuffer(object):
    """Uniform reservoir sampler over a stream of data."""

    def __init__(self, reservoir_buffer_capacity: int):
        self._reservoir_buffer_capacity = int(reservoir_buffer_capacity)
        self._data: List[Any] = []
        self._keys: List[Any] = []
        self._add_calls = 0

    def add(self, element: Any, key=None, weight=1.0) -> None:
        """Potentially add ``element`` using standard reservoir sampling."""
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
            self._keys.append(key)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
                self._keys[idx] = key
        self._add_calls += 1

    def sample(self, num_samples: int) -> list[Any]:
        """Return ``num_samples`` uniformly sampled elements from the buffer."""
        if len(self._data) < num_samples:
            raise ValueError(f"{num_samples} elements could not be sampled from size {len(self._data)}")
        return random.sample(self._data, num_samples)

    def clear(self) -> None:
        self._data = []
        self._keys = []
        self._add_calls = 0

    def resize(self, capacity: int) -> None:
        """Resize the reservoir, retaining a uniform subset when shrinking."""
        capacity = max(int(capacity), 0)
        if len(self._data) > capacity:
            selected = random.sample(range(len(self._data)), capacity)
            self._data = [self._data[index] for index in selected]
            self._keys = [self._keys[index] for index in selected]
        self._reservoir_buffer_capacity = capacity

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterable[Any]:
        return iter(self._data)

    @property
    def data(self) -> list[Any]:
        return self._data

    def get_data(self) -> list[Any]:
        return self._data

    def shuffle_data(self) -> None:
        paired = list(zip(self._data, self._keys))
        random.shuffle(paired)
        if paired:
            self._data, self._keys = map(list, zip(*paired))

    def get_num_calls(self) -> int:
        return self._add_calls

    def diagnostics(self) -> Dict[str, float]:
        return _distribution_diagnostics(
            self._keys,
            len(self._data),
            self._add_calls,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "data": list(self._data),
            "keys": list(self._keys),
            "add_calls": int(self._add_calls),
            "capacity": int(self._reservoir_buffer_capacity),
        }

    def load_state_dict(self, state) -> None:
        self._data = list(state.get("data", []))
        self._keys = list(state.get("keys", [None] * len(self._data)))
        if len(self._keys) != len(self._data):
            self._keys = [None] * len(self._data)
        self._add_calls = int(state.get("add_calls", len(self._data)))
        self._reservoir_buffer_capacity = int(
            state.get("capacity", self._reservoir_buffer_capacity)
        )


class AllSamplesBuffer(ReservoirBuffer):
    """Append-only replay used to remove finite-capacity approximation."""

    def __init__(self):
        super().__init__(0)

    def add(self, element: Any, key=None, weight=1.0) -> None:
        self._data.append(element)
        self._keys.append(key)
        self._add_calls += 1

    def resize(self, capacity: int) -> None:
        del capacity


class InfosetStratifiedBuffer(object):
    """Equal-capacity per-infoset reservoirs within one global capacity."""

    def __init__(self, capacity: int):
        self._capacity = int(capacity)
        self._data_by_key = {}
        self._seen_by_key = Counter()
        self._key_order = []
        self._add_calls = 0

    def _quotas(self):
        count = len(self._key_order)
        if count == 0:
            return {}
        base, remainder = divmod(self._capacity, count)
        return {
            key: base + (1 if index < remainder else 0)
            for index, key in enumerate(self._key_order)
        }

    def _rebalance(self):
        for key, quota in self._quotas().items():
            data = self._data_by_key[key]
            if len(data) > quota:
                self._data_by_key[key] = random.sample(data, quota)

    def add(self, element: Any, key=None, weight=1.0) -> None:
        del weight
        if key not in self._data_by_key:
            self._data_by_key[key] = []
            self._key_order.append(key)
            self._rebalance()
        quota = self._quotas()[key]
        seen = int(self._seen_by_key[key])
        data = self._data_by_key[key]
        if len(data) < quota:
            data.append(element)
        elif quota > 0:
            index = np.random.randint(0, seen + 1)
            if index < quota:
                data[index] = element
        self._seen_by_key[key] += 1
        self._add_calls += 1

    def get_data(self):
        return [
            element
            for key in self._key_order
            for element in self._data_by_key[key]
        ]

    def get_num_calls(self):
        return self._add_calls

    def clear(self):
        self._data_by_key = {}
        self._seen_by_key = Counter()
        self._key_order = []
        self._add_calls = 0

    def __len__(self):
        return sum(len(data) for data in self._data_by_key.values())

    def diagnostics(self):
        keys = [
            key
            for key in self._key_order
            for _ in self._data_by_key[key]
        ]
        return _distribution_diagnostics(keys, len(self), self._add_calls)

    def state_dict(self):
        return {
            "data_by_key": {
                key: list(data) for key, data in self._data_by_key.items()
            },
            "seen_by_key": dict(self._seen_by_key),
            "key_order": list(self._key_order),
            "add_calls": self._add_calls,
            "capacity": self._capacity,
        }

    def load_state_dict(self, state):
        self._data_by_key = {
            key: list(data)
            for key, data in state.get("data_by_key", {}).items()
        }
        self._seen_by_key = Counter(state.get("seen_by_key", {}))
        self._key_order = list(state.get("key_order", self._data_by_key))
        self._add_calls = int(state.get("add_calls", sum(self._seen_by_key.values())))
        self._capacity = int(state.get("capacity", self._capacity))


class RareHistoryQuotaBuffer(object):
    """Protect a minimum per-infoset reservoir and reservoir-sample overflow."""

    def __init__(self, capacity: int, quota_per_infoset: int):
        self._capacity = int(capacity)
        self._configured_quota = int(quota_per_infoset)
        self._protected = {}
        self._seen_by_key = Counter()
        self._key_order = []
        self._overflow = ReservoirBuffer(self._capacity)
        self._add_calls = 0

    def _effective_quota(self):
        if not self._key_order:
            return self._configured_quota
        return min(
            self._configured_quota,
            self._capacity // len(self._key_order),
        )

    def _rebalance(self):
        quota = self._effective_quota()
        displaced = []
        for key in self._key_order:
            data = self._protected[key]
            if len(data) > quota:
                kept_indices = set(random.sample(range(len(data)), quota))
                displaced.extend(
                    (element, key)
                    for index, element in enumerate(data)
                    if index not in kept_indices
                )
                self._protected[key] = [
                    element
                    for index, element in enumerate(data)
                    if index in kept_indices
                ]
        overflow_capacity = max(
            self._capacity - quota * len(self._key_order),
            0,
        )
        self._overflow.resize(overflow_capacity)
        for element, key in displaced:
            self._overflow.add(element, key=key)

    def add(self, element: Any, key=None, weight=1.0) -> None:
        del weight
        if key not in self._protected:
            self._protected[key] = []
            self._key_order.append(key)
            self._rebalance()
        quota = self._effective_quota()
        data = self._protected[key]
        seen = int(self._seen_by_key[key])
        if len(data) < quota:
            data.append(element)
        elif quota > 0:
            index = np.random.randint(0, seen + 1)
            if index < quota:
                displaced = data[index]
                data[index] = element
                self._overflow.add(displaced, key=key)
            else:
                self._overflow.add(element, key=key)
        else:
            self._overflow.add(element, key=key)
        self._seen_by_key[key] += 1
        self._add_calls += 1

    def get_data(self):
        protected = [
            element
            for key in self._key_order
            for element in self._protected[key]
        ]
        return protected + self._overflow.get_data()

    def get_num_calls(self):
        return self._add_calls

    def clear(self):
        self._protected = {}
        self._seen_by_key = Counter()
        self._key_order = []
        self._overflow.clear()
        self._add_calls = 0

    def __len__(self):
        return sum(len(data) for data in self._protected.values()) + len(
            self._overflow
        )

    def diagnostics(self):
        keys = [
            key
            for key in self._key_order
            for _ in self._protected[key]
        ] + list(self._overflow._keys)
        result = _distribution_diagnostics(keys, len(self), self._add_calls)
        result["rare_history_quota"] = int(self._effective_quota())
        return result

    def state_dict(self):
        return {
            "protected": {
                key: list(data) for key, data in self._protected.items()
            },
            "seen_by_key": dict(self._seen_by_key),
            "key_order": list(self._key_order),
            "overflow": self._overflow.state_dict(),
            "add_calls": self._add_calls,
            "capacity": self._capacity,
            "configured_quota": self._configured_quota,
        }

    def load_state_dict(self, state):
        self._protected = {
            key: list(data)
            for key, data in state.get("protected", {}).items()
        }
        self._seen_by_key = Counter(state.get("seen_by_key", {}))
        self._key_order = list(state.get("key_order", self._protected))
        self._overflow.load_state_dict(state.get("overflow", {}))
        self._add_calls = int(state.get("add_calls", sum(self._seen_by_key.values())))
        self._capacity = int(state.get("capacity", self._capacity))
        self._configured_quota = int(
            state.get("configured_quota", self._configured_quota)
        )


class WeightedReservoirBuffer(object):
    """Priority reservoir with inclusion biased by a positive stream weight."""

    def __init__(self, capacity: int, weight_floor: float = 1e-6):
        self._capacity = int(capacity)
        self._weight_floor = float(weight_floor)
        self._heap = []
        self._add_calls = 0

    def add(self, element: Any, key=None, weight=1.0) -> None:
        weight = float(weight)
        if not np.isfinite(weight) or weight <= 0.0:
            weight = self._weight_floor
        else:
            weight = max(weight, self._weight_floor)
        priority = float(np.log(max(random.random(), 1e-15)) / weight)
        item = (priority, self._add_calls, element, key, weight)
        if len(self._heap) < self._capacity:
            heapq.heappush(self._heap, item)
        elif item[0] > self._heap[0][0]:
            heapq.heapreplace(self._heap, item)
        self._add_calls += 1

    def get_data(self):
        return [item[2] for item in self._heap]

    def get_num_calls(self):
        return self._add_calls

    def clear(self):
        self._heap = []
        self._add_calls = 0

    def __len__(self):
        return len(self._heap)

    def diagnostics(self):
        result = _distribution_diagnostics(
            [item[3] for item in self._heap],
            len(self._heap),
            self._add_calls,
        )
        weights = np.asarray([item[4] for item in self._heap], dtype=np.float64)
        result["stored_weight_mean"] = (
            float(np.mean(weights)) if weights.size else 0.0
        )
        result["stored_weight_min"] = (
            float(np.min(weights)) if weights.size else 0.0
        )
        result["stored_weight_max"] = (
            float(np.max(weights)) if weights.size else 0.0
        )
        return result

    def state_dict(self):
        return {
            "heap": list(self._heap),
            "add_calls": self._add_calls,
            "capacity": self._capacity,
            "weight_floor": self._weight_floor,
        }

    def load_state_dict(self, state):
        self._heap = list(state.get("heap", []))
        heapq.heapify(self._heap)
        self._add_calls = int(state.get("add_calls", len(self._heap)))
        self._capacity = int(state.get("capacity", self._capacity))
        self._weight_floor = float(state.get("weight_floor", self._weight_floor))


def make_regret_replay_buffer(
    mode: str,
    capacity: int,
    *,
    rare_history_quota: int = 64,
    weight_floor: float = 1e-6,
):
    """Construct one regret replay backend from a configuration mode."""
    mode = str(mode).lower()
    if mode not in VALID_REGRET_REPLAY_MODES:
        raise ValueError(
            "mode must be one of "
            f"{sorted(VALID_REGRET_REPLAY_MODES)}, got {mode!r}."
        )
    if int(capacity) <= 0:
        raise ValueError("capacity must be positive.")
    if int(rare_history_quota) <= 0:
        raise ValueError("rare_history_quota must be positive.")
    if not np.isfinite(weight_floor) or float(weight_floor) <= 0.0:
        raise ValueError("weight_floor must be positive and finite.")
    if mode == RESERVOIR:
        return ReservoirBuffer(capacity)
    if mode == ALL_SAMPLES:
        return AllSamplesBuffer()
    if mode == INFOSET_STRATIFIED:
        return InfosetStratifiedBuffer(capacity)
    if mode == RARE_HISTORY_QUOTA:
        return RareHistoryQuotaBuffer(capacity, rare_history_quota)
    return WeightedReservoirBuffer(capacity, weight_floor)
