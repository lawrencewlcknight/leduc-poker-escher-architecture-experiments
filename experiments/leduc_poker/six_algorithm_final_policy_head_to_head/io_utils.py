"""Small deterministic output helpers for Experiment 17."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from escher_poker.json_utils import json_safe


def write_json(path: str | Path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
    return path


def write_csv(path: str | Path, rows: Sequence[Mapping]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    return path


__all__ = ["write_csv", "write_json"]
