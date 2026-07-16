"""Shared chart-title convention for ESCHER experiment outputs."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


DEFAULT_ALGORITHM_VARIANT = "ESCHER"
DEFAULT_POKER_VARIANT = "Leduc"

_KNOWN_POKER_VARIANTS = {
    "leduc": "Leduc",
    "leduc_poker": "Leduc",
    "kuhn": "Kuhn",
    "kuhn_poker": "Kuhn",
}


def algorithm_variant_label(
    config: Mapping[str, Any] | None = None,
    *,
    algorithm: str | None = None,
) -> str:
    """Return the algorithm label used at the front of chart titles."""
    if algorithm:
        return str(algorithm).strip()
    if config:
        for key in ("algorithm_variant", "algorithm_name", "algorithm"):
            value = config.get(key)
            if value:
                return str(value).strip()
    return DEFAULT_ALGORITHM_VARIANT


def poker_variant_label(
    config: Mapping[str, Any] | None = None,
    *,
    game_name: str | None = None,
) -> str:
    """Return a compact poker-game label, such as ``Leduc`` or ``Kuhn``."""
    raw_value = game_name
    if raw_value is None and config:
        for key in ("poker_variant", "game_variant", "game_name", "game"):
            value = config.get(key)
            if value:
                raw_value = str(value)
                break
        if raw_value is None:
            raw_value = _game_name_from_experiment(config.get("experiment_name"))
    if not raw_value:
        return DEFAULT_POKER_VARIANT

    normalised = str(raw_value).strip()
    lowered = normalised.lower()
    if lowered in _KNOWN_POKER_VARIANTS:
        return _KNOWN_POKER_VARIANTS[lowered]

    lowered = re.sub(r"[_\-\s]+poker$", "", lowered)
    lowered = re.sub(r"^poker[_\-\s]+", "", lowered)
    if lowered in _KNOWN_POKER_VARIANTS:
        return _KNOWN_POKER_VARIANTS[lowered]

    words = re.split(r"[_\-\s]+", lowered)
    return " ".join(word.capitalize() for word in words if word) or DEFAULT_POKER_VARIANT


def chart_title_prefix(
    config: Mapping[str, Any] | None = None,
    *,
    algorithm: str | None = None,
    game_name: str | None = None,
) -> str:
    """Return the mandatory title prefix for experiment figures."""
    return (
        f"{algorithm_variant_label(config, algorithm=algorithm)} - "
        f"{poker_variant_label(config, game_name=game_name)} - "
    )


def prefixed_chart_title(
    title: str,
    config: Mapping[str, Any] | None = None,
    *,
    algorithm: str | None = None,
    game_name: str | None = None,
) -> str:
    """Normalise an experiment chart title to ``Algorithm - Poker - Title``."""
    prefix = chart_title_prefix(config, algorithm=algorithm, game_name=game_name)
    base_title = str(title).strip()
    if base_title.startswith(prefix):
        return base_title
    return f"{prefix}{_strip_legacy_prefix(base_title)}"


def set_chart_title(
    ax: Any,
    title: str,
    config: Mapping[str, Any] | None = None,
    *,
    algorithm: str | None = None,
    game_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """Set a Matplotlib axis title using the repository-wide convention."""
    return ax.set_title(
        prefixed_chart_title(
            title,
            config,
            algorithm=algorithm,
            game_name=game_name,
        ),
        **kwargs,
    )


def _game_name_from_experiment(experiment_name: Any) -> str | None:
    if not experiment_name:
        return None
    value = str(experiment_name).lower()
    for game_name in ("leduc_poker", "kuhn_poker"):
        if game_name in value:
            return game_name
    return None


def _strip_legacy_prefix(title: str) -> str:
    cleaned = title.strip()
    cleaned = re.sub(
        r"^(?:leduc|kuhn)\s+poker\s+escher\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^quick\s+escher\s+", "Quick ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^escher\s*[-:]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^escher\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()
