"""Tests for repository-wide chart-title conventions."""

from __future__ import annotations

from pathlib import Path

from escher_poker.chart_titles import (
    chart_title_prefix,
    poker_variant_label,
    prefixed_chart_title,
)


def test_chart_title_prefix_uses_algorithm_and_poker_variant() -> None:
    assert chart_title_prefix({"game_name": "leduc_poker"}) == "ESCHER - Leduc - "
    assert chart_title_prefix({"game_name": "kuhn_poker"}) == "ESCHER - Kuhn - "
    assert (
        chart_title_prefix({"algorithm_variant": "DREAM", "game_name": "leduc_poker"})
        == "DREAM - Leduc - "
    )


def test_prefixed_chart_title_normalises_legacy_titles() -> None:
    assert (
        prefixed_chart_title(
            "Leduc Poker ESCHER: Exploitability Across Seeds",
            {"game_name": "leduc_poker"},
        )
        == "ESCHER - Leduc - Exploitability Across Seeds"
    )
    assert (
        prefixed_chart_title(
            "ESCHER baseline: intermediate exploitability curve",
            {"game_name": "leduc_poker"},
        )
        == "ESCHER - Leduc - baseline: intermediate exploitability curve"
    )
    assert (
        prefixed_chart_title(
            "ESCHER - Kuhn - Exploitability Across Seeds",
            {"game_name": "kuhn_poker"},
        )
        == "ESCHER - Kuhn - Exploitability Across Seeds"
    )


def test_poker_variant_label_infers_from_experiment_name() -> None:
    assert poker_variant_label({"experiment_name": "kuhn_poker_escher_test"}) == "Kuhn"


def test_plotting_code_uses_shared_title_helper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    allowed = {repo_root / "escher_poker" / "chart_titles.py"}
    offenders = []
    for package_dir in (repo_root / "escher_poker", repo_root / "experiments"):
        for path in package_dir.rglob("*.py"):
            if path in allowed:
                continue
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in ("ax.set_title(", "plt.title(", ".suptitle(")):
                offenders.append(str(path.relative_to(repo_root)))

    assert offenders == []
