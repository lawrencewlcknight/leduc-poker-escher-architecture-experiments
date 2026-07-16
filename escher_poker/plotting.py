"""Plotting helpers for ESCHER experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .chart_titles import set_chart_title
from .constants import (
    AVERAGE_POLICY_VALUE_TARGET_LABEL,
    LEDUC_AVERAGE_POLICY_VALUE_TARGET,
    NASH_EXPLOITABILITY_TARGET,
    NASH_EXPLOITABILITY_TARGET_LABEL,
)


def _stack_curve(results: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.vstack([np.asarray(result[key], dtype=np.float64) for result in results])


def _safe_sem(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros(matrix.shape[1])
    return stats.sem(matrix, axis=0, nan_policy="omit")


def plot_multiseed_results(
    results: List[Dict[str, Any]],
    run_dir: str | Path,
    *,
    average_policy_value_target: float = LEDUC_AVERAGE_POLICY_VALUE_TARGET,
) -> None:
    """Create thesis-style plots matching the Deep CFR repo look and feel."""
    run_dir = Path(run_dir)
    iterations = np.asarray(results[0]["iterations"], dtype=np.float64)
    exploitability_mat = _stack_curve(results, "exploitability")
    average_policy_value_mat = _stack_curve(results, "average_policy_value")
    value_error_mat = _stack_curve(results, "policy_value_error")
    nodes_mat = _stack_curve(results, "nodes_touched")

    mean_exploitability = np.mean(exploitability_mat, axis=0)
    se_exploitability = _safe_sem(exploitability_mat)
    mean_average_policy_value = np.mean(average_policy_value_mat, axis=0)
    se_average_policy_value = _safe_sem(average_policy_value_mat)
    mean_value_error = np.mean(value_error_mat, axis=0)
    se_value_error = _safe_sem(value_error_mat)
    mean_nodes = np.mean(nodes_mat, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result["iterations"], result["exploitability"], alpha=0.25, linewidth=1)
    ax.plot(iterations, mean_exploitability, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        iterations,
        mean_exploitability - se_exploitability,
        mean_exploitability + se_exploitability,
        alpha=0.2,
        label="Mean $\\pm$ s.e.",
    )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        linestyle="--",
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Leduc Poker ESCHER: Exploitability Across Seeds")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_iteration_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result["iterations"], result["average_policy_value"], alpha=0.25, linewidth=1)
    ax.plot(iterations, mean_average_policy_value, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        iterations,
        mean_average_policy_value - se_average_policy_value,
        mean_average_policy_value + se_average_policy_value,
        alpha=0.2,
        label="Mean $\\pm$ s.e.",
    )
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Average policy value")
    set_chart_title(ax, "Leduc Poker ESCHER: Average Policy Value Across Seeds")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_iteration_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result["nodes_touched"], result["exploitability"], alpha=0.25, linewidth=1)
    ax.plot(mean_nodes, mean_exploitability, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        mean_nodes,
        mean_exploitability - se_exploitability,
        mean_exploitability + se_exploitability,
        alpha=0.2,
        label="Mean $\\pm$ s.e.",
    )
    ax.axhline(
        NASH_EXPLOITABILITY_TARGET,
        linestyle="--",
        label=NASH_EXPLOITABILITY_TARGET_LABEL,
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Exploitability (NashConv/2)")
    set_chart_title(ax, "Leduc Poker ESCHER: Exploitability by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "exploitability_by_nodes_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result["nodes_touched"], result["average_policy_value"], alpha=0.25, linewidth=1)
    ax.plot(mean_nodes, mean_average_policy_value, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        mean_nodes,
        mean_average_policy_value - se_average_policy_value,
        mean_average_policy_value + se_average_policy_value,
        alpha=0.2,
        label="Mean $\\pm$ s.e.",
    )
    ax.axhline(
        average_policy_value_target,
        linestyle="--",
        label=AVERAGE_POLICY_VALUE_TARGET_LABEL,
    )
    ax.set_xlabel("Nodes touched")
    ax.set_ylabel("Average policy value")
    set_chart_title(ax, "Leduc Poker ESCHER: Average Policy Value by Nodes Touched")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "average_policy_value_by_nodes_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.plot(result["iterations"], result["policy_value_error"], alpha=0.25, linewidth=1)
    ax.plot(iterations, mean_value_error, linewidth=2, label="Mean across seeds")
    ax.fill_between(
        iterations,
        mean_value_error - se_value_error,
        mean_value_error + se_value_error,
        alpha=0.2,
        label="Mean $\\pm$ s.e.",
    )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel(r"$|v(\sigma) - v^*_{\mathrm{Leduc}}|$")
    set_chart_title(ax, "Leduc Poker ESCHER: Policy-Value Error")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "policy_value_error_multiseed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(results: List[Dict[str, Any]], run_dir: str | Path) -> None:
    """Plot loss diagnostics. These are not primary strategic metrics."""
    run_dir = Path(run_dir)
    iterations = np.asarray(results[0]["iterations"], dtype=np.float64)
    policy_loss_mat = np.vstack([result["diagnostics"]["policy_loss"].astype(float) for result in results])
    regret_loss_p0_mat = np.vstack([result["diagnostics"]["regret_loss_player_0"].astype(float) for result in results])
    regret_loss_p1_mat = np.vstack([result["diagnostics"]["regret_loss_player_1"].astype(float) for result in results])
    value_loss_mat = np.vstack([result["diagnostics"]["value_loss"].astype(float) for result in results])
    value_test_loss_mat = np.vstack([result["diagnostics"]["value_test_loss"].astype(float) for result in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, np.nanmean(policy_loss_mat, axis=0), linewidth=2, label="Policy loss")
    if len(results) > 1:
        ax.fill_between(
            iterations,
            np.nanmean(policy_loss_mat, axis=0) - stats.sem(policy_loss_mat, axis=0, nan_policy="omit"),
            np.nanmean(policy_loss_mat, axis=0) + stats.sem(policy_loss_mat, axis=0, nan_policy="omit"),
            alpha=0.2,
        )
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("MSE loss")
    set_chart_title(ax, "ESCHER Average-Policy Network Loss Diagnostic")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "policy_loss_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, np.nanmean(regret_loss_p0_mat, axis=0), linewidth=2, label="Regret loss P0")
    ax.plot(iterations, np.nanmean(regret_loss_p1_mat, axis=0), linewidth=2, label="Regret loss P1")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("MSE loss")
    set_chart_title(ax, "ESCHER Regret-Network Loss Diagnostic")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "regret_loss_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iterations, np.nanmean(value_loss_mat, axis=0), linewidth=2, label="Value train loss")
    ax.plot(iterations, np.nanmean(value_test_loss_mat, axis=0), linewidth=2, label="Value test loss")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("MSE loss")
    set_chart_title(ax, "ESCHER History-Value Network Loss Diagnostic")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "value_loss_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
