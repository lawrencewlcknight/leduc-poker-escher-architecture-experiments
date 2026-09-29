"""Thesis-ready plots of the saved grouped-policy trajectory."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from escher_poker.chart_titles import set_chart_title


def title(ax, text):
    set_chart_title(ax, text, algorithm="UCV-ESCHER (grouped wide)")


def plot_results(metrics, pairs, pair_summaries, hours, output):
    seeds = sorted({r["seed"] for r in metrics})
    for axis in ("actual_active_hours", "nodes_touched"):
        fig, ax = plt.subplots(figsize=(9, 5))
        scale = 1e6 if axis == "nodes_touched" else 1
        for seed in seeds:
            rows = sorted([r for r in metrics if r["seed"] == seed],
                          key=lambda r: r["checkpoint"])
            ax.plot([r[axis]/scale for r in rows], [r["exploitability"] for r in rows],
                    color="tab:green", alpha=.2, linewidth=.8)
        x, y, se = [], [], []
        for h in hours:
            rows = [r for r in metrics if r["checkpoint"] == h]
            values = np.array([r["exploitability"] for r in rows])
            x.append(np.mean([r[axis] for r in rows]) / scale)
            y.append(values.mean())
            se.append(values.std(ddof=1)/np.sqrt(len(values)) if len(values) > 1 else 0)
        ax.errorbar(x, y, yerr=se, marker="o", color="tab:green",
                    label="Mean ± one SE; faint lines are individual seeds")
        ax.axhline(0, color="black", linestyle="--", linewidth=.7)
        ax.set_xlabel("Actual active training hours" if scale == 1 else
                      "Training nodes touched (millions; mean checkpoint coordinates)")
        ax.set_ylabel("Exploitability (NashConv / 2)")
        title(ax, "Saved neural policy exploitability")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"exploitability_by_{'training_time' if scale == 1 else 'nodes'}.png", dpi=200)
        plt.close(fig)
    matrix = np.full((len(hours), len(hours)), np.nan)
    for row in pair_summaries:
        i, j = hours.index(row["later_checkpoint"]), hours.index(row["earlier_checkpoint"])
        matrix[i, j] = row["mean_ev"]
    limit = max(float(np.nanmax(np.abs(matrix))), 1e-8)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(hours)), [f"{h}h" for h in hours], rotation=45)
    ax.set_yticks(range(len(hours)), [f"{h}h" for h in hours])
    ax.set_xlabel("Earlier checkpoint (target active hours)")
    ax.set_ylabel("Later checkpoint (target active hours)")
    title(ax, "Mean exact later-versus-earlier two-seat EV")
    fig.colorbar(im, ax=ax, label="Positive EV favours the later policy")
    fig.tight_layout()
    fig.savefig(output / "head_to_head_later_vs_earlier.png", dpi=200)
    plt.close(fig)
    for kind in ("adjacent", "final_vs_earlier"):
        fig, ax = plt.subplots(figsize=(9, 5))
        records = []
        for i, hour in enumerate(hours):
            if kind == "adjacent":
                if i == 0:
                    continue
                a, b, x = hour, hours[i-1], hour
            else:
                if hour == hours[-1]:
                    continue
                a, b, x = hours[-1], hour, hour
            values = [r["A_EV_seat_averaged"] for r in pairs
                      if r["checkpoint_a"] == a and r["checkpoint_b"] == b]
            records.append((x, np.mean(values), np.std(values, ddof=1)/np.sqrt(len(values))
                            if len(values) > 1 else 0))
        x, y, se = zip(*records)
        ax.errorbar(x, y, yerr=se, marker="o", capsize=3, label="Mean ± one SE across seeds")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("Later target hour" if kind == "adjacent" else "Earlier target hour")
        ax.set_ylabel("Exact two-seat expected value")
        title(ax, "Adjacent checkpoint improvement" if kind == "adjacent" else
              f"Final {hours[-1]}h policy versus earlier policies")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / f"head_to_head_{kind}.png", dpi=200)
        plt.close(fig)
