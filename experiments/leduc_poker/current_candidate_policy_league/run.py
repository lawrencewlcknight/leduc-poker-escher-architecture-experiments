"""Exact cross-seed, two-seat leagues from immutable saved Leduc policies."""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import importlib.metadata
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .sources import (ENDPOINTS, LABELS, artifact, build_records, sha256,
                      validate_records, write_json)

LOG = logging.getLogger(__name__)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def checked_table(game, candidate, reference):
    from open_spiel.python import policy
    from open_spiel.python.algorithms import exploitability
    from escher_poker.checkpoint_analysis import exact_seat_averaged_value_for_a
    table = policy.tabular_policy_from_callable(game, candidate.action_probabilities)
    x = table.action_probability_array
    if (not np.isfinite(x).all() or (x < 0).any()
            or not np.allclose(x.sum(axis=1), 1, atol=1e-6)
            or np.any(np.abs(x[table.legal_actions_mask == 0]) > 1e-8)):
        raise ValueError("Invalid probabilities or illegal action mass")
    x /= x.sum(axis=1, keepdims=True)
    value = float(exploitability.nash_conv(game, table) / 2)
    if not np.isfinite(reference) or abs(value - reference) > 1e-5:
        raise ValueError(f"Reloaded exploitability {value} differs from source {reference}")
    if abs(exact_seat_averaged_value_for_a(game, table, table)["A_EV_seat_averaged"]) > 1e-10:
        raise ValueError("Self-play seat reversal check failed")
    return table, value


def tabulate(records, source, output):
    import pyspiel
    from open_spiel.python import policy
    from .policies import load_neural, sd_policies
    game = pyspiel.load_game("leduc_poker")
    tables, diagnostics = {}, []
    # One archive per seed in memory; small exact policy tables persist to disk.
    for (a, seed), group in itertools.groupby(sorted(records, key=lambda r: (r["algorithm"], r["seed"])),
                                             key=lambda r: (r["algorithm"], r["seed"])):
        group = list(group)
        sd = None
        for r in group:
            path = artifact(r, source)
            r["artifact_sha256"] = sha256(path)
            cache = output / "tables" / f"{r['endpoint']}_{a}_{seed}.npz"
            identity = {"record": r, "adapter": sha256(Path(__file__).with_name("policies.py")),
                        "runner": sha256(Path(__file__))}
            sidecar = cache.with_suffix(".json")
            if cache.exists() and sidecar.exists():
                saved = json.loads(sidecar.read_text())
                if saved["identity"] != identity or saved["sha256"] != sha256(cache):
                    raise ValueError("Changed cached policy table inputs; choose a new output")
                table = policy.TabularPolicy(game)
                with np.load(cache, allow_pickle=False) as data:
                    if list(data["keys"]) != list(table.state_lookup):
                        raise ValueError("Cached information-set order differs")
                    table.action_probability_array[:] = data["probabilities"]
                candidate = table
            else:
                if a == "sd_cfr":
                    if sd is None:
                        sd = sd_policies(game, group, path)
                    candidate = sd[r["iteration"]]
                else:
                    candidate = load_neural(game, r, path)
            table, value = checked_table(game, candidate, r["source_exploitability"])
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, keys=np.asarray(list(table.state_lookup)), probabilities=table.action_probability_array)
            write_json(sidecar, {"identity": identity, "sha256": sha256(cache)})
            key = (r["endpoint"], a, seed)
            tables[key] = table
            diagnostics.append({**r, "exploitability": value,
                                "reload_absolute_error": abs(value - r["source_exploitability"]),
                                "node_budget_difference": r["nodes_touched"] - 15_000_000
                                if r["endpoint"] == "node_15m" else None})
            LOG.info("Validated %s %s seed %s: %.8f", r["endpoint"], a, seed, value)
            del candidate
        del sd
        gc.collect()
    return game, tables, diagnostics


def pair_values(game, tables, records, output):
    from escher_poker.checkpoint_analysis import exact_seat_averaged_value_for_a
    rows = []
    for ep, algorithms in ENDPOINTS.items():
        for a, b in itertools.combinations(algorithms, 2):
            group_a = sorted(r["seed"] for r in records if r["endpoint"] == ep and r["algorithm"] == a)
            group_b = sorted(r["seed"] for r in records if r["endpoint"] == ep and r["algorithm"] == b)
            for sa, sb in itertools.product(group_a, group_b):
                pa, pb = tables[(ep, a, sa)], tables[(ep, b, sb)]
                values = exact_seat_averaged_value_for_a(game, pa, pb)
                if not all(np.isfinite(v) for v in values.values()):
                    raise ValueError("Non-finite matchup")
                # Independent reversal of each algorithm-pair's first match guards sign convention.
                if sa == group_a[0] and sb == group_b[0]:
                    reverse = exact_seat_averaged_value_for_a(game, pb, pa)
                    if abs(values["A_EV_seat_averaged"] + reverse["A_EV_seat_averaged"]) > 1e-10:
                        raise ValueError("Seat/algorithm antisymmetry check failed")
                rows.append(dict(endpoint=ep, algorithm_a=a, algorithm_b=b, seed_a=sa, seed_b=sb, **values))
            # Persist useful partial analysis if a later stage fails.
            write_csv(output / "analysis" / "pairwise_exact_values.csv", rows)
            LOG.info("Completed %s: %s versus %s", ep, a, b)
    return rows


def summarise(records, pairs, *, replicates=10000):
    """Cluster bootstrap on training seeds, never on the 25 dependent matchups."""
    rng = np.random.default_rng(460029)
    summaries, strengths, policy_stats = [], [], []
    for ep, algorithms in ENDPOINTS.items():
        groups = {a: sorted([r for r in records if r["endpoint"] == ep and r["algorithm"] == a],
                            key=lambda r: r["seed"]) for a in algorithms}
        # Common seed labels from the historical cohort are jointly resampled.
        draws = {}
        for group in groups.values():
            cohort = group[0]["cohort"]
            labels = tuple(r["seed"] for r in group)
            key = (cohort, labels)
            if key not in draws:
                draws[key] = rng.integers(0, len(group), size=(replicates, len(group)))
        indices = {a: draws[(g[0]["cohort"], tuple(r["seed"] for r in g))] for a, g in groups.items()}
        scores = {a: [] for a in algorithms}
        score_boot = {a: [] for a in algorithms}
        for a, b in itertools.combinations(algorithms, 2):
            sa, sb = [r["seed"] for r in groups[a]], [r["seed"] for r in groups[b]]
            selected = [r for r in pairs if r["endpoint"] == ep and r["algorithm_a"] == a and r["algorithm_b"] == b]
            if len(selected) != len(sa) * len(sb):
                raise ValueError("Incomplete cross-seed league")
            lookup = {(r["seed_a"], r["seed_b"]): r["A_EV_seat_averaged"] for r in selected}
            if len(lookup) != len(selected):
                raise ValueError("Duplicate matchup")
            matrix = np.asarray([[lookup[(x, y)] for y in sb] for x in sa])
            boot = matrix[indices[a][:, :, None], indices[b][:, None, :]].mean(axis=(1, 2))
            lo, hi = np.quantile(boot, [.025, .975])
            summaries.append(dict(endpoint=ep, algorithm_a=a, algorithm_b=b, n_seeds_a=len(sa),
                                  n_seeds_b=len(sb), n_matchups=len(selected), mean_ev=float(matrix.mean()),
                                  bootstrap_se=float(boot.std(ddof=1)), ci95_low=float(lo), ci95_high=float(hi)))
            scores[a].append(matrix.mean()); scores[b].append(-matrix.mean())
            score_boot[a].append(boot); score_boot[b].append(-boot)
        for a in algorithms:
            boot = np.mean(score_boot[a], axis=0)
            lo, hi = np.quantile(boot, [.025, .975])
            strengths.append(dict(endpoint=ep, algorithm=a, mean_ev=float(np.mean(scores[a])),
                                  bootstrap_se=float(boot.std(ddof=1)), ci95_low=float(lo), ci95_high=float(hi)))
            values = np.array([r["exploitability"] for r in groups[a]])
            policy_stats.append(dict(endpoint=ep, algorithm=a, n_seeds=len(values),
                                     mean_exploitability=float(values.mean()),
                                     std_exploitability=float(values.std(ddof=1)) if len(values)>1 else None,
                                     sem_exploitability=float(values.std(ddof=1)/np.sqrt(len(values)))
                                     if len(values)>1 else None,
                                     mean_nodes_touched=float(np.mean([r["nodes_touched"] for r in groups[a]]))))
    return summaries, strengths, policy_stats


def plot(summaries, strengths, metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for ep, algorithms in ENDPOINTS.items():
        folder = output / ep
        folder.mkdir(parents=True, exist_ok=True)
        matrix = np.zeros((len(algorithms), len(algorithms)))
        for r in summaries:
            if r["endpoint"] == ep:
                i,j = algorithms.index(r["algorithm_a"]), algorithms.index(r["algorithm_b"])
                matrix[i,j],matrix[j,i] = r["mean_ev"], -r["mean_ev"]
        write_csv(folder/"head_to_head_mean_ev_matrix.csv",
                  [{"row_algorithm":a, **{b:float(matrix[i,j]) for j,b in enumerate(algorithms)}}
                   for i,a in enumerate(algorithms)])
        fig, ax = plt.subplots(figsize=(10,8))
        bound = max(float(np.abs(matrix).max()), 1e-6)
        im = ax.imshow(matrix, cmap="RdBu", vmin=-bound, vmax=bound)
        labels = [LABELS[a] for a in algorithms]
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for (i,j), v in np.ndenumerate(matrix):
            ax.text(j,i,"—" if i==j else f"{v:+.4f}",ha="center",va="center",
                    color="white" if abs(v)>.65*bound else "black")
        smoke_label = "SMOKE CHECK — " if all(r["n_seeds"]==1 for r in metrics if r["endpoint"]==ep) else ""
        ax.set_title(f"{smoke_label}Current candidates: {ep}\nPositive values favour row algorithm")
        fig.colorbar(im, ax=ax, label="Exact seat-averaged payoff (game utility units)")
        fig.tight_layout(); fig.savefig(folder/"head_to_head_mean_ev_heatmap.png",dpi=220); plt.close(fig)
        for name, data, field, error, ylabel in (
                ("algorithm_strength",strengths,"mean_ev","bootstrap_se","Mean payoff versus other algorithms (± bootstrap SE)"),
                ("endpoint_exploitability",metrics,"mean_exploitability","sem_exploitability","Exploitability (NashConv / 2; ± SE)")):
            rows = [next(r for r in data if r["endpoint"]==ep and r["algorithm"]==a) for a in algorithms]
            fig,ax=plt.subplots(figsize=(10,5))
            ax.bar(labels,[r[field] for r in rows],yerr=[r[error] or 0 for r in rows],capsize=4)
            ax.set_ylabel(ylabel); ax.set_title(smoke_label+ep); ax.tick_params(axis="x",rotation=30)
            fig.tight_layout(); fig.savefig(folder/f"{name}.png",dpi=220); plt.close(fig)


def run(source, output, *, smoke=False):
    import torch
    torch.set_num_threads(1)
    output = Path(output); source = Path(source)
    if (output / "SUCCESS.json").exists():
        raise FileExistsError("Completed output exists; use a new RUN_ID/output directory")
    started = time.perf_counter()
    manifest = source / ("smoke_manifest.json" if smoke else "manifest.json")
    records = json.loads(manifest.read_text())
    validate_records(records, smoke=smoke)
    game, tables, metrics = tabulate(records, source, output)
    write_csv(output/"analysis/policy_metrics.csv", metrics)
    pairs = pair_values(game, tables, records, output)
    summaries,strengths,policy_stats = summarise(metrics,pairs)
    for name,rows in (("head_to_head_summary",summaries),("algorithm_strength",strengths),("endpoint_summary",policy_stats)):
        write_csv(output/"analysis"/f"{name}.csv",rows)
    plot(summaries,strengths,policy_stats,output/"analysis")
    metadata = dict(experiment_id=46, status="complete", smoke=smoke, training_performed=False,
                    experiment_name="current_candidate_policy_league",
                    created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), command=sys.argv,
                    packages={p:importlib.metadata.version(p) for p in ("torch","numpy","open_spiel","h5py")},
                    fitting_performed=False, num_policies=len(records), num_matchups=len(pairs),
                    num_seat_assignments=2*len(pairs), elapsed_seconds=time.perf_counter()-started,
                    manifest_sha256=sha256(manifest),
                    evaluator_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    uncertainty="10000 training-seed cluster bootstrap replicates; common historical seed labels jointly resampled; pointwise exploratory 95% intervals, not multiplicity adjusted",
                    limitations=["Retrospective selected configurations, not a new frozen training benchmark",
                                 "15M Deep CFR snapshots are 14.88--14.96M; other policies cross 15M",
                                 "36h is each source's active-training clock; deferred DREAM/ESCHER fitting is additional cost",
                                 "36h league has no VR arms: comparable saved 36h policies do not exist"],
                    records=metrics)
    write_json(output/"analysis/aggregate_summary.json",metadata)
    write_json(output/"experiment_metadata.json",metadata)
    write_json(output/"SUCCESS.json",dict(status="complete",smoke=smoke,num_matchups=len(pairs)))
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","smoke","run"))
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output-root",type=Path)
    p.add_argument("--smoke-only",action="store_true")
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.action=="prepare":
        records=build_records(args.source_root,smoke=args.smoke_only)
        print(f"Prepared {len(records)} policies; no training or fitting")
    else:
        if not args.output_root: p.error("--output-root required")
        result=run(args.source_root,args.output_root,smoke=args.action=="smoke")
        print(json.dumps({k:result[k] for k in ("status","num_policies","num_matchups","elapsed_seconds")}))


if __name__=="__main__":
    main()
