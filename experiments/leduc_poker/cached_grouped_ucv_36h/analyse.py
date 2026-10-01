"""Shared Experiment 35 diagnostics plus reproducible efficiency comparisons."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from experiments.leduc_poker.grouped_wide_policy_confirmation import analyse as base
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_csv, write_json
from .config import (ALGORITHM_ID, CANDIDATE_ID, LABEL, PRODUCTION_SEEDS, SMOKE_SEEDS,
                     checkpoint_schedule, contract, load_reference)
from .worker import verify_completed, configuration


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("Missing or non-finite analysis values")
    se = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.
    half = float(stats.t.ppf(.975, len(a)-1)) * se if len(a) > 1 else 0.
    return {"mean": float(a.mean()), "se": se, "ci95_low": float(a.mean()-half),
            "ci95_high": float(a.mean()+half), "n": len(a)}


def load_workers(root, smoke):
    expected = set(SMOKE_SEEDS if smoke else PRODUCTION_SEEDS)
    workers = {}
    for p in Path(root).glob("*/worker_result.json"):
        result = json.loads(p.read_text())
        manifest = json.loads((p.parent / "run_manifest.json").read_text())
        verify_completed(p.parent, manifest)
        seed = result["seed"]
        if (seed in workers or seed not in expected or result.get("experiment_id") != 48
                or result.get("smoke") != smoke or result.get("status") != "complete"
                or result.get("checkpoint_schedule") != list(checkpoint_schedule(smoke))
                or manifest["contract"] != json.loads(json.dumps(contract(smoke)))
                or result["config"] != json.loads(json.dumps(configuration(smoke)))
                or result.get("training_states") != []):
            raise ValueError(f"Unexpected Experiment 48 worker: {p}")
        workers[seed] = (p, result)
    if set(workers) != expected:
        raise ValueError("Missing required seeds; partial aggregates are not thesis evidence")
    if len({r["repository_commit"] for _, r in workers.values()}) != 1:
        raise ValueError("Mixed worker commits")
    return workers


def candidate_rows(metrics):
    return [{"algorithm_id": ALGORITHM_ID, "label": LABEL, "seed": int(r["seed"]),
             "checkpoint_id": r["checkpoint_id"], "hours": r["checkpoint_target_active_hours"],
             "actual_hours": float(r["actual_active_hours"]), "nodes": int(r["nodes_touched"]),
             "iteration": int(r["completed_iteration"]), "exploitability": float(r["exploitability"])}
            for r in metrics if r["policy_id"] == CANDIDATE_ID]


def late_metrics(rows, smoke):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[(r["algorithm_id"], r["seed"])].append(r)
    output = []
    for (alg, seed), records in sorted(by_seed.items()):
        records.sort(key=lambda r: r["hours"])
        late = [r for r in records if smoke or r["hours"] >= 24]
        e = np.asarray([r["exploitability"] for r in late])
        if not len(e):
            raise ValueError("Missing late window")
        output.append({"algorithm_id": alg, "seed": seed,
            "final_exploitability": records[-1]["exploitability"],
            "final_nodes_touched": records[-1]["nodes"],
            "late_window_mean_exploitability": float(e.mean()),
            "late_window_adjacent_rmssd": float(np.sqrt(np.mean(np.diff(e)**2))) if len(e)>1 else 0.})
    return output


def trajectory_plot(rows, output, by_nodes=False, diagnostics=False):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)) if by_nodes else plt.subplots(figsize=(10, 6))
    axes = np.atleast_1d(axes)
    algorithms = sorted({r["algorithm_id"] for r in rows})
    palette = {a: plt.get_cmap("tab10")(i) for i, a in enumerate(algorithms)}
    for ax_index, ax in enumerate(axes):
        for alg in algorithms:
            selected = [r for r in rows if r["algorithm_id"] == alg]
            sequences = []
            for seed in sorted({r["seed"] for r in selected}):
                part = sorted([r for r in selected if r["seed"] == seed], key=lambda r:r["hours"])
                x = np.array([r["nodes"]/1e6 if by_nodes else r["hours"] for r in part])
                y = np.array([r["exploitability"] for r in part])
                # A single completed iteration can cross several thresholds.
                unique = np.r_[np.diff(x) != 0, True]
                x, y = x[unique], y[unique]
                sequences.append((x,y))
                ax.plot(x,y,color=palette[alg],alpha=.12,lw=.6)
            if by_nodes:
                low, high = max(x[0] for x,y in sequences), min(x[-1] for x,y in sequences)
                if high < low:
                    # Individual runs are still shown, but no shared range exists.
                    continue
                grid = np.array([low]) if high == low else np.linspace(low, high, 200)
                values = np.array([np.interp(grid,x,y) for x,y in sequences])
            else:
                grid = sequences[0][0]
                if any(not np.array_equal(x,grid) for x,y in sequences):
                    raise ValueError("Time checkpoint grids differ within an algorithm")
                values = np.array([y for x,y in sequences])
            mean = values.mean(axis=0)
            se = values.std(axis=0,ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.zeros_like(mean)
            ax.plot(grid,mean,label=selected[0]["label"],color=palette[alg],lw=2)
            ax.fill_between(grid,np.maximum(mean-se,1e-8),mean+se,color=palette[alg],alpha=.12)
        ax.set_xlabel("Training nodes (millions)" if by_nodes else "Recorded active-hour checkpoint threshold")
        ax.set_ylabel("Exact exploitability (NashConv / 2)")
        ax.set_yscale("log")
        ax.grid(alpha=.2)
        if by_nodes and ax_index == 1:
            ax.set_xlim(0,15)
            ax.set_title("Common-budget view (no extrapolation)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Policy diagnostics" if diagnostics else "Cached UCV and historical references")
    fig.tight_layout(); fig.savefig(output,dpi=180); plt.close(fig)


def endpoint_rows(nodes, combined, reference, smoke):
    node_label = "smoke_node" if smoke else "node_15m"
    output = [dict(r, endpoint=node_label) for r in nodes]
    if not smoke:
        output.extend(dict(r, endpoint="node_15m") for r in reference["node_endpoints"])
    # One final observation per training seed, never all smoke timepoints as
    # pseudo-replicates of the endpoint.
    final = {}
    for r in combined:
        key = (r["algorithm_id"], r["seed"])
        if key not in final or r["hours"] > final[key]["hours"]:
            final[key] = r
    if not smoke and any(r["hours"] != 36 for r in final.values()):
        raise ValueError("A final 36-hour endpoint is missing")
    output.extend(dict(r, endpoint="smoke_final" if smoke else "time_36h")
                  for r in final.values())
    return output


def aggregate(root, smoke=False):
    root = Path(root)
    workers = load_workers(root / "workers", smoke)
    # Reuse the original exact reload checks and same-reservoir diagnostic schema.
    metrics, inventory, manifests = base._metrics(workers, smoke=smoke)
    summaries = base._summaries(metrics)
    effects, paired = base._paired(metrics)
    candidate = candidate_rows(metrics)
    times = [r for r in candidate if r["hours"] is not None]
    nodes = [r for r in candidate if r["hours"] is None]
    reference = load_reference()
    historical = [] if smoke else reference["trajectories"]
    combined = times + historical
    # Preserve the longitudinal CSV names/columns consumed by prior comparisons.
    for r in combined:
        r.update(algorithm_label=r["label"], checkpoint_type="active_time",
            checkpoint_target_active_hours=r["hours"], actual_active_hours=r["actual_hours"],
            nodes_touched=r["nodes"], completed_iteration=r["iteration"],
            seed_cohort="experiment_35_labels" if r["algorithm_id"] in (ALGORITHM_ID,CANDIDATE_ID) else "independent_historical",
            comparison_role="cached_followup" if r["algorithm_id"]==ALGORITHM_ID else "historical_reference")
    late = late_metrics(combined, smoke)
    checkpoints = []
    for alg, cp in sorted({(r["algorithm_id"],r["checkpoint_id"]) for r in combined}):
        part = [r for r in combined if r["algorithm_id"] == alg and r["checkpoint_id"] == cp]
        d=describe([r["exploitability"] for r in part])
        checkpoints.append({"algorithm_id":alg,"algorithm_label":part[0]["label"],
            "checkpoint_id":cp,"hours":part[0]["hours"],**d,
            "checkpoint_target_active_hours":part[0]["hours"],
            "mean_actual_active_hours":float(np.mean([r["actual_hours"] for r in part])),
            "mean_exploitability":d["mean"],"standard_error_exploitability":d["se"],
            "standard_deviation_exploitability":d["se"]*np.sqrt(d["n"]),
            "ci95_lower_exploitability":d["ci95_low"],"ci95_upper_exploitability":d["ci95_high"],
            "n_seeds":d["n"],"mean_nodes":float(np.mean([r["nodes"] for r in part])),
            "mean_nodes_touched":float(np.mean([r["nodes"] for r in part]))})
    comparisons = []
    for comparator in sorted({r["algorithm_id"] for r in late}-{ALGORITHM_ID}):
        a = {r["seed"]:r for r in late if r["algorithm_id"] == ALGORITHM_ID}
        b = {r["seed"]:r for r in late if r["algorithm_id"] == comparator}
        for field in ("final_exploitability", "late_window_mean_exploitability", "late_window_adjacent_rmssd", "final_nodes_touched"):
            if comparator == CANDIDATE_ID:
                if a.keys() != b.keys():
                    raise ValueError("Experiment 35 seed pairing is incomplete")
                result = describe([a[s][field]-b[s][field] for s in a])
                design = "same-seed historical difference, not contemporaneous randomisation"
            else:
                result = base._welch_comparison([r[field] for r in a.values()],[r[field] for r in b.values()])
                design = "independent historical cohorts, exploratory Welch interval"
            comparisons.append({"comparator":comparator,"metric":field,"design":design,**result})
    workload, component_rows = [], []
    for seed, (p, result) in workers.items():
        for criterion in ("iteration","nodes"):
            entry = result["matched_workload"].get(criterion)
            workload.append({"seed":seed,"criterion":criterion,"reached":entry is not None,
                **({k:v for k,v in entry.items() if k!="reference"} if entry else {}),
                "reference":entry["reference"] if entry else reference["milestones"].get(str(seed))})
        component_rows.extend(base._read_csv(p.parent / "component_timings.csv"))
    workload_summary = []
    for criterion in ("iteration", "nodes"):
        reached = [r for r in workload if r["criterion"] == criterion and r["reached"]]
        # Do not average only the faster subset when a seed fails to reach it.
        complete = len(reached) == len(workers)
        for field in ("active_hours", "hours_saved"):
            workload_summary.append({"criterion": criterion, "metric": field,
                "n_reached": len(reached), "n_required": len(workers),
                "all_seeds_reached": complete,
                **(describe([r[field] for r in reached]) if complete else {})})
    output = root / "analysis"; output.mkdir(exist_ok=True)
    outputs = {"worker_manifest":manifests, "snapshot_inventory":inventory,
        "checkpoint_policy_metrics":metrics,"checkpoint_policy_summary":summaries,
        "paired_policy_effects":effects,"paired_policy_summary":paired,
        "late_window_metrics_by_seed":base._late_window(metrics,smoke=smoke),
        "all_algorithm_checkpoint_policy_metrics":combined,"all_algorithm_checkpoint_summary":checkpoints,
        "all_algorithm_late_window_by_seed":late,"historical_comparisons":comparisons,
        "component_timings":component_rows,
        "matched_workload_summary":workload_summary,
        "matched_workload": [{**r,"reference":json.dumps(r["reference"],sort_keys=True)} for r in workload]}
    for name, rows in outputs.items():
        write_csv(output / f"{name}.csv", rows)
    endpoints = endpoint_rows(nodes, combined, reference, smoke)
    write_csv(output / "endpoint_seed_metrics.csv", endpoints)
    endpoint_summary = []
    for a,e in sorted({(r["algorithm_id"],r["endpoint"]) for r in endpoints}):
        part=[r for r in endpoints if r["algorithm_id"]==a and r["endpoint"]==e]
        endpoint_summary.append({"algorithm_id":a,"endpoint":e,**describe([r["exploitability"] for r in part]),
            "mean_nodes":float(np.mean([r["nodes"] for r in part])),"mean_actual_hours":float(np.mean([r["actual_hours"] for r in part]))})
    write_csv(output / "endpoint_summary.csv",endpoint_summary)
    diagnostic_rows=[{"algorithm_id":r["policy_id"],"label":r["policy_label"],"seed":r["seed"],
        "hours":r["checkpoint_target_active_hours"],"nodes":r["nodes_touched"],"exploitability":r["exploitability"]}
        for r in metrics if r["checkpoint_type"]=="active_time"]
    for by_nodes,suffix in ((False,"training_time"),(True,"nodes_touched")):
        trajectory_plot(combined,output / f"exploitability_by_{suffix}.png",by_nodes)
        trajectory_plot(diagnostic_rows,output / f"average_policy_diagnostics_by_{suffix}.png",by_nodes,True)
    # Paired gap chart uses the original data, but does not call this a fresh confirmation.
    fig,ax=plt.subplots(figsize=(9,5))
    for policy in sorted({r["comparator_id"] for r in paired}):
        part=sorted([r for r in paired if r["comparator_id"]==policy and r["checkpoint_type"]=="active_time"],key=lambda r:r["checkpoint_target_active_hours"])
        ax.plot([r["checkpoint_target_active_hours"] for r in part],[r["mean_candidate_minus_comparator"] for r in part],label=policy)
    ax.axhline(0,color="black",lw=.8);ax.legend(fontsize=8);ax.set(xlabel="Active-hour threshold",ylabel="Candidate minus diagnostic exploitability",title="Same-reservoir policy gaps (follow-up)")
    fig.tight_layout();fig.savefig(output/"paired_policy_gaps.png",dpi=180);plt.close(fig)
    manifest={"status":"complete","experiment_id":48,"smoke":smoke,"contract":contract(smoke),
        "num_workers":len(workers),"num_playable_snapshots":len(inventory),
        "matched_workload":workload,"matched_workload_summary":workload_summary,
        "endpoint_summary":endpoint_summary,
        "comparisons":comparisons,"source_provenance":reference["source_files"],
        "worker_commits":sorted({r["repository_commit"] for _,r in workers.values()}),
        "limitations":["Historical same-seed follow-up, not fresh confirmation or an isolated hardware-randomised test",
            "Intervals are exploratory and unadjusted; the training seed is the inferential unit",
            "Time curves use crossed thresholds; source clocks/output fitting costs differ",
            "Node mean curves interpolate only over each method's common observed seed range",
            "Component timers overlap where explicitly labelled; do not sum inclusive/subset timers",
            "No new head-to-head league computed; saved policies support existing exact evaluators"]}
    for name in ("aggregate_summary.json","aggregate_manifest.json","experiment_metadata.json"):
        write_json(output/name,manifest)
    write_json(output/"frozen_reference.json",reference)
    (output/"README.md").write_text("# Experiment 48 analysis\n\n"+"\n".join("- "+v for v in manifest["limitations"])+"\n\nNo full training states are retained. SUCCESS is completion, not a claim of performance superiority.\n")
    return manifest
