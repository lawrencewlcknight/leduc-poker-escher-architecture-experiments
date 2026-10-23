"""Complete paired-seed budget analysis; no selection or missing-run pooling."""
import csv
import json
from pathlib import Path
import numpy as np
from .config import ARMS, tasks, schedule, configuration, contract, NONINFERIORITY_MARGIN
from .worker import verify_files, write_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_csv
from experiments.leduc_poker.critic_averaging_mechanism.analyse import effect


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def aggregate(root, smoke=False):
    root=Path(root); plan=schedule(smoke)
    curves=[]; mechanisms=[]; costs=[]; manifests=[]; endpoints={}; deployed={}
    for index,(arm,seed) in enumerate(tasks(smoke)):
        directory=root/"workers"/f"task_{index:03d}_{arm}_{seed}"
        receipt=json.loads((directory/"SUCCESS.json").read_text())
        if receipt["status"]!="complete":
            raise ValueError("Incomplete receipt")
        verify_files(directory,receipt)
        manifest=json.loads((directory/"manifest.json").read_text()); manifests.append(manifest)
        result=json.loads((directory/"result.json").read_text())
        config=json.loads(json.dumps(configuration(arm,smoke)))
        if (manifest["arm"],manifest["seed"])!=(arm,seed) or manifest["config"]!=config or manifest["contract"]!=json.loads(json.dumps(contract(smoke))):
            raise ValueError("Worker differs from frozen contract")
        if (result["arm"],result["seed"],result["smoke"],result["status"],result["iteration"])!=(arm,seed,smoke,"complete",plan["end"]):
            raise ValueError("Incomplete or mismatched worker")
        required={"manifest.json","result.json","metrics.csv","diagnostics.csv","critic_fits.csv"}
        required.update(f"policy_{i:04d}.pkl" for i in plan["snapshots"])
        required.update(f"diagnostic_{i:04d}.csv" for i in plan["diagnose"])
        if not required.issubset(receipt["files"]):
            raise ValueError("Required artifact absent from receipt")
        if result["trajectories"]!=2*config["num_traversals"]*plan["end"] or result["policy_fit_count"]!=plan["end"]:
            raise ValueError("Training or policy-fitting budget mismatch")
        rows=read_csv(directory/"metrics.csv")
        if len(rows)!=len(plan["evaluate"]) or {int(r["iteration"]) for r in rows}!=set(plan["evaluate"]):
            raise ValueError("Incomplete or duplicate evaluation schedule")
        for row in rows:
            if (row["arm"],int(row["seed"]))!=(arm,seed) or int(row["policy_fit_count"])!=int(row["iteration"]):
                raise ValueError("Mismatched metric or policy fit")
            if not np.isfinite([float(row[k]) for k in ("exact_average_exploitability","deployed_exploitability","learning_seconds")]).all():
                raise ValueError("Nonfinite metric")
        late=[r for r in rows if int(r["iteration"]) in plan["late"]]
        endpoints[seed,arm]=float(np.mean([float(r["exact_average_exploitability"]) for r in late]))
        deployed[seed,arm]=float(np.mean([float(r["deployed_exploitability"]) for r in late]))
        curves.extend(rows)
        diagnostic_rows=read_csv(directory/"diagnostics.csv")
        expected={(i,f,c) for i in plan["diagnose"] for f in range(2) for c in ("deployed","zero_same_proposal")}
        if len(diagnostic_rows)!=len(expected) or {(int(r["iteration"]),int(r["fold"]),r["control"]) for r in diagnostic_rows}!=expected:
            raise ValueError("Incomplete diagnostic schedule")
        for row in diagnostic_rows:
            numbers=[float(row[k]) for k in ("max_absolute_advantage_bias","max_absolute_value_bias","advantage_variance")]
            if not np.isfinite(numbers).all() or max(numbers[:2])>1e-9:
                raise ValueError("Estimator audit failed")
        mechanisms.extend(diagnostic_rows)
        fits=read_csv(directory/"critic_fits.csv")
        if len(fits)!=2*plan["end"] or {(int(r["iteration"]),int(r["fold"])) for r in fits}!={(i,f) for i in range(1,plan["end"]+1) for f in range(2)}:
            raise ValueError("Incomplete critic-fit ledger")
        updates=[0,0]
        for row in fits:
            configured=int(row["configured_updates"])
            expected_updates=configured if int(row["replay_rows"])>=config["baseline_batch_size"] else 0
            if configured!=config["baseline_network_train_steps"] or int(row["actual_updates"])!=expected_updates:
                raise ValueError("Actual critic steps differ from contract")
            fold=int(row["fold"]); updates[fold]+=int(row["actual_updates"])
            if int(row["cumulative_updates"])!=updates[fold]:
                raise ValueError("Corrupt cumulative update count")
            if arm=="D" and any(float(row[k])!=0 for k in ("actual_updates","target_version","history_size","cache_bytes")):
                raise ValueError("Zero control performed critic work")
        if updates!=result["critic_updates"] or result["configured_critic_updates"]!=config["baseline_network_train_steps"]:
            raise ValueError("Worker update totals disagree")
        costs.append({"arm":arm,"seed":seed,"learning_seconds":result["learning_seconds"],
            "diagnostic_seconds":result["diagnostic_seconds"],"persistence_seconds":result["persistence_seconds"],
            "scheduled_evaluation_seconds":sum(float(r["evaluation_seconds"]) for r in rows),
            "peak_rss_mib":result["peak_rss_mib"],"nodes":result["nodes"],
            "critic_updates":sum(updates),**result["component_seconds"]})
    if len({m["source"] for m in manifests})!=1 or len({json.dumps(m["runtime"],sort_keys=True) for m in manifests})!=1:
        raise ValueError("Mixed source/runtime identities")
    seeds=sorted({s for _,s in tasks(smoke)})
    blocks=[]; contrasts={}; neural_contrasts={}
    for seed in seeds:
        row={"seed":seed,**{a:endpoints[seed,a] for a in ARMS}}
        row.update({f"{a}_minus_A":endpoints[seed,a]-endpoints[seed,"A"] for a in "BCD"})
        blocks.append(row)
    for arm in "BCD":
        name=f"{arm}_minus_A"
        contrasts[name]=effect([r[name] for r in blocks],alpha=.05/3)
        high=contrasts[name]["ci_high"]
        contrasts[name]["upper_interval_below_loss_margin"]=high is not None and high<NONINFERIORITY_MARGIN
        neural_contrasts[name]=effect([deployed[s,arm]-deployed[s,"A"] for s in seeds],alpha=.05/3)
    summary={"status":"complete","smoke":smoke,"seeds":seeds,"runs":len(tasks(smoke)),
        "primary_exact_average":contrasts,"secondary_deployed_policy":neural_contrasts,
        "interval":"98.333% paired t; exploratory with three seeds; no equivalence claim",
        "noninferiority_margin":NONINFERIORITY_MARGIN,
        "scope":"native adaptive proposal/calibration; fixed iterations and trajectories",
        "zero_arm":"component removal, not a frozen random critic",
        "timing":"component timers overlap; learning_seconds includes policy fitting and exact-average accumulation"}
    output=root/"analysis"; output.mkdir(exist_ok=True)
    write_json(output/"summary.json",summary)
    for name,rows in (("seed_blocks",blocks),("curves",curves),("mechanism_diagnostics",mechanisms),("costs_by_seed",costs)):
        write_csv(output/f"{name}.csv",rows)
    (output/"report.md").write_text("# Critic update budgets\n\n"+
        ("SMOKE ONLY: reduced budgets and one development seed.\n\n" if smoke else "Exploratory three-seed study. Negative differences favour the smaller budget.\n\n")+
        "Primary: mean exact-average exploitability at the fixed late checkpoints.\n\n"+
        "| Contrast | Mean | 98.333% paired interval |\n|---|---:|---|\n"+
        "".join(f"| {k} | {v['mean']:.6g} | [{v['ci_low']}, {v['ci_high']}] |\n" for k,v in contrasts.items())+
        "\nDeployed-policy contrasts, complete curves, native-proposal exact audits and costs are retained separately. "
        "A non-significant difference does not establish equivalence. The zero arm retains residual calibration. "
        "These are matched-iteration results; equal-time or FHP conclusions require subsequent experiments.\n")
    return summary
