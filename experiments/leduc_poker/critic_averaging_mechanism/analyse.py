"""Seed-block inference. Missing runs fail closed; no checkpoint selection."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .config import tasks, schedule, configuration, contract, MINIMUM_WORTHWHILE_EFFECT
from .worker import verify_files, write_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_csv


def effect(values, alpha=.05):
    x = np.asarray(values,dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite effect")
    n=len(x); mean=float(x.mean())
    if n<2:
        return {"n":n,"mean":mean,"ci_low":None,"ci_high":None,"se":None}
    se=float(x.std(ddof=1)/np.sqrt(n)); margin=float(t.ppf(1-alpha/2,n-1)*se)
    return {"n":n,"mean":mean,"se":se,"ci_low":mean-margin,"ci_high":mean+margin}


def aggregate(root,smoke=False):
    root=Path(root); plan=schedule(smoke); endpoints={}; curves=[]; manifests=[]; mechanism=[]; drift=[]
    for index,(arm,seed) in enumerate(tasks(smoke)):
        d=root/"workers"/f"task_{index:03d}_{arm}_{seed}"
        receipt=json.loads((d/"SUCCESS.json").read_text())
        verify_files(d,receipt)
        m=json.loads((d/"manifest.json").read_text()); manifests.append(m)
        result=json.loads((d/"result.json").read_text())
        if (m["arm"],m["seed"],m["contract"]["smoke"]) != (arm,seed,smoke):
            raise ValueError("Mismatched worker identity")
        expected_contract=json.loads(json.dumps(contract(smoke)))
        expected_config=json.loads(json.dumps(configuration(arm,smoke)))
        if m["contract"] != expected_contract or m["config"] != expected_config:
            raise ValueError("Worker configuration differs from frozen contract")
        if (result["arm"],result["seed"],result["smoke"]) != (arm,seed,smoke):
            raise ValueError("Result identity differs from manifest")
        required={"manifest.json","result.json","metrics.csv","diagnostics.csv","critic_fits.csv"}
        required.update(f"policy_{i:04d}.pkl" for i in plan["fit"])
        required.update(f"diagnostic_{i:04d}.csv" for i in plan["diagnose"])
        if not required.issubset(receipt["files"]):
            raise ValueError("Required artifact absent from receipt")
        if result["trajectories"] != 2*expected_config["num_traversals"]*plan["end"]:
            raise ValueError("Wrong trajectory budget")
        if result["status"]!="complete" or result["iteration"]!=plan["end"]:
            raise ValueError("Incomplete worker")
        rows=list(csv.DictReader((d/"metrics.csv").open()))
        expected=set(plan["evaluate"])|set(plan["fit"])
        if len(rows)!=len(expected) or {int(r["iteration"]) for r in rows}!=expected:
            raise ValueError("Incomplete or duplicate evaluation schedule")
        values=[float(r["exact_average_exploitability"]) for r in rows if int(r["iteration"]) in plan["late"]]
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite primary outcome")
        endpoints[seed,arm]=float(np.mean(values)); curves.extend(rows)
        with (d/"diagnostics.csv").open() as f:
            diagnostic_rows=list(csv.DictReader(f))
        expected_diagnostics={(i,fold,window) for i in plan["diagnose"] for fold in range(2) for window in (1,4)}
        if len(diagnostic_rows)!=len(expected_diagnostics) or {
            (int(r["iteration"]),int(r["fold"]),int(r["control_window"])) for r in diagnostic_rows} != expected_diagnostics:
            raise ValueError("Incomplete diagnostic schedule")
        for r in diagnostic_rows:
            if max(float(r["max_absolute_advantage_bias"]),float(r["max_absolute_value_bias"])) > 1e-9:
                raise ValueError("Estimator mean audit failed")
        mechanism.extend(diagnostic_rows)
        with (d/"critic_fits.csv").open() as f:
            fit_rows=list(csv.DictReader(f))
        late_rows=[r for r in fit_rows if min(plan["late"]) <= int(r["iteration"]) <= plan["end"]]
        if not late_rows:
            raise ValueError("Missing late critic fits")
        drift.append({"arm":arm,"seed":seed,"late_fit_rows":len(late_rows),
            "mean_td_label_update_rms":float(np.mean([float(r["td_label_update_rms"]) for r in late_rows])),
            "scope":"descriptive; changing replay panel"})
    if len({m["source"] for m in manifests})!=1 or len({json.dumps(m["runtime"],sort_keys=True) for m in manifests})!=1:
        raise ValueError("Workers used different source/runtime contracts")
    seeds=sorted({seed for _,seed in tasks(smoke)})
    contrasts={"D_minus_A":[],"B_minus_A":[],"C_minus_A":[],"interaction":[]}
    blocks=[]
    for seed in seeds:
        a,b,c,d=(endpoints[seed,arm] for arm in "ABCD")
        row={"seed":seed,"A":a,"B":b,"C":c,"D":d,"D_minus_A":d-a,
             "B_minus_A":b-a,"C_minus_A":c-a,"interaction":d-b-c+a}
        blocks.append(row)
        for name in contrasts:contrasts[name].append(row[name])
    output=root/"analysis"; output.mkdir(exist_ok=True)
    summary={name:dict(effect(values,.05 if name=="D_minus_A" else .05/3),
        interval="95% paired t" if name=="D_minus_A" else "98.333% paired t (three secondary contrasts)")
        for name,values in contrasts.items()}
    x=np.asarray(contrasts["D_minus_A"])
    # Exact sign-flip reference requires symmetric paired differences under H0;
    # seed pairing alone does not establish a randomized-treatment test.
    observed=abs(x.mean())
    p=sum(abs(np.mean(x*np.array(signs)))>=observed-1e-15
          for signs in itertools.product((-1,1),repeat=len(x)))/2**len(x)
    primary=summary["D_minus_A"]
    summary.update(smoke=smoke,status="complete",minimum_worthwhile_effect=MINIMUM_WORTHWHILE_EFFECT,
        primary_sign_flip_p=p if len(x)>1 else None,
        sign_flip_assumption="symmetric paired differences; sensitivity analysis, not randomized treatment assignment",
        practical_benefit_established=(primary["ci_high"] is not None and primary["ci_high"] < -MINIMUM_WORTHWHILE_EFFECT),
        inferential_unit="independent training seed block",scope="fixed critic-independent proposal")
    write_json(output/"summary.json",summary)
    write_csv(output/"seed_blocks.csv",blocks); write_csv(output/"curves.csv",curves)
    write_csv(output/"mechanism_diagnostics.csv",mechanism)
    write_csv(output/"td_drift_by_seed.csv",drift)
    (output/"report.md").write_text("# Critic averaging roles\n\n"+
        ("SMOKE ONLY: no scientific inference.\n\n" if smoke else "Negative differences favour averaging.\n\n")+
        "Primary: mean exact-average exploitability at the fixed late checkpoints.\n\n"+
        "| Contrast | Mean | Interval |\n|---|---:|---|\n"+
        "".join(f"| {k} | {summary[k]['mean']:.6g} | [{summary[k]['ci_low']}, {summary[k]['ci_high']}] |\n" for k in contrasts)+
        "\nThe secondary intervals adjust for three contrasts. Small seed counts limit precision; "
        "no production-sampler, FHP, or neural-convergence claim follows from this study.\n")
    return summary
