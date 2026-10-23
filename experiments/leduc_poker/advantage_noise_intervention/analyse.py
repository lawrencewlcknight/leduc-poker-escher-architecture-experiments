"""Paired effects without treating histories, players or repetitions as seeds."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from experiments.leduc_poker.critic_averaging_mechanism.analyse import effect
from experiments.leduc_poker.critic_update_budget.worker import verify_files
from experiments.critic_target_cache_benchmark import write_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_csv
from .config import tasks, contract, GAMMAS
from .worker import worker_name

OUTCOMES = ("exact_regret_mse", "policy_tv", "sensitive_policy_tv", "other_policy_tv", "sensitivity_tv_gap",
            "hybrid_exploitability", "joint_exploitability")


def aggregate(root, smoke=False):
    root = Path(root); plan = json.loads(json.dumps(contract(smoke)))
    manifests = []; raw = []; effects = []; timings = []
    for seed, checkpoint in tasks(smoke):
        directory = root/"workers"/worker_name(seed, checkpoint)
        receipt = json.loads((directory/"SUCCESS.json").read_text())
        if receipt["status"] != "complete":
            raise ValueError("Incomplete worker receipt")
        verify_files(directory, receipt)
        required = {"manifest.json", "result.json", "oracle_audit.json", "reference.csv", "metrics.csv"}
        required.update(f"rep_{r:02d}/{name}" for r in range(plan["repetitions"]) for name in (
            "SUCCESS.json", "metrics.json", "information_states.csv", "regret_models.pt", "panel_0.npz", "panel_1.npz", "timing.json"))
        if not required.issubset(receipt["files"]):
            raise ValueError("Missing required artifact")
        manifest = json.loads((directory/"manifest.json").read_text()); manifests.append(manifest)
        result = json.loads((directory/"result.json").read_text())
        if manifest["contract"] != plan or (manifest["seed"], manifest["checkpoint"]) != (seed, checkpoint):
            raise ValueError("Worker differs from frozen contract")
        if (result["seed"], result["checkpoint"], result["smoke"], result["pilot"], result["status"], result["fits"]) != (
                seed, checkpoint, smoke, False, "complete", plan["repetitions"]*6):
            raise ValueError("Incomplete or mismatched worker")
        audit = json.loads((directory/"oracle_audit.json").read_text())
        for entry in audit.values():
            if entry["max_mean_error"] > 1e-9 or entry["max_centering_error"] > 1e-9 or entry["minimum_covariance_eigenvalue"] < -1e-9:
                raise ValueError("Failed oracle audit")
        with (directory/"metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        expected = {(r, p, g) for r in range(plan["repetitions"]) for p in range(2) for g in GAMMAS}
        if len(rows) != len(expected) or {(int(r["repetition"]), int(r["player"]), float(r["gamma"])) for r in rows} != expected:
            raise ValueError("Missing or duplicated fitting results")
        by_key = {}
        for row in rows:
            if (int(row["seed"]), row["checkpoint"], int(row["optimizer_steps"])) != (seed, checkpoint, plan["fit_steps"]):
                raise ValueError("Fit identity or budget differs")
            if not np.isfinite([float(row[k]) for k in OUTCOMES]).all():
                raise ValueError("Nonfinite outcome")
            by_key[int(row["repetition"]), int(row["player"]), float(row["gamma"])] = row
        for rep in range(plan["repetitions"]):
            repdir = directory/f"rep_{rep:02d}"
            verify_files(repdir, json.loads((repdir/"SUCCESS.json").read_text()))
            canonical = json.loads((repdir/"metrics.json").read_text())
            for row in canonical:
                csv_row = by_key[row["repetition"], row["player"], row["gamma"]]
                if any(float(csv_row[k]) != row[k] for k in OUTCOMES):
                    raise ValueError("CSV and repetition outcomes disagree")
            for player in range(2):
                triplet = [by_key[rep, player, g] for g in GAMMAS]
                for field in ("minibatch_sha256", "previous_target_sha256", "minibatch_seed", "rows", "fit_iteration"):
                    if len({r[field] for r in triplet}) != 1:
                        raise ValueError(f"Unpaired {field}")
            for gamma in (0., .5):
                for outcome in OUTCOMES:
                    difference = np.mean([float(by_key[rep, p, gamma][outcome])-float(by_key[rep, p, 1.][outcome]) for p in range(2)])
                    effects.append({"seed": seed, "checkpoint": checkpoint, "repetition": rep,
                                    "gamma": gamma, "outcome": outcome, "difference": float(difference)})
        raw.extend(rows)
        timings.append({"seed": seed, "checkpoint": checkpoint, "peak_rss_mib": result["peak_rss_mib"],
                        "last_session_seconds": result["last_session_seconds"],
                        "total_fit_seconds": sum(float(r["fit_seconds"]) for r in rows)})
    if len({m["code"] for m in manifests}) != 1 or len({json.dumps(m["runtime"], sort_keys=True) for m in manifests}) != 1:
        raise ValueError("Mixed source code or runtime")
    if not smoke and len({m["source"]["source_commit"] for m in manifests}) != 1:
        raise ValueError("Mixed historical source commits")
    grouped = defaultdict(list)
    for row in effects:
        grouped[row["seed"], row["checkpoint"], row["gamma"], row["outcome"]].append(row["difference"])
    conditional = []; seed_values = defaultdict(list)
    for (seed, checkpoint, gamma, outcome), values in sorted(grouped.items()):
        conditional.append({"seed": seed, "checkpoint": checkpoint, "gamma": gamma, "outcome": outcome,
                            **effect(values), "scope": "conditional paired repetition interval"})
        seed_values[seed, gamma, outcome].append(float(np.mean(values)))
    seeds = [{"seed": seed, "gamma": gamma, "outcome": outcome, "difference": float(np.mean(values))}
             for (seed, gamma, outcome), values in sorted(seed_values.items())]
    across = []
    for gamma in (0., .5):
        for outcome in OUTCOMES:
            values = [r["difference"] for r in seeds if r["gamma"] == gamma and r["outcome"] == outcome]
            across.append({"gamma": gamma, "outcome": outcome, **effect(values),
                           "primary": gamma == 0. and outcome == "exact_regret_mse"})
    output = root/"analysis"; output.mkdir(exist_ok=True)
    summary = {"status": "complete", "smoke": smoke, "workers": len(manifests), "fits": len(raw),
        "source_seed_count": len({s for s, _ in tasks(smoke)}), "effects": across,
        "intervals": "95% paired t; one primary contrast, secondary intervals descriptive and unadjusted",
        "warning": "Repetitions are conditional on saved states; three source seeds remain n=3. "
                   "Policy fidelity is not equilibrium quality; one-step exploitability need not decrease."}
    write_json(output/"summary.json", summary)
    for name, rows in (("fits", raw), ("paired_repetitions", effects), ("conditional_effects", conditional),
                       ("seed_effects", seeds), ("timings", timings)):
        write_csv(output/f"{name}.csv", rows)
    (output/"report.md").write_text("# Advantage-noise intervention\n\n" +
        ("SMOKE ONLY. Reduced budgets and a synthetic checkpoint.\n\n" if smoke else "Oracle diagnostic on frozen Experiment 35 checkpoints.\n\n") +
        "Negative differences favour lower noise. Primary: gamma 0 minus 1 in exact-mean regret MSE.\n\n" +
        "| Gamma | Outcome | Mean difference | 95% seed interval |\n|---|---|---:|---|\n" +
        "".join(f"| {r['gamma']} | {r['outcome']} | {r['mean']:.6g} | [{r['ci_low']}, {r['ci_high']}] |\n" for r in across) +
        "\n"+summary["warning"]+" Secondary intervals are descriptive, not multiplicity-controlled discoveries.\n")
    return summary
