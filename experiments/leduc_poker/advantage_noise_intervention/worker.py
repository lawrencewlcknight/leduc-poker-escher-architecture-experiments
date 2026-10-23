"""Atomic paired-repetition outputs and strict identity-checked restart."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.leduc_poker.critic_update_budget.worker import source_identity as base_identity, runtime, verify_files
from experiments.critic_target_cache_benchmark import sha256, write_json
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_csv
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _peak_rss_mb
from experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.diagnostics import exact_exploitability
from .config import contract, tasks, GAMMAS, derived_seed
from .core import FrozenProblem, fit, evaluate_fit
from .source import load_source


def code_identity():
    digest = hashlib.sha256(base_identity().encode())
    test = Path(__file__).resolve().parents[3]/"tests/test_advantage_noise_intervention.py"
    if test.exists():
        digest.update(test.read_bytes())
    return digest.hexdigest()


def worker_name(seed, checkpoint):
    return f"seed_{seed}_{checkpoint}"


def receipt(directory, names):
    return {"status": "complete", "files": {name: sha256(directory/name) for name in sorted(names)}}


def upload(directory):
    remote = os.environ.get("EXP52_REMOTE_WORKER")
    if remote:
        if not remote.startswith("gs://") or any(c.isspace() for c in remote):
            raise ValueError("Invalid worker URI")
        subprocess.run(["gcloud", "storage", "rsync", "--recursive", "--exclude", "(^|/)SUCCESS\\.json$|.*\\.tmp$",
                        str(directory), remote], check=True)
        # A failed data upload cannot advertise a newly completed repetition.
        for marker in sorted(directory.glob("rep_*/SUCCESS.json")):
            subprocess.run(["gcloud", "storage", "cp", str(marker),
                            remote+"/"+str(marker.relative_to(directory))], check=True)
        if (directory/"SUCCESS.json").exists():
            subprocess.run(["gcloud", "storage", "cp", str(directory/"SUCCESS.json"),
                            remote+"/SUCCESS.json"], check=True)


def run_worker(root, index=0, source=None, smoke=False, pilot=False, stop_after=None):
    torch.set_num_threads(1 if smoke else 8)
    plan = contract(smoke, pilot)
    seed, checkpoint = tasks(smoke)[index]
    directory = Path(root)/"workers"/worker_name(seed, checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    solver, identity = load_source(source, seed, checkpoint, smoke)
    manifest = json.loads(json.dumps({"contract": plan, "source": identity,
        "code": code_identity(), "runtime": runtime(), "seed": seed, "checkpoint": checkpoint}))
    path = directory/"manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Source/configuration/runtime/code changed; use a new run ID")
    elif any(directory.iterdir()):
        raise ValueError("Nonempty output without manifest")
    else:
        write_json(path, manifest)
    if (directory/"SUCCESS.json").exists():
        verify_files(directory, json.loads((directory/"SUCCESS.json").read_text()))
        return json.loads((directory/"result.json").read_text())
    problem = FrozenProblem(solver)
    write_json(directory/"oracle_audit.json", problem.audit)
    references = []
    for player, rows in problem.references.items():
        for row in rows:
            for action in np.flatnonzero(row["mask"]):
                references.append({"player": player, "information_state": row["key"][1], "action": int(action),
                    "weight": row["weight"], "mean_advantage": float(row["mu"][action]),
                    "reference_regret": float(row["target"][action]), "reference_probability": float(row["policy"][action]),
                    "sensitivity": row["sensitivity"], "sensitive": row["sensitive"]})
    write_csv(directory/"reference.csv", references)
    all_metrics = []
    for repetition in range(plan["repetitions"]):
        repdir = directory/f"rep_{repetition:02d}"; repdir.mkdir(exist_ok=True)
        done = repdir/"SUCCESS.json"
        if done.exists():
            saved = json.loads(done.read_text()); verify_files(repdir, saved)
            all_metrics.extend(json.loads((repdir/"metrics.json").read_text()))
            continue
        rep_start = time.perf_counter()
        collection_start = time.perf_counter()
        panels = {p: problem.collect(p, plan["traversals_per_player"],
            derived_seed(seed, checkpoint, repetition, p, "collection")) for p in range(2)}
        collection_seconds = time.perf_counter()-collection_start
        for player, panel in panels.items():
            with (repdir/f"panel_{player}.npz.tmp").open("wb") as handle:
                np.savez_compressed(handle, **panel)
            (repdir/f"panel_{player}.npz.tmp").replace(repdir/f"panel_{player}.npz")
        metrics = []; details = []; models = {}; policies = {}
        for player in range(2):
            minibatch_seed = derived_seed(seed, checkpoint, repetition, player, "minibatches")
            hashes = set(); previous = set()
            for gamma in GAMMAS:
                trainer, timing = fit(problem, player, panels[player], gamma,
                    plan["fit_steps"], plan["batch_size"], minibatch_seed)
                evaluation, state_rows, table = evaluate_fit(problem, player, trainer)
                row = {"seed": seed, "checkpoint": checkpoint, "repetition": repetition,
                    "player": player, "gamma": gamma, "rows": len(panels[player]["sampled"]),
                    "fit_iteration": problem.iteration, "minibatch_seed": minibatch_seed,
                    "collection_seconds_both_players": collection_seconds,
                    "baseline_exploitability": problem.initial_exploitability, **timing, **evaluation}
                metrics.append(row)
                details.extend(dict(r, gamma=gamma) for r in state_rows)
                models[f"player_{player}_gamma_{gamma}"] = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
                policies[player, gamma] = table
                hashes.add(timing["minibatch_sha256"]); previous.add(timing["previous_target_sha256"])
            if len(hashes) != 1 or len(previous) != 1:
                raise RuntimeError("Unpaired minibatches or previous-regret targets")
        for gamma in GAMMAS:
            joint = {key: policies[key[0], gamma][key] for key in problem.table}
            exploitability = exact_exploitability(solver.game, joint)
            for row in metrics:
                if row["gamma"] == gamma:
                    row["joint_exploitability"] = exploitability
        problem.assert_frozen()
        write_json(repdir/"metrics.json", metrics)
        write_csv(repdir/"information_states.csv", details)
        torch.save(models, repdir/"regret_models.pt.tmp")
        (repdir/"regret_models.pt.tmp").replace(repdir/"regret_models.pt")
        write_json(repdir/"timing.json", {"repetition_seconds": time.perf_counter()-rep_start})
        write_json(done, receipt(repdir, ["metrics.json", "information_states.csv", "regret_models.pt",
                                         "panel_0.npz", "panel_1.npz", "timing.json"]))
        all_metrics.extend(metrics)
        upload(directory)
        print(json.dumps({"seed": seed, "checkpoint": checkpoint, "repetition": repetition,
                          "status": "paired_repetition_complete"}), flush=True)
        if stop_after is not None and repetition+1 >= stop_after and repetition+1 < plan["repetitions"]:
            return {"status": "interrupted_for_resume_test"}
    write_csv(directory/"metrics.csv", all_metrics)
    result = {"status": "complete", "seed": seed, "checkpoint": checkpoint,
        "smoke": smoke, "pilot": pilot, "fits": len(all_metrics), "repetitions": plan["repetitions"],
        "fit_steps": plan["fit_steps"], "last_session_seconds": time.perf_counter()-started,
        "peak_rss_mib": _peak_rss_mb(), "frozen_inputs_unchanged": True}
    write_json(directory/"result.json", result)
    files = [str(p.relative_to(directory)) for p in directory.rglob("*")
             if p.is_file() and p.name != "SUCCESS.json" and not p.name.endswith(".tmp")]
    # Also protect repetition receipts against alteration.
    files.extend(f"rep_{r:02d}/SUCCESS.json" for r in range(plan["repetitions"]))
    write_json(directory/"SUCCESS.json", receipt(directory, files))
    upload(directory)
    return result
