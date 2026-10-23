"""Fixed-workload workers, atomic checkpoints and strict identity checks."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from escher_poker.policy_snapshots import save_torch_policy_snapshot
from experiments.leduc_poker.grouped_wide_policy_confirmation.worker import _peak_rss_mb
from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_csv, sha256, validate_playable_snapshot
from experiments.leduc_poker.ucv_residual_target_factorial.training_state import save_training_state, read_training_state
from .config import configuration, contract, schedule, tasks
from .solver import make_solver, capture_rng, restore_rng
from .state import pack, unpack
from .diagnostics import exact_diagnostics


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    tmp.replace(path)


def source_identity():
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for folder in ("adaptive_escher", "unbiased_escher", "vr_deep_cfr", "escher_poker", "experiments", "gcp"):
        for p in sorted((root/folder).rglob("*")):
            if p.suffix in (".py", ".sh"):
                digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    for name in ("requirements.txt", "setup.py", "pyproject.toml", "tests/test_critic_averaging_mechanism.py"):
        p = root/name
        if p.is_file():
            digest.update(name.encode()); digest.update(p.read_bytes())
    return digest.hexdigest()


def runtime():
    import pyspiel
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "open_spiel": getattr(pyspiel,"__version__","unknown"), "platform": platform.system(),
            "threads": torch.get_num_threads()}


def remote_copy(local, remote):
    subprocess.run(["gcloud", "storage", "cp", str(local), remote], check=True)


def save_checkpoint(directory, solver, arm, seed, config, identity, progress):
    # Two alternating complete states, with a small pointer committed last.
    slot = (solver.num_iteration // (2 if progress["smoke"] else 100)) % 2
    path = directory / f"state_{slot}.pt"
    started = time.perf_counter()
    save_training_state(path, pack(solver, arm, seed, config, identity, progress))
    marker = {"file": path.name, "sha256": sha256(path), "iteration": solver.num_iteration}
    remote = os.environ.get("EXP50_REMOTE_WORKER")
    if remote:
        if not remote.startswith("gs://") or any(c.isspace() for c in remote):
            raise ValueError("Invalid remote worker URI")
        # Persist every artifact needed by this state before advancing the
        # remote pointer, including earlier playable snapshots and metrics.
        for artifact in sorted(directory.iterdir()):
            if artifact.suffix in (".json", ".csv", ".pkl") and artifact.name not in ("latest.json", "SUCCESS.json"):
                remote_copy(artifact, remote+"/"+artifact.name)
        remote_copy(path, remote+"/"+path.name)
    marker["persistence_seconds"] = progress["persistence_seconds"] + time.perf_counter()-started
    write_json(directory/"latest.json", marker)
    if remote:
        remote_copy(directory/"latest.json", remote+"/latest.json")
    progress["persistence_seconds"] += time.perf_counter()-started


def verify_files(directory, receipt):
    for name, expected in receipt["files"].items():
        p = (directory/name).resolve()
        if directory.resolve() not in p.parents or not p.is_file() or sha256(p) != expected:
            raise ValueError(f"Missing or corrupt artifact: {name}")


def run_worker(directory, arm, seed, smoke=False, stop_after=None):
    if (arm,seed) not in tasks(smoke):
        raise ValueError("Worker not in the frozen seed/arm schedule")
    torch.set_num_threads(1 if smoke else 8)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    config, plan, identity = configuration(arm,smoke), schedule(smoke), source_identity()
    manifest = json.loads(json.dumps({"contract": contract(smoke), "arm": arm, "seed": seed,
                                      "config": config, "source": identity, "runtime": runtime()}))
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Source/configuration/runtime identity changed; use a new run ID")
    elif any(directory.iterdir()):
        raise ValueError("Nonempty worker without manifest")
    else:
        write_json(manifest_path, manifest)
    remote = os.environ.get("EXP50_REMOTE_WORKER")
    if remote:
        remote_copy(manifest_path, remote+"/manifest.json")
    if (directory/"SUCCESS.json").exists():
        verify_files(directory, json.loads((directory/"SUCCESS.json").read_text()))
        return json.loads((directory/"result.json").read_text())
    solver = make_solver(seed, config)
    progress = {"smoke": smoke, "metrics": [], "diagnostics": [], "learning_seconds": 0.,
                "diagnostic_seconds": 0., "persistence_seconds": 0.}
    if (directory/"latest.json").exists():
        marker = json.loads((directory/"latest.json").read_text())
        if marker["file"] not in ("state_0.pt", "state_1.pt"):
            raise ValueError("Invalid checkpoint filename")
        state_path = directory/marker["file"]
        if sha256(state_path) != marker["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        payload = read_training_state(state_path)
        progress = unpack(solver,payload,arm,seed,config,identity)
        progress["persistence_seconds"] = marker["persistence_seconds"]
        write_csv(directory/"metrics.csv", progress["metrics"])
        if progress["diagnostics"]:
            write_csv(directory/"diagnostics.csv", progress["diagnostics"])
        if solver.num_iteration != marker["iteration"]:
            raise ValueError("Checkpoint iteration mismatch")
    session_start = time.perf_counter()
    while solver.num_iteration < plan["end"]:
        started = time.perf_counter(); solver.iteration()
        progress["learning_seconds"] += time.perf_counter()-started
        iteration = solver.num_iteration
        if iteration in set(plan["evaluate"]) | set(plan["fit"]):
            row = solver.evaluate_mechanism(fit=iteration in plan["fit"])
            row.update(arm=arm,seed=seed,learning_seconds=progress["learning_seconds"])
            progress["metrics"].append(row)
            write_csv(directory/"metrics.csv", progress["metrics"])
            if iteration in plan["fit"]:
                save_torch_policy_snapshot(solver, directory/f"policy_{iteration:04d}.pkl",
                    seed=seed, iteration=iteration, arm=arm, config=config,
                    stage_label="critic_averaging_mechanism", checkpoint_target_nodes=solver.nodes_touched)
                validate_playable_snapshot("unbiased_control_variate_escher", directory/f"policy_{iteration:04d}.pkl")
        if iteration in plan["diagnose"]:
            started = time.perf_counter(); rng = capture_rng()
            try:
                summary, detail = exact_diagnostics(solver)
            finally:
                restore_rng(rng)
            progress["diagnostic_seconds"] += time.perf_counter()-started
            progress["diagnostics"].extend(dict(r,arm=arm,seed=seed) for r in summary)
            write_csv(directory/f"diagnostic_{iteration:04d}.csv",detail)
            write_csv(directory/"diagnostics.csv",progress["diagnostics"])
        if iteration in plan["save"]:
            write_csv(directory/"critic_fits.csv", [dict(row,fold=fold,arm=arm,seed=seed)
                for fold,member in enumerate(solver.q_value_trainer.members) for row in member.fit_diagnostics])
            save_checkpoint(directory,solver,arm,seed,config,identity,progress)
            print(json.dumps({"arm":arm,"seed":seed,"iteration":iteration,
                              "nodes":solver.nodes_touched,"learning_seconds":progress["learning_seconds"]}),flush=True)
            if stop_after is not None and iteration >= stop_after and iteration < plan["end"]:
                return {"status":"interrupted_for_resume_test","iteration":iteration}
    if solver.episode != 2*config["num_traversals"]*plan["end"]:
        raise RuntimeError("Trajectory count differs from contract")
    result = {"status":"complete","arm":arm,"seed":seed,"smoke":smoke,
        "iteration":solver.num_iteration,"nodes":solver.nodes_touched,"trajectories":solver.episode,
        "late_mean":float(np.mean([r["exact_average_exploitability"] for r in progress["metrics"] if r["iteration"] in plan["late"]])),
        "learning_seconds":progress["learning_seconds"], "diagnostic_seconds":progress["diagnostic_seconds"],
        "persistence_seconds":progress["persistence_seconds"],"component_seconds":solver.component_seconds,
        "peak_rss_mib":_peak_rss_mb(),"last_session_seconds":time.perf_counter()-session_start}
    write_json(directory/"result.json",result)
    files = {p.name:sha256(p) for p in directory.iterdir() if p.suffix in (".json",".csv",".pkl") and p.name not in ("SUCCESS.json","latest.json")}
    write_json(directory/"SUCCESS.json",{"status":"complete","files":files})
    return result
