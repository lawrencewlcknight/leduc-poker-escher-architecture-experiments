"""Experiment 49: fetch, CPU pilot, full frozen-policy audit, and aggregation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pyspiel
import torch
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from escher_poker.policy_snapshots import LoadedESCHERPolicy
from experiments.leduc_poker.grouped_ucv_temporal_head_to_head.run import (
    digest, fetch_source, source_records, write_csv, write_json,
)
from .config import (ARMS, BUDGETS, FIXED_UPDATES, HOURS, REPETITIONS, SEEDS, SOURCE_URI,
                     RNaDConfig, protocol)
from .evaluate import (derived_seed, evaluate_policy, load_policy, save_policy, stitched_case)
from .game import Blueprint
from .rnad import legal_distribution, new_value_model, prepare_value_model


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def provenance():
    repository = Path(__file__).resolve().parents[3]
    git_commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    return {
        "protocol": protocol(),
        "git_commit": git_commit,
        "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name)
                      for name in ("numpy", "torch", "scipy", "open_spiel")},
        "device": "cpu", "threads_per_process": 1,
    }


def case_id(case):
    budget = (f"s{case['seconds']:g}" if case["seconds"] is not None else f"u{case['updates']}")
    return f"seed{case['seed']}_h{case['hour']}_r{case['repeat']}_{case['arm']}_{budget}"


def cases(mode):
    result = []
    seeds = SEEDS if mode == "full" else SEEDS[:1]
    hours = HOURS if mode == "full" else (36,)
    repeats = REPETITIONS if mode == "full" else 2
    budgets = BUDGETS if mode == "full" else (.1, .5)
    for seed, hour, repeat in itertools.product(seeds, hours, range(repeats)):
        for arm, seconds in itertools.product(ARMS, budgets):
            result.append(dict(seed=seed, hour=hour, repeat=repeat, arm=arm,
                               seconds=seconds, updates=None))
        if hour == 36:
            result.append(dict(seed=seed, hour=hour, repeat=repeat, arm="rnad_bayesian",
                               seconds=.5, updates=None))
            if repeat == 0:
                for arm in ARMS:
                    result.append(dict(seed=seed, hour=hour, repeat=repeat, arm=arm,
                                       seconds=None, updates=FIXED_UPDATES if mode == "full" else 64))
    # Counterbalance arm order, retaining a reproducible execution manifest.
    np.random.default_rng(490049).shuffle(result)
    return result


def source_key(seed, hour):
    return f"seed{seed}_h{hour}"


def load_source(record):
    if digest(record["local_path"]) != record["sha256"]:
        raise ValueError("Source changed after validation")
    game = pyspiel.load_game("leduc_poker")
    loaded = LoadedESCHERPolicy(game, record["local_path"])
    blueprint = Blueprint(game, loaded)
    # Check the batched actor path against the native saved-policy loader.
    with torch.no_grad():
        batched, _ = legal_distribution(loaded._torch_model(torch.from_numpy(blueprint.features)),
                                       torch.from_numpy(blueprint.legal))
    if np.max(np.abs(batched.numpy() - blueprint.probs)) > 2e-6:
        raise ValueError("Batched blueprint actor does not reproduce native inference")
    observed = float(exploitability.nash_conv(game, blueprint.table) / 2)
    if abs(observed - float(record["source_exploitability"])) > 1e-5:
        raise ValueError("Reloaded blueprint exploitability disagrees with source")
    return loaded, blueprint


def prepare(record, output, panel, contract_hash, resume=False):
    seed = int(record["seed"])
    hour = int(float(record["checkpoint_target_active_hours"]))
    key = source_key(seed, hour)
    folder = Path(output) / "preparation"
    folder.mkdir(parents=True, exist_ok=True)
    path, metadata = folder / (key + ".pt"), folder / (key + ".json")
    if resume and metadata.exists():
        existing = json.loads(metadata.read_text())
        if (existing["contract_hash"] != contract_hash or existing["source_sha256"] != record["sha256"]
                or digest(path) != existing["critic_sha256"]):
            raise ValueError("Incompatible or corrupted critic preparation")
        existing["critic_path"] = str(path.resolve())
        return existing
    started = time.perf_counter()
    loaded, blueprint = load_source(record)
    loading_seconds = time.perf_counter() - started
    critic, fit = prepare_value_model(blueprint, loaded._torch_model,
                                      derived_seed(seed, hour, "critic"))
    torch.save(critic.state_dict(), path)
    baseline = evaluate_policy(blueprint, blueprint.probs, panel)
    if abs(baseline["ev_vs_blueprint"]) > 1e-10:
        raise ValueError("Self-play seat-averaged EV must be zero")
    save_policy(folder / (key + "_blueprint.npz"), blueprint, blueprint.probs)
    result = {"contract_hash": contract_hash, "source_sha256": record["sha256"],
              "critic_sha256": digest(path), "critic_path": str(path.resolve()),
              "fit": fit, "baseline": baseline, "seed": seed, "hour": hour,
              "blueprint_loading_indexing_seconds": loading_seconds,
              "blueprint_training_nodes": int(record["nodes_touched"]),
              "blueprint_completed_iteration": int(record["completed_iteration"]),
              "blueprint_active_seconds": float(record["active_seconds"]),
              "public_roots": len(blueprint.roots), "information_sets": len(blueprint.keys)}
    write_json(metadata, result)
    return result


_WORKER = {}


def worker_init(records, preparation, output, contract_hash, panel_paths):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    _WORKER.update(records=records, preparation=preparation, output=output,
                   contract_hash=contract_hash, cache={}, panel={})
    game = pyspiel.load_game("leduc_poker")
    for seed, path in panel_paths.items():
        snapshot = LoadedESCHERPolicy(game, path)
        _WORKER["panel"][seed] = Blueprint(game, snapshot).table
    # One-time framework startup is not a recurring decision cost.
    # Each actual resolver still creates its own fresh optimizer in its timer.
    dummy = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([dummy])
    (dummy * dummy).sum().backward()
    optimizer.step()


def run_case(case):
    name = case_id(case)
    key = source_key(case["seed"], case["hour"])
    if key not in _WORKER["cache"]:
        loaded, blueprint = load_source(_WORKER["records"][key])
        critic = new_value_model(loaded._torch_model)
        prep = _WORKER["preparation"][key]
        if digest(prep["critic_path"]) != prep["critic_sha256"]:
            raise ValueError("Prepared critic hash mismatch")
        critic.load_state_dict(torch.load(prep["critic_path"], map_location="cpu", weights_only=True))
        _WORKER["cache"][key] = loaded._torch_model, blueprint, critic.eval()
    actor, blueprint, critic = _WORKER["cache"][key]
    probabilities, root_metrics, timings = stitched_case(blueprint, actor, critic, case, RNaDConfig())
    scores = evaluate_policy(blueprint, probabilities, _WORKER["panel"],
                             _WORKER["preparation"][key]["baseline"])
    opponents = {k: v for k, v in scores["panel_evs"].items() if int(k) != case["seed"]}
    scores["other_seed_panel_mean_ev"] = float(np.mean(list(opponents.values())))
    output = Path(_WORKER["output"])
    path = output / "policies" / (name + ".npz")
    save_policy(path, blueprint, probabilities)
    reloaded = load_policy(path, blueprint)
    if not np.array_equal(probabilities, reloaded.action_probability_array):
        raise ValueError("Playable policy did not reload identically")
    result = {"case": case, "case_id": name, "contract_hash": _WORKER["contract_hash"],
              "source_sha256": _WORKER["records"][key]["sha256"],
              "policy_file": f"policies/{name}.npz", "policy_sha256": digest(path),
              "scores": scores, "timing": timings, "roots": root_metrics}
    write_json(output / "tasks" / (name + ".json"), result)
    return name, scores["exploitability_change"], timings["solve_seconds"]


def validate_completed(output, case, contract_hash):
    path = Path(output) / "tasks" / (case_id(case) + ".json")
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if (record["case"] != case or record["contract_hash"] != contract_hash
            or record.get("policy_file") != f"policies/{case_id(case)}.npz"
            or digest(Path(output) / record["policy_file"]) != record["policy_sha256"]):
        raise ValueError(f"Corrupt or incompatible completed task: {path}")
    return True


def mean_interval(values):
    from scipy.stats import t
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * se) if se is not None else None
    sd = float(values.std(ddof=1)) if n > 1 else None
    return dict(mean=mean, sd=sd, se=se, ci95_low=mean - half if half is not None else None,
                ci95_high=mean + half if half is not None else None, n_training_seeds=n)


def aggregate(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    expected = manifest["cases"]
    groups, seed_rows, task_rows, root_rows, panel_rows = {}, [], [], [], []
    preparation = {}
    for key in sorted(set(source_key(c["seed"], c["hour"]) for c in expected)):
        path = output / "preparation" / (key + ".json")
        if not path.exists():
            raise ValueError(f"Missing required task preparation: {key}")
        record = json.loads(path.read_text())
        if record["contract_hash"] != manifest["contract_hash"]:
            raise ValueError("Preparation contract mismatch")
        preparation[key] = record
    durations = []
    for case in expected:
        if not validate_completed(output, case, manifest["contract_hash"]):
            raise ValueError(f"Missing required task {case_id(case)}; cannot aggregate incomplete experiment")
        item = json.loads((output / "tasks" / (case_id(case) + ".json")).read_text())
        key = (case["hour"], case["arm"], case["seconds"], case["updates"], case["seed"])
        groups.setdefault(key, []).append(item)
        flat = dict(case)
        flat.update({k: v for k, v in item["scores"].items() if k != "panel_evs"})
        flat.update(item["timing"])
        task_rows.append(flat)
        durations.append(flat["solve_and_diagnostics_seconds"] + flat["exact_evaluation_seconds"])
        root_rows.extend(dict(case, **root) for root in item["roots"])
        baseline = preparation[source_key(case["seed"], case["hour"])]["baseline"]
        for opponent, ev in item["scores"]["panel_evs"].items():
            panel_rows.append(dict(case, opponent_seed=int(opponent),
                                   opponent_is_own_seed=int(opponent) == case["seed"],
                                   two_seat_ev=ev, blueprint_two_seat_ev=baseline["panel_evs"][opponent],
                                   ev_change=ev - baseline["panel_evs"][opponent]))
    for key, items in groups.items():
        hour, arm, seconds, updates, seed = key
        deltas = [i["scores"]["exploitability_change"] for i in items]
        seed_rows.append({
            "hour": hour, "arm": arm, "seconds": seconds, "updates": updates,
            "seed": seed, "nested_repetitions": len(items),
            "exploitability": float(np.mean([i["scores"]["exploitability"] for i in items])),
            "exploitability_change": float(np.mean(deltas)),
            "ev_vs_blueprint": float(np.mean([i["scores"]["ev_vs_blueprint"] for i in items])),
            "panel_ev_change": float(np.mean([i["scores"]["panel_ev_change"] for i in items])),
            "resolver_sd_exploitability": float(np.std(deltas, ddof=1)) if len(items) > 1 else None,
            "worst_repeat_exploitability_change": max(deltas),
        })
    for prep in preparation.values():
        seed_rows.append(dict(hour=prep["hour"], arm="blueprint", seconds=0., updates=None,
                              seed=prep["seed"], nested_repetitions=1,
                              exploitability=prep["baseline"]["exploitability"],
                              exploitability_change=0., ev_vs_blueprint=0., panel_ev_change=0.,
                              resolver_sd_exploitability=None, worst_repeat_exploitability_change=0.))
    summaries = []
    group_keys = list(dict.fromkeys((r["hour"], r["arm"], r["seconds"], r["updates"]) for r in seed_rows))
    for key in group_keys:
        entries = [r for r in seed_rows if (r["hour"], r["arm"], r["seconds"], r["updates"]) == key]
        for metric in ("exploitability", "exploitability_change", "ev_vs_blueprint", "panel_ev_change"):
            summary = dict(hour=key[0], arm=key[1], seconds=key[2], updates=key[3], metric=metric)
            summary.update(mean_interval([r[metric] for r in entries]))
            summary["worst_seed"] = max(r[metric] for r in entries) if "exploitability" in metric else min(r[metric] for r in entries)
            summaries.append(summary)
    paired = []
    for hour in sorted(set(r["hour"] for r in seed_rows)):
        for seconds in sorted(set(r["seconds"] for r in seed_rows if r["seconds"] is not None)):
            lookup = {(r["seed"], r["arm"]): r for r in seed_rows
                      if r["hour"] == hour and r["seconds"] == seconds}
            seeds = [s for s in SEEDS if (s, ARMS[0]) in lookup and (s, ARMS[1]) in lookup]
            if not seeds:
                continue
            differences = np.asarray([lookup[(s, ARMS[1])]["exploitability"] -
                                      lookup[(s, ARMS[0])]["exploitability"] for s in seeds])
            signs = np.asarray(list(itertools.product((-1, 1), repeat=len(seeds))))
            p = float(np.mean(np.abs(np.mean(signs * differences, axis=1)) >= abs(differences.mean()) - 1e-14))
            paired.append(dict(hour=hour, seconds=seconds, contrast="RNaD minus CFR+ exploitability",
                               **mean_interval(differences), exact_two_sided_sign_flip_p=p))
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    for filename, rows in (("cases.csv", task_rows), ("public_root_timing.csv", root_rows),
                           ("training_seed_summaries.csv", seed_rows), ("summary.csv", summaries),
                           ("seed_summary.csv", seed_rows), ("variant_aggregate_summary.csv", summaries),
                           ("paired_differences_vs_baseline.csv", [r for r in seed_rows if r["arm"] != "blueprint"]),
                           ("panel_matchups.csv", panel_rows),
                           ("paired_rnad_vs_cfr.csv", paired)):
        if rows:
            write_csv(analysis / filename, rows)
    write_csv(analysis / "critic_preparation.csv", [
        dict(seed=p["seed"], hour=p["hour"], source_sha256=p["source_sha256"],
             blueprint_exploitability=p["baseline"]["exploitability"],
             blueprint_loading_indexing_seconds=p["blueprint_loading_indexing_seconds"],
             blueprint_training_nodes=p["blueprint_training_nodes"],
             blueprint_completed_iteration=p["blueprint_completed_iteration"],
             blueprint_active_seconds=p["blueprint_active_seconds"], **p["fit"])
        for p in preparation.values()])
    write_json(analysis / "experiment_metadata.json", json.loads((output / "experiment_metadata.json").read_text()))
    # A labelled extrapolation from the pilot, NOT a promise or a GPU estimate.
    estimate = None
    if manifest["mode"] == "pilot":
        full = cases("full")
        estimates = []
        for c in full:
            matching = [r for r in task_rows if r["arm"] == c["arm"] and
                        (r["updates"] is None) == (c["updates"] is None)]
            if c["updates"] is None:
                overhead = np.mean([r["solve_and_diagnostics_seconds"] + r["exact_evaluation_seconds"]
                                    - 60 * r["seconds"] for r in matching])
                estimates.append(60 * c["seconds"] + max(0.0, overhead))
            else:
                estimates.append(float(np.mean([
                    r["solve_and_diagnostics_seconds"] * c["updates"] / r["updates"]
                    + r["exact_evaluation_seconds"] for r in matching])))
        estimate = {"serial_case_hours": sum(estimates) / 3600,
                    "ideal_case_hours_at_configured_workers": sum(estimates) / 3600 / manifest["workers"],
                    "qualification": "Pilot extrapolation; excludes new blueprint preparation, cloud startup, contention and stragglers."}
    write_json(analysis / "summary.json", {
        "mode": manifest["mode"], "contract_hash": manifest["contract_hash"],
        "cases_complete": len(expected), "primary": "36h, RNaD gadget, 0.5 seconds per resolve",
        "inferential_unit": "training seed; nested resolver repetitions averaged first",
        "scope": "pointwise exploratory intervals; pilot is engineering evidence only",
        "runtime_extrapolation": estimate, "groups": summaries, "paired_comparisons": paired,
        "total_case_seconds": sum(durations),
    })
    make_plot(analysis, summaries)
    write_json(output / "SUCCESS.json", {"mode": manifest["mode"], "contract_hash": manifest["contract_hash"],
                                          "cases_complete": len(expected)})


def make_plot(analysis, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm in ARMS:
        rows = sorted([r for r in summaries if r["hour"] == 36 and r["arm"] == arm
                       and r["seconds"] is not None and r["metric"] == "exploitability_change"],
                      key=lambda r: r["seconds"])
        if rows:
            means = np.asarray([r["mean"] for r in rows])
            ax.plot([r["seconds"] for r in rows], means, marker="o", label=arm)
            if all(r["ci95_low"] is not None for r in rows):
                ax.fill_between([r["seconds"] for r in rows], [r["ci95_low"] for r in rows],
                                [r["ci95_high"] for r in rows], alpha=.15)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xscale="log", xlabel="Seconds per public-root resolve (soft deadline)",
           ylabel="Exploitability change (NashConv / 2; lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(analysis / "exploitability_vs_budget.png", dpi=200)
    plt.close(fig)


def smoke(output):
    """No source download required: full-game zero identity and finite tiny solves."""
    from vr_deep_cfr.solver import MLP
    torch.set_num_threads(1)
    torch.manual_seed(490049)
    game = pyspiel.load_game("leduc_poker")
    blueprint = Blueprint(game, policy.TabularPolicy(game))
    actor = MLP(blueprint.features.shape[1], [16, 16], game.num_distinct_actions())
    critic = new_value_model(actor)
    config = replace(RNaDConfig(), batch_size=8)
    rows = []
    for arm in (*ARMS, "rnad_bayesian"):
        case = dict(seed=1, hour=36, repeat=0, arm=arm, seconds=0.0, updates=None)
        zero, _, _ = stitched_case(blueprint, actor, critic, case, config)
        if not np.array_equal(zero, blueprint.probs):
            raise ValueError("Zero-budget identity failed")
        case.update(seconds=None, updates=2)
        candidate, _, timings = stitched_case(blueprint, actor, critic, case, config)
        first_round = np.asarray(["[Round 1]" in key for key in blueprint.keys])
        if not np.array_equal(candidate[first_round], blueprint.probs[first_round]):
            raise ValueError("Resolving changed first-round play")
        score = float(exploitability.nash_conv(game, blueprint.as_policy(candidate)) / 2)
        if not np.isfinite(score):
            raise ValueError("Non-finite smoke exploitability")
        rows.append(dict(arm=arm, exploitability=score, **timings))
    write_json(Path(output) / "smoke.json", {"passed": True, "synthetic_uniform_source": True, "arms": rows})


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = Path(args.source_root)
    output = Path(args.output_root)
    contract = provenance()
    records = source_records(source, SEEDS, HOURS)
    contract["source_hashes"] = {r["filename"]: r["sha256"] for r in records}
    contract_hash = fingerprint(contract)
    if args.mode == "full":
        if not args.pilot_output:
            raise ValueError("Run the CPU pilot first and supply --pilot-output")
        pilot = json.loads((Path(args.pilot_output) / "SUCCESS.json").read_text())
        if pilot.get("mode") != "pilot" or pilot.get("contract_hash") != contract_hash:
            raise ValueError("Pilot must have completed with the same code, dependencies and sources")
    expected = cases(args.mode)
    manifest_path = output / "manifest.json"
    manifest = {"mode": args.mode, "workers": args.workers, "contract": contract,
                "contract_hash": contract_hash, "cases": expected}
    if output.exists() and any(output.iterdir()):
        if not args.resume or not manifest_path.exists():
            raise ValueError("Output directory is not empty; choose a new path or explicitly --resume")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    if not args.resume:
        write_json(output / "experiment_metadata.json", {
            "experiment_name": "leduc_poker_rnad_ucv_resolving",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "module": "experiments.leduc_poker.rnad_ucv_resolving.run",
            "manifest": manifest,
            "metric": "heads-up exploitability = exact OpenSpiel NashConv / 2",
            "output_convention_exceptions": [
                "Frozen Experiment 35 seed cohort, not Experiment 28 training seeds",
                "Test-time audit: no training-node curves, convergence AUC or training-time endpoints",
                "Pointwise 95% intervals shown; nested repetitions are not training replicates",
            ],
        })
    print(f"Experiment 49 {args.mode}: {len(expected)} cases, CPU workers={args.workers}", flush=True)
    panel, panel_paths = {}, {}
    indexed = {}
    for row in records:
        seed = int(row["seed"])
        hour = int(float(row["checkpoint_target_active_hours"]))
        indexed[source_key(seed, hour)] = row
        if hour == 36:
            _, bp = load_source(row)
            panel[seed], panel_paths[seed] = bp.table, row["local_path"]
    preparation = {}
    for key in sorted(set(source_key(c["seed"], c["hour"]) for c in expected)):
        print(f"Preparing independent value fit: {key}", flush=True)
        preparation[key] = prepare(indexed[key], output, panel, contract_hash, args.resume)
    pending = [c for c in expected if not validate_completed(output, c, contract_hash)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init,
                             initargs=(indexed, preparation, str(output.resolve()), contract_hash, panel_paths)) as pool:
        futures = [pool.submit(run_case, c) for c in pending]
        for future in as_completed(futures):
            name, change, seconds = future.result()
            print(f"Completed {name}: exploitability change={change:+.6f}, solve={seconds:.1f}s", flush=True)
    aggregate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--source-root", required=True)
    fetch.add_argument("--source-uri", default=SOURCE_URI)
    for name in ("smoke", "aggregate"):
        sub.add_parser(name).add_argument("--output-root", required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--source-root", required=True)
    execute.add_argument("--output-root", required=True)
    execute.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    execute.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    execute.add_argument("--pilot-output")
    execute.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_source(args.source_root, args.source_uri, SEEDS, HOURS)
        source_records(args.source_root, SEEDS, HOURS)
    elif args.command == "smoke":
        smoke(args.output_root)
    elif args.command == "aggregate":
        aggregate(args.output_root)
    else:
        run(args)


if __name__ == "__main__":
    main()
