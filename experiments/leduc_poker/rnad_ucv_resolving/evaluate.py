"""Deadline accounting, stitched-policy evaluation and reloadable artefacts."""

import hashlib
import time
from pathlib import Path

import numpy as np
from open_spiel.python.algorithms import exploitability

from escher_poker.checkpoint_analysis import exact_seat_averaged_value_for_a
from .game import CFRPlus, PublicProblem, check_probabilities
from .rnad import NeuralResolver


def derived_seed(*parts):
    text = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little")


def resolve(blueprint, root_key, hero, arm, actor, critic, seed, config,
            seconds=None, updates=None):
    if (seconds is None) == (updates is None):
        raise ValueError("Specify exactly one of seconds or updates")
    if (seconds is not None and (not np.isfinite(seconds) or seconds < 0)
            or updates is not None and updates < 0):
        raise ValueError("Negative or invalid resolve budget")
    if arm not in ("cfr_plus_gadget", "rnad_gadget", "rnad_bayesian"):
        raise ValueError("Unknown resolver arm")
    start = time.perf_counter()
    # Do not even construct a subgame for the explicit zero-budget identity check.
    if seconds == 0 or updates == 0:
        return {}, {"root": root_key, "hero": hero, "accepted_updates": 0,
                    "attempted_updates": 0, "elapsed_seconds": time.perf_counter() - start,
                    "fallback": True, "reason": "zero_budget"}
    problem = PublicProblem(blueprint, root_key, hero, gadget=arm != "rnad_bayesian")
    last = problem.blueprint.copy()
    preparation = time.perf_counter() - start
    accepted, attempted, samples, decisions = 0, 0, 0, 0
    fit_seconds, extraction_seconds = 0.0, 0.0
    solver = None
    if problem.mass > 0 and (seconds is None or preparation < seconds):
        solver = (CFRPlus(problem) if arm == "cfr_plus_gadget" else
                  NeuralResolver(problem, actor, critic, seed, config))
        preparation = time.perf_counter() - start
        while (accepted < updates if updates is not None else time.perf_counter() - start < seconds):
            before = time.perf_counter()
            solver.step()
            fit_seconds += time.perf_counter() - before
            attempted += 1
            before = time.perf_counter()
            candidate = solver.probabilities()
            check_probabilities(candidate, problem.legal)
            extraction_seconds += time.perf_counter() - before
            if seconds is not None and time.perf_counter() - start > seconds:
                # Never deploy a partially computed or over-budget update.
                break
            last = candidate.copy()
            accepted += 1
        samples = getattr(solver, "samples", 0)
        decisions = getattr(solver, "decisions", 0)
    # Export only the resolving player's actions, not the opponent's new strategy.
    result = {int(row): last[i].copy() for i, row in enumerate(problem.global_rows)
              if problem.row_players[i] == hero}
    elapsed = time.perf_counter() - start
    before = time.perf_counter()
    violation = (problem.bound_violation(last) if problem.gadget and problem.mass > 0 else None)
    return result, {
        "root": root_key, "hero": hero, "accepted_updates": accepted,
        "attempted_updates": attempted, "elapsed_seconds": elapsed,
        "setup_seconds": preparation, "fit_seconds": fit_seconds,
        "extraction_seconds": extraction_seconds, "counterfactual_root_mass": problem.mass,
        "sampled_continuations": samples, "sampled_decisions": decisions,
        "fallback": accepted == 0,
        "reason": "zero_reach" if problem.mass == 0 else ("no_completed_update" if not accepted else "completed"),
        "budget_overrun_seconds": max(0.0, elapsed - seconds) if seconds is not None else 0.0,
        "max_opponent_bound_excess": violation,
        "exact_diagnostic_seconds": time.perf_counter() - before,
    }


def stitched_case(blueprint, actor, critic, case, config):
    result = blueprint.probs.copy()
    root_metrics = []
    seen = set()
    started = time.perf_counter()
    for root_key in blueprint.roots:
        for hero in (0, 1):
            seed = derived_seed(case["seed"], case["hour"], case["repeat"], root_key, hero)
            rows, metrics = resolve(blueprint, root_key, hero, case["arm"], actor, critic,
                                    seed, config, seconds=case["seconds"], updates=case["updates"])
            for row, probabilities in rows.items():
                if row in seen:
                    raise ValueError("An information set belongs to two public-root solves")
                result[row] = probabilities
                seen.add(row)
            root_metrics.append(metrics)
    check_probabilities(result, blueprint.legal)
    elapsed = np.asarray([m["elapsed_seconds"] for m in root_metrics])
    violations = [m["max_opponent_bound_excess"] for m in root_metrics
                  if m.get("max_opponent_bound_excess") is not None]
    return result, root_metrics, {
        "solve_and_diagnostics_seconds": time.perf_counter() - started,
        "solve_seconds": float(elapsed.sum()),
        "latency_median_seconds": float(np.median(elapsed)),
        "latency_p95_seconds": float(np.quantile(elapsed, .95)),
        "latency_max_seconds": float(elapsed.max()),
        "accepted_updates": sum(m["accepted_updates"] for m in root_metrics),
        "attempted_updates": sum(m["attempted_updates"] for m in root_metrics),
        "fallback_fraction": float(np.mean([m["fallback"] for m in root_metrics])),
        "overrun_fraction": float(np.mean([m.get("budget_overrun_seconds", 0) > 1e-3
                                             for m in root_metrics])),
        "max_bound_excess": max(violations) if violations else None,
        "roots_evaluated": len(root_metrics),
    }


def save_policy(path, blueprint, probabilities):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    check_probabilities(probabilities, blueprint.legal)
    with path.open("wb") as handle:
        np.savez_compressed(handle, keys=np.asarray(blueprint.keys), probabilities=probabilities)


def load_policy(path, blueprint):
    with np.load(path, allow_pickle=False) as saved:
        if saved["keys"].tolist() != blueprint.keys:
            raise ValueError("Saved policy has incompatible information-set indexing")
        probabilities = saved["probabilities"].copy()
    return blueprint.as_policy(probabilities)


def evaluate_policy(blueprint, probabilities, panel, baseline_scores=None):
    started = time.perf_counter()
    candidate = blueprint.as_policy(probabilities)
    own = exact_seat_averaged_value_for_a(blueprint.game, candidate, blueprint.table)
    against_panel = {
        str(seed): exact_seat_averaged_value_for_a(blueprint.game, candidate, opponent)["A_EV_seat_averaged"]
        for seed, opponent in panel.items()
    }
    result = {
        "exploitability": float(exploitability.nash_conv(blueprint.game, candidate) / 2),
        "ev_vs_blueprint": own["A_EV_seat_averaged"],
        "ev_vs_blueprint_seat0": own["A_EV_as_player0"],
        "ev_vs_blueprint_seat1": own["A_EV_as_player1"],
        "panel_evs": against_panel,
        "panel_mean_ev": float(np.mean(list(against_panel.values()))),
        "exact_evaluation_seconds": time.perf_counter() - started,
    }
    if baseline_scores is not None:
        result["exploitability_change"] = result["exploitability"] - baseline_scores["exploitability"]
        result["panel_ev_change"] = result["panel_mean_ev"] - baseline_scores["panel_mean_ev"]
    return result
