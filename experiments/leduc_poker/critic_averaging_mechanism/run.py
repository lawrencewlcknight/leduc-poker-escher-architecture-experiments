"""CLI: validated workers, smoke, engineering pilot and complete-block analysis."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from .config import ARMS, PILOT_SEED, configuration, tasks
from .solver import make_solver
from .worker import run_worker, write_json, sha256, source_identity, runtime, _peak_rss_mb
from .analyse import aggregate


def validate_pilot_source(payload):
    if payload["type"] != "experiment_35_full_training_state":
        raise ValueError("Mature pilot requires an Experiment 35 full state")
    iteration=int(payload["solver"]["num_iteration"])
    if iteration < 300:
        raise ValueError("Use an Experiment 35 state at iteration 300 or later")
    baseline=configuration("D",False)
    for key in ("num_layers","num_hiddens","num_traversals","baseline_buffer_size","baseline_batch_size",
                "baseline_network_train_steps","fixed_control_variate_beta","q_ensemble_size",
                "average_policy_network_layers","use_instantaneous_predictor","critic_target_average_window",
                "advantage_network_train_steps","learning_rate","average_policy_loss"):
        if json.dumps(payload["config"].get(key)) != json.dumps(baseline[key]):
            raise ValueError(f"Historical configuration mismatch: {key}")
    return iteration


def pilot(root, source_state=None, source_sha256=None):
    """Four independent short continuations; never pooled with confirmation."""
    from experiments.leduc_poker.ucv_residual_target_factorial.training_state import read_training_state, restore_training_state
    torch.set_num_threads(8)
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError("Pilot output must be empty")
    if source_state:
        if not source_sha256 or sha256(source_state)!=source_sha256:
            raise ValueError("Historical source checksum mismatch")
    results=[]
    for arm in ARMS:
        config=configuration(arm,smoke=source_state is None)
        solver=make_solver(PILOT_SEED,config)
        source_iteration=0
        if source_state is not None:
            # Reload per arm: CPU optimizer tensors can alias a loaded state.
            payload=read_training_state(source_state)
            source_iteration=validate_pilot_source(payload)
            # Drop the historical calibration state for this explicitly
            # controlled development intervention.
            adapted=dict(payload); adapted["calibration"]=None
            restore_training_state(solver,adapted,variant_id=payload["variant_id"],seed=payload["seed"],
                repository_commit=payload["repository_commit"],config=payload["config"],
                expected_state_type="experiment_35_full_training_state")
            solver._checkpoint_resume_rng_state=None
            for member in solver.q_value_trainer.members:member.install_snapshots()
            del payload, adapted
            gc.collect()
        times=[]
        for _ in range(3 if source_state is not None else 6):
            start=time.perf_counter();solver.iteration();times.append(time.perf_counter()-start)
        results.append({"arm":arm,"seconds_per_iteration":times,
                        "projection_600_hours":sum(times)/len(times)*600/3600,
                        "mature_replay":source_state is not None,"source_iteration":source_iteration,
                        "process_peak_rss_mib":_peak_rss_mb(),
                        "replay_rows":[len(m.buffer) for m in solver.q_value_trainer.members],
                        "component_seconds":solver.component_seconds})
        del solver
        gc.collect()
    write_json(root/"pilot.json",{"status":"complete","source_sha256":source_sha256,
        "source_code":source_identity(),"runtime":runtime(),"results":results,
        "scope":"engineering estimate only; no effect-based selection or confirmation",
        "warning":"Fresh reduced smoke is not a production runtime forecast" if source_state is None else
                  "Mature-iteration extrapolation excludes setup, checkpoint persistence and scheduled diagnostics"})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("worker","aggregate","smoke","pilot"))
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--task-index",type=int)
    p.add_argument("--source-state",type=Path)
    p.add_argument("--source-sha256")
    a=p.parse_args()
    if a.action=="worker":
        if a.task_index is None or not 0<=a.task_index<len(tasks()):p.error("Invalid task index")
        arm,seed=tasks()[a.task_index]
        run_worker(a.output_root/"workers"/f"task_{a.task_index:03d}_{arm}_{seed}",arm,seed)
    elif a.action=="aggregate":aggregate(a.output_root)
    elif a.action=="pilot":pilot(a.output_root,a.source_state,a.source_sha256)
    else:
        # The real solver smoke is also rerun on the cloud runtime before any
        # production array. Unit tests separately exercise identity and resume.
        for index,(arm,seed) in enumerate(tasks(True)):
            run_worker(a.output_root/"workers"/f"task_{index:03d}_{arm}_{seed}",arm,seed,True)
        aggregate(a.output_root,True)
        write_json(a.output_root/"SMOKE_SUCCESS.json",{"status":"complete","source":source_identity(),"runtime":runtime()})


if __name__=="__main__":main()
