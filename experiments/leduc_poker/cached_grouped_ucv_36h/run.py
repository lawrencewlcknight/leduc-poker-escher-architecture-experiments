"""Experiment 48 CLI: cloud smoke, five independent training tasks and analysis."""
import argparse
import json
from pathlib import Path
from .config import PRODUCTION_SEEDS, load_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("smoke","worker","aggregate","validate-reference"))
    p.add_argument("--output-root",type=Path)
    p.add_argument("--task-index",type=int)
    a=p.parse_args()
    reference=load_reference()
    if a.action=="validate-reference":
        print(json.dumps({"status":"valid","num_reference_rows":len(reference["trajectories"])}));return
    if a.output_root is None:
        p.error("--output-root is required")
    if a.action=="worker":
        if a.task_index is None or not 0<=a.task_index<len(PRODUCTION_SEEDS):
            p.error("worker --task-index must be 0 through 4")
        from .worker import run_worker,worker_directory
        seed=PRODUCTION_SEEDS[a.task_index]
        result=run_worker(seed=seed,directory=worker_directory(a.output_root,a.task_index,seed))
    elif a.action=="aggregate":
        from .analyse import aggregate
        result=aggregate(a.output_root)
    else:
        from .smoke import equivalence_smoke
        from .worker import run_worker,worker_directory
        from .analyse import aggregate
        from experiments.leduc_poker.promoted_ucv_cross_entropy_36h.common import write_json
        a.output_root.mkdir(parents=True,exist_ok=True)
        # Exercise production thread/minibatch shapes BEFORE any expensive jobs.
        report=equivalence_smoke(threads=8)
        write_json(a.output_root/"integration_equivalence.json",report)
        run_worker(seed=0,directory=worker_directory(a.output_root,0,0),smoke=True)
        result=aggregate(a.output_root,smoke=True)
        write_json(a.output_root/"SMOKE_SUCCESS.json",{"passed":True,"equivalence":report,
                   "playable_reload_and_aggregation":result["status"]})
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
