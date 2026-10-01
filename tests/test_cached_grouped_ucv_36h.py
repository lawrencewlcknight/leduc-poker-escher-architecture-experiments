from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from experiments.leduc_poker.cached_grouped_ucv_36h import config
from experiments.leduc_poker.cached_grouped_ucv_36h.worker import (
    configuration, make_solver, run_worker, verify_completed,
)
from experiments.leduc_poker.grouped_wide_policy_confirmation import config as base
from experiments.leduc_poker.cached_grouped_ucv_36h.smoke import assert_exact

ROOT = Path(__file__).resolve().parents[1]


def test_only_cache_and_safety_ceiling_differ():
    different={k for k in set(config.CANDIDATE_CONFIG)|set(base.CANDIDATE_CONFIG)
               if config.CANDIDATE_CONFIG.get(k)!=base.CANDIDATE_CONFIG.get(k)}
    assert different=={"cache_frozen_critic_targets","max_num_iterations"}
    assert config.PRODUCTION_SEEDS==base.PRODUCTION_SEEDS
    assert config.checkpoint_schedule()==base.checkpoint_schedule()
    assert [r["target_active_hours"] for r in config.checkpoint_schedule()[:-1]]==list(range(2,37,2))
    assert config.checkpoint_schedule()[-1]["target_nodes"]==15_000_000
    assert config.contract()["training_state_retention"]=="none"


def test_frozen_reference_complete_and_correct_primary():
    ref=config.load_reference()
    assert len(ref["trajectories"])==540
    assert set(ref["milestones"])==set(map(str,config.PRODUCTION_SEEDS))
    for r in ref["milestones"].values():
        assert 0<r["iteration"]<base.MAX_ITERATIONS
        assert 36<=r["actual_hours"]<37
    sd=[r for r in ref["trajectories"] if r["algorithm_id"]=="sd_cfr"]
    assert len(sd)==90
    assert len({r["seed"] for r in sd})==5
    assert len([r for r in ref["node_endpoints"] if r["algorithm_id"]=="sd_cfr"])==5


def test_no_silent_restart_or_contract_change(tmp_path):
    assert verify_completed(tmp_path,{"run":1}) is None
    (tmp_path/"run_manifest.json").write_text('{"run": 1}')
    with pytest.raises(ValueError,match="Interrupted"):
        verify_completed(tmp_path,{"run":1})
    with pytest.raises(ValueError,match="different"):
        verify_completed(tmp_path,{"run":2})


def test_exact_checker_catches_policy_or_replay_change():
    a={"replay":torch.tensor([1.,2.]),"state":[3,4]}
    assert_exact(a,deepcopy(a))
    b=deepcopy(a);b["replay"][0]+=0.001
    with pytest.raises(AssertionError):
        assert_exact(a,b)


def test_skipped_early_fit_does_not_claim_a_cache():
    torch.set_num_threads(1)
    c=configuration(True);c["baseline_batch_size"]=2048
    solver=make_solver(0,c)
    solver.collect_training_data(0)
    for m in solver.q_value_trainer.members:
        assert len(m.buffer)<2048
        assert m.train_model(1) is None
    assert solver.critic_cache_rows==0


def builder():
    spec=importlib.util.spec_from_file_location("exp48_batch",ROOT/"gcp/cached_grouped_ucv_36h_batch.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind",["controller","smoke","train","aggregate"])
def test_cloud_jobs_are_bounded_and_shell_valid(kind):
    args=SimpleNamespace(kind=kind,project="p",region="europe-west1",bucket="gs://b",
        run_id="exp48-test",repo_ref="a"*40,service_account="runner@p.iam.gserviceaccount.com",parallelism=5)
    job=builder().build_job(args);group=job["taskGroups"][0];task=group["taskSpec"]
    assert task["maxRetryCount"]==0
    assert group["taskCountPerNode"]==1
    if kind=="train":
        assert group["taskCount"]==group["parallelism"]==5
        assert task["maxRunDuration"]=="194400s"
    script=task["runnables"][0]["script"]["text"]
    subprocess.run(["bash","-n"],input=script,text=True,check=True)
    assert "training_states" not in script
    if kind=="controller":
        assert "export RUN_ID EXP48_REMOTE_CONTROLLER=1" in script
    if kind=="aggregate":
        assert '"$OUTPUT/workers"' in script
        assert '"$OUTPUT/analysis"' in script


def test_launcher_orders_smoke_before_training_and_has_no_automatic_resume():
    text=(ROOT/"gcp/run_cached_grouped_ucv_36h.sh").read_text()
    assert text.index('ensure "$RUN_ID-smoke"')<text.index('ensure "$RUN_ID-train"')<text.index('ensure "$RUN_ID-aggregate"')
    assert "SMOKE_SUCCESS.json" in text
    assert "REPO_REF does not contain Experiment 48" in text
    subprocess.run(["bash","-n",str(ROOT/"gcp/run_cached_grouped_ucv_36h.sh")],check=True)


def test_partial_or_wrong_experiment_is_rejected_by_aggregator(tmp_path):
    from experiments.leduc_poker.cached_grouped_ucv_36h.analyse import load_workers
    with pytest.raises(ValueError,match="Missing required seeds"):
        load_workers(tmp_path,True)


def test_smoke_endpoints_do_not_count_checkpoints_as_independent_seeds():
    from experiments.leduc_poker.cached_grouped_ucv_36h.analyse import endpoint_rows
    rows=[{"algorithm_id":"test","seed":0,"hours":h} for h in (0.001,0.002,0.003)]
    final=endpoint_rows([],rows,{},True)
    assert len(final)==1
    assert final[0]["hours"]==0.003
    assert final[0]["endpoint"]=="smoke_final"


def test_production_aggregation_reference_joins(tmp_path,monkeypatch):
    """Synthetic worker metrics exercise production joins, not policy quality."""
    from experiments.leduc_poker.cached_grouped_ucv_36h import analyse
    ref=config.load_reference()
    rows=[r for r in ref["trajectories"]+ref["node_endpoints"]
          if r["algorithm_id"]==config.CANDIDATE_ID]
    metrics=[]
    for r in rows:
        for index,policy in enumerate(analyse.base.DIAGNOSTIC_POLICY_ORDER):
            metrics.append({"seed":r["seed"],"policy_id":policy,
                "policy_label":analyse.base.POLICY_LABELS[policy],
                "checkpoint_id":r.get("checkpoint_id","node_15m"),
                "checkpoint_type":"active_time" if r.get("hours") is not None else "nodes",
                "checkpoint_target_active_hours":r.get("hours"),
                "actual_active_hours":r["actual_hours"],"nodes_touched":r["nodes"],
                "completed_iteration":r["iteration"],
                "exploitability":r["exploitability"]+index*0.001})
    workers={}
    for seed in config.PRODUCTION_SEEDS:
        directory=tmp_path/"workers"/str(seed);directory.mkdir(parents=True)
        (directory/"component_timings.csv").write_text("seed,critic_fit\n%d,1.0\n"%seed)
        workers[seed]=(directory/"worker_result.json",{"matched_workload":{},"repository_commit":"synthetic"})
    monkeypatch.setattr(analyse,"load_workers",lambda *args:workers)
    monkeypatch.setattr(analyse.base,"_metrics",lambda *args,**kwargs:(metrics,[{}]*95,[]))
    result=analyse.aggregate(tmp_path)
    assert result["num_workers"]==5
    assert len(result["comparisons"])==24
    assert all(r["n"]==5 for r in result["endpoint_summary"])
    assert all(not r["all_seeds_reached"] and "mean" not in r
               for r in result["matched_workload_summary"])
    paired=[r for r in result["comparisons"] if r["comparator"]==config.CANDIDATE_ID]
    assert len(paired)==4
    assert all(r["mean"]==0 for r in paired)
    output=tmp_path/"analysis"
    combined=analyse.base._read_csv(output/"all_algorithm_checkpoint_policy_metrics.csv")
    assert len(combined)==630
    assert {r["algorithm_id"] for r in combined}=={
        r["algorithm_id"] for r in ref["trajectories"]}|{config.ALGORITHM_ID}
    assert all("nodes_touched" in r and "checkpoint_target_active_hours" in r for r in combined)


@pytest.mark.smoke
def test_full_smoke_and_corrupt_output_detection(tmp_path):
    from experiments.leduc_poker.cached_grouped_ucv_36h.smoke import equivalence_smoke
    from experiments.leduc_poker.cached_grouped_ucv_36h.analyse import aggregate
    from experiments.leduc_poker.cached_grouped_ucv_36h.worker import worker_directory
    report=equivalence_smoke(threads=8)
    assert report["passed"] and report["bitwise_equal"]
    directory=worker_directory(tmp_path,0,0)
    result=run_worker(seed=0,directory=directory,smoke=True)
    assert not list(tmp_path.rglob("*.pt"))
    assert len(result["snapshots"])==4
    again=run_worker(seed=0,directory=directory,smoke=True)
    assert again==json.loads(json.dumps(result))
    summary=aggregate(tmp_path,smoke=True)
    assert summary["num_playable_snapshots"]==4
    assert summary["experiment_id"]==48
    assert all(r["n"]==1 for r in summary["endpoint_summary"])
    assert all(r["all_seeds_reached"] and r["n"]==1
               for r in summary["matched_workload_summary"])
    for name in ("exploitability_by_training_time.png","exploitability_by_nodes_touched.png",
                 "average_policy_diagnostics_by_training_time.png","endpoint_summary.csv"):
        assert (tmp_path/"analysis"/name).stat().st_size>0
    snapshot=directory/result["snapshots"][0]["relative_path"]
    snapshot.write_bytes(b"corrupted")
    with pytest.raises(ValueError,match="corrupt"):
        aggregate(tmp_path,smoke=True)
