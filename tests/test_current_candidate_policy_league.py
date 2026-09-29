"""Experiment 46 contracts, exact evaluation and seed-cluster inference."""
from copy import deepcopy
import importlib.util
import itertools
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pyspiel
import pytest
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability

from experiments.leduc_poker.current_candidate_policy_league import sources as src
from experiments.leduc_poker.current_candidate_policy_league import run as runner


def records(smoke=False):
    rows=[]
    for ep, algorithms in src.ENDPOINTS.items():
        for a in algorithms:
            seeds=src.DEEP_LONG_SEEDS if a=="deep_cfr" and ep=="time_36h" else src.SEEDS[a]
            for seed in seeds[:1] if smoke else seeds:
                rows.append(dict(endpoint=ep,algorithm=a,seed=seed,nodes_touched=15_010_000,
                                 active_seconds=129601,exploitability=.1,
                                 cohort="ucv_fresh" if a=="ucv" else
                                 "deep_long" if a=="deep_cfr" and ep=="time_36h" else "historical_common"))
    return rows


def pairs(rows):
    out=[]
    for ep, algorithms in src.ENDPOINTS.items():
        for a,b in itertools.combinations(algorithms,2):
            aa=[r for r in rows if r["endpoint"]==ep and r["algorithm"]==a]
            bb=[r for r in rows if r["endpoint"]==ep and r["algorithm"]==b]
            for i,ra in enumerate(aa):
                for j,rb in enumerate(bb):
                    out.append(dict(endpoint=ep,algorithm_a=a,algorithm_b=b,
                                    seed_a=ra["seed"],seed_b=rb["seed"],A_EV_seat_averaged=.01+.002*i-.001*j))
    return out


def test_complete_design():
    rows=records()
    src.validate_records(rows)
    assert len(rows)==60
    assert len(pairs(rows))==775
    assert len(pairs(records(True)))==31
    assert set(src.ENDPOINTS["time_36h"])=={"deep_cfr","sd_cfr","dream","escher","ucv"}


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "wrong_seed", "old_budget", "short_time"])
def test_incomplete_design_rejected(mutation):
    rows=records()
    if mutation=="drop": rows.pop()
    elif mutation=="duplicate": rows.append(deepcopy(rows[0]))
    elif mutation=="wrong_seed": rows[0]["seed"]=-1
    elif mutation=="old_budget": rows[0]["nodes_touched"]=14_000_000
    else: rows[-1]["active_seconds"]=11*3600
    with pytest.raises(ValueError): src.validate_records(rows)


def test_historical_deep_boundary_exception_is_narrow():
    rows=records()
    rows[0]["nodes_touched"]=14_882_576
    src.validate_records(rows)
    next(r for r in rows if r["algorithm"]=="ucv")["nodes_touched"]=14_882_576
    with pytest.raises(ValueError): src.validate_records(rows)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../../escape"])
def test_no_path_traversal(path):
    with pytest.raises(ValueError): src.safe_relative(path)


def test_duplicate_source_rows_rejected():
    with pytest.raises(ValueError): src.one([{"seed":1},{"seed":1}],seed=1)


def test_exact_reload_and_self_play():
    game=pyspiel.load_game("leduc_poker")
    p=policy.TabularPolicy(game)
    exp=exploitability.nash_conv(game,p)/2
    tab,value=runner.checked_table(game,p,exp)
    assert value==pytest.approx(exp)
    with pytest.raises(ValueError,match="differs"):
        runner.checked_table(game,p,exp+.01)


def test_negative_probabilities_rejected():
    game=pyspiel.load_game("leduc_poker")
    p=policy.TabularPolicy(game)
    p.action_probability_array[:]=-1
    with pytest.raises(ValueError): runner.checked_table(game,p,.1)


def test_cluster_bootstrap_and_strength_are_reproducible():
    rows=records(); values=pairs(rows)
    result=runner.summarise(rows,values,replicates=200)
    assert result==runner.summarise(rows,values,replicates=200)
    summary,strength,metrics=result
    assert len(summary)==31 and len(strength)==12 and len(metrics)==12
    assert all(r["n_matchups"]==25 for r in summary)
    assert all(r["bootstrap_se"]>0 for r in summary)
    for ep in src.ENDPOINTS:
        assert sum(r["mean_ev"] for r in strength if r["endpoint"]==ep)==pytest.approx(0,abs=1e-12)
    with pytest.raises(ValueError,match="Incomplete"):
        runner.summarise(rows,values[:-1],replicates=10)


def test_batch_is_bounded_evaluation_only(tmp_path):
    path=Path(__file__).parents[1]/"gcp/current_candidate_policy_league_batch.py"
    spec=importlib.util.spec_from_file_location("league_batch",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    job=module.build_job(SimpleNamespace(bucket="gs://example",run_id="exp46-league-test",
                                         repo_ref="a"*40,service_account="runner@example"))
    task=job["taskGroups"][0]
    assert task["taskCount"]==1 and task["taskSpec"]["maxRetryCount"]==0
    assert task["taskSpec"]["maxRunDuration"]=="28800s"
    script=task["taskSpec"]["runnables"][0]["script"]["text"]
    assert "h5py==3.13.0" in script and src.DEEP_REPO_REF in script
    assert "run smoke" in script and "batch jobs submit" not in script
    file=tmp_path/"batch.sh"; file.write_text(script)
    subprocess.run(["bash","-n",str(file)],check=True)


def test_keras_adapter_plain_weights(tmp_path):
    h5py=pytest.importorskip("h5py")
    from experiments.leduc_poker.current_candidate_policy_league.policies import KerasPlainPolicy
    path=tmp_path/"policy.h5"
    with h5py.File(path,"w") as f:
        for name,shape in zip(("layers/dense","layers/dense_1","layers/dense_2","out_layer"),
                              ((30,256),(256,256),(256,128),(128,3))):
            f[name+"/vars/0"]=np.zeros(shape,np.float32)
            f[name+"/vars/1"]=np.zeros(shape[1],np.float32)
    game=pyspiel.load_game("leduc_poker")
    baseline=policy.TabularPolicy(game)
    runner.checked_table(game,KerasPlainPolicy(path),exploitability.nash_conv(game,baseline)/2)
    with h5py.File(path,"a") as f: f["unexpected"]=np.zeros(1)
    with pytest.raises(ValueError): KerasPlainPolicy(path)


def test_sd_archive_prefixes_without_training(tmp_path, monkeypatch):
    import torch
    from experiments.leduc_poker.current_candidate_policy_league.policies import sd_policies
    repo = Path(__file__).resolve().parents[3] / "leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments"
    if not repo.exists():
        pytest.skip("Sibling pinned SD-CFR checkout not present")
    monkeypatch.setenv("DEEP_CFR_REPO", str(repo))
    monkeypatch.syspath_prepend(str(repo))
    from deep_cfr_poker.sd_cfr import SDCFRArchive, AdvantageSnapshot
    from deep_cfr_poker.networks import build_network
    archive = SDCFRArchive(game_name="leduc_poker", num_players=2, num_actions=3,
                          embedding_size=30, advantage_network_type="mlp",
                          advantage_network_layers=(8,), metadata={"seed":1234})
    for player in (0,1):
        for iteration in (1,2):
            torch.manual_seed(10*player+iteration)
            model=build_network("mlp",30,(8,),3)
            archive._entries[player].append(AdvantageSnapshot(player,iteration,model.state_dict()))
    path=tmp_path/"archive.pt"
    archive.save(path)
    game=pyspiel.load_game("leduc_poker")
    result=sd_policies(game,[{"seed":1234,"iteration":1},{"seed":1234,"iteration":2}],path)
    assert set(result)=={1,2}
    for p in result.values():
        exp=exploitability.nash_conv(game,p)/2
        runner.checked_table(game,p,exp)
    with pytest.raises(ValueError,match="lacks"):
        sd_policies(game,[{"seed":1234,"iteration":3}],path)
