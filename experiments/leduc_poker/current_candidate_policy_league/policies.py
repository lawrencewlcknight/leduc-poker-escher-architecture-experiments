"""Inference-only adapters. All outputs must reproduce published diagnostics."""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from .sources import DEEP_REPO_REF


class KerasPlainPolicy:
    """Read the exact four saved Dense layers without installing TensorFlow."""
    def __init__(self, path):
        import h5py
        names = ("layers/dense", "layers/dense_1", "layers/dense_2", "out_layer")
        shapes = ((30, 256), (256, 256), (256, 128), (128, 3))
        with h5py.File(path) as f:
            datasets = []
            f.visititems(lambda n, o: datasets.append(n) if isinstance(o, h5py.Dataset) else None)
            if set(datasets) != {n + "/vars/" + k for n in names for k in ("0", "1")}:
                raise ValueError("ESCHER weights are not the selected plain four-Dense architecture")
            self.layers = [(np.array(f[n + "/vars/0"]), np.array(f[n + "/vars/1"])) for n in names]
        if any(w.shape != shape or b.shape != (shape[1],)
               for (w, b), shape in zip(self.layers, shapes)):
            raise ValueError("Unexpected ESCHER dimensions")

    def action_probabilities(self, state, player_id=None):
        p = state.current_player() if player_id is None else player_id
        legal = state.legal_actions(p)
        x = np.asarray(state.information_state_tensor(p), dtype=np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.where(x >= 0, x, 0.2 * x)
        logits = x[legal].astype(np.float64)
        probs = np.exp(logits - logits.max())
        return dict(zip(legal, probs / probs.sum()))


def load_neural(game, record, path):
    from experiments.leduc_poker.six_algorithm_final_policy_head_to_head.policies import (
        TorchStateDictPolicy,
    )
    a = record["algorithm"]
    if a == "escher":
        return KerasPlainPolicy(path)
    if a == "ucv":
        from experiments.leduc_poker.grouped_ucv_temporal_head_to_head.config import EXPECTED_CONFIG
        from escher_poker.policy_snapshots import LoadedESCHERPolicy
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        if (payload.get("game") != "leduc_poker" or payload.get("candidate_id") != "grouped_wide_ucv"
                or payload.get("seed") != record["seed"]
                or payload.get("checkpoint_id") != record["source_checkpoint"]
                or int(payload["nodes_touched"]) != record["nodes_touched"]):
            raise ValueError("Wrong grouped UCV snapshot")
        for k, v in EXPECTED_CONFIG.items():
            if payload["frozen_config"].get(k) != v:
                raise ValueError(f"Wrong grouped UCV configuration: {k}")
        return LoadedESCHERPolicy(game, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["seed"]) != record["seed"]:
        raise ValueError("Wrong snapshot seed")
    if a == "dream":
        if (payload.get("kind") != "dream_frozen_reservoir_distilled_policy"
                or payload.get("experiment_id") != 46
                or payload.get("checkpoint_id") != record["source_checkpoint"]
                or int(payload["metrics"]["nodes_touched"]) != record["nodes_touched"]):
            raise ValueError("Wrong DREAM fitted policy")
        return TorchStateDictPolicy(game, path, "dream")
    if a == "deep_cfr":
        if payload.get("type") != "deep_cfr_policy_snapshot" or payload.get("game") != "leduc_poker":
            raise ValueError("Wrong Deep CFR policy")
        return TorchStateDictPolicy(game, path, "deep_cfr")
    if a.startswith("vr_deep_"):
        from vr_deep_cfr.policy_snapshots import LoadedVRPolicy
        if payload.get("algorithm_id") != a or payload.get("game") != "leduc_poker":
            raise ValueError("Wrong VR policy")
        return LoadedVRPolicy(game, path)
    raise ValueError(a)


def sd_policies(game, records, path):
    """Use the pinned original exact own-reach-weighted uniform-mixture evaluator."""
    import subprocess
    repo = Path(os.environ["DEEP_CFR_REPO"])
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != DEEP_REPO_REF:
        raise ValueError(f"SD-CFR evaluator must be pinned to {DEEP_REPO_REF}")
    sys.path.append(str(repo))
    module = importlib.import_module("deep_cfr_poker.sd_cfr")
    if not Path(module.__file__).resolve().is_relative_to(repo.resolve()):
        raise ValueError("SD-CFR imported from the wrong checkout")
    archive = module.SDCFRArchive.load(path)
    if archive.game_name != "leduc_poker" or int(archive.metadata["seed"]) != records[0]["seed"]:
        raise ValueError("Wrong SD-CFR archive")
    checkpoints = [r["iteration"] for r in records]
    for entries in archive.entries_by_player.values():
        if any(cp not in {x.iteration for x in entries} for cp in checkpoints):
            raise ValueError("SD-CFR archive lacks a requested checkpoint")
    return module.exact_average_policies_at_checkpoints(game, archive, checkpoints, weighting="uniform")
