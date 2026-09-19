# Experiment 43: integrated extended average-policy fitting

Experiment 43 integrates Experiment 41's selected 1,700-update grouped policy
refinement into fresh 36-hour UCV-ESCHER trajectories. It uses the five
Experiment 29 seed labels so every historical comparator can be imported
unchanged.

## Frozen design

The regret and critic learner is exactly the Experiment 29 candidate:

- fixed control-variate beta of one, two critics and no predictive path;
- four-fit averaged critic target;
- three-layer, 64-hidden-unit cumulative-regret and average-policy networks;
- one-million-entry replay reservoirs;
- ordinary reset soft-target cross-entropy policy fitting with 5,000 minibatch
  updates;
- two-hour active-time checkpoints, a 15-million-node checkpoint and a
  36-active-hour horizon.

After every saved checkpoint, the worker saves the ordinary policy, clones it,
constructs one iteration-weighted mean target per observed information state,
and performs exactly 1,700 additional full-batch Adam updates. The refinement
uses learning rate `3e-4` and gradient clipping at `10.0`. It receives a fresh
optimizer at each checkpoint. All Python, NumPy, PyTorch and CUDA RNG states
are restored afterwards, and refinement, evaluation and snapshot time are
excluded from the active-training clock. The refined clone can therefore not
alter subsequent UCV experience or regret learning.

At the final 36-hour checkpoint, playable refined policies are additionally
saved at 400, 800, 1,200 and 1,600 updates. The 1,700-update policy is the sole
pre-specified primary candidate; no measured checkpoint controls training or
selection.

Training the refinement consumes only replay tensors, action targets, legal
action masks, iteration labels and information-state identities. It does not
enumerate the game tree or require a best response. Exact exploitability,
exact tabular averages and reach-weighted policy errors are Leduc-only
post-training diagnostics.

## Analysis

The aggregate analysis imports the frozen Experiment 29 analysis, which
contains Deep CFR, original UCV-ESCHER, simplified UCV-ESCHER and the revised
cross-entropy UCV candidate on the same seed labels. It produces:

- all five algorithms by active training time and nodes touched;
- ordinary versus refined policy trajectories;
- neural-to-empirical and neural-to-exact distillation gaps;
- paired seed-level endpoint effects;
- final-checkpoint refinement-horizon diagnostics;
- late-window RMSSD, slope and worst-rebound summaries;
- refinement runtime, memory, policy-error and exact ordinary-versus-refined
  head-to-head diagnostics.

This remains post-selection development evidence because Experiment 41 used
the same underlying seed labels to select 1,700 updates.

## Local smoke test

From the repository root:

```bash
./gcp/run_integrated_extended_average_policy_fitting.sh smoke-local
```

The smoke test executes real ordinary fitting, grouped refinement, playable
snapshot validation, continuation-state restoration, historical join and all
aggregate charts on a miniature schedule.

## Google Cloud run

Use the same cloud variables as Experiments 19--42:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp43-fit-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_integrated_extended_average_policy_fitting.sh run
```

Five `n2-standard-8` workers run in parallel. The cloud-owned controller runs
the cloud smoke test, production array and aggregation, so the laptop may be
disconnected after submission. Use the same launcher with `status` or `resume`
to inspect or recover the run.

Production allows 54 hours per worker, one automatic retry, and a 150 GB
balanced boot disk. Training states are synchronised at 24 and 36 hours.
