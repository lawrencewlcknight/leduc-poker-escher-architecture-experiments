# Experiment 26: UCV direct-advantage replay at 36 hours

## Question and intervention

This architecture-development study asks whether the long-horizon volatility
seen in UCV-ESCHER is caused partly by recursively fitting each cumulative-
regret network against its preceding approximation. It starts from the
non-predictive fast core selected in Experiment 23: fixed `beta=1`, two
cross-fitted critics, no instantaneous predictor, residual calibration and
residual-adaptive full-support sampling.

The new arm retains the UCV estimator and its single-action external-sampling
traversal, but replaces the cumulative-regret learner as follows:

1. Store each utility-normalised instantaneous UCV advantage observation in a
   persistent uniform reservoir for the relevant player.
2. Fit the player-specific advantage network directly against those observations
   using the linearly weighted objective `2t/T`, a training-scale-normalised
   form of Deep CFR's iteration weighting.
3. Apply regret matching to the network's learned weighted advantages.

The previous fitted advantage network is never part of the next supervised
target. Warm-starting the network parameters between fits remains enabled; that
is optimisation state, not target bootstrapping. The traversal still samples
one action and does not enumerate counterfactual branches, so the variant
remains model-free in the same operational sense as UCV-ESCHER.

## Frozen design

- one new candidate, `ucv_advantage_replay`;
- five independent `n2-standard-8` workers;
- the same seeds as Experiment 21: `104729`, `130363`, `155921`, `181081`,
  `205759`;
- 36 active training hours per worker;
- playable snapshots at the first completed outer iteration crossing every
  two-hour threshold from 2 through 36 hours;
- one additional playable snapshot at the first completed iteration crossing
  15 million training nodes;
- an 800-iteration safety cap and a 50-hour Google Batch hard limit.

Aggregation validates and imports the immutable Experiment 21 archive
`exp21-36h-20260830-141641`, including the SHA-256 of its checkpoint metric
table. It then creates joined Deep CFR, Original UCV and UCV advantage-replay
time and node trajectories, exact 36-hour head-to-head values, and paired
late-window summaries. Experiment 21 files are never changed.

The joined comparison is efficient and paired, but it is development evidence,
not a fresh held-out confirmation. The difference from Original UCV cannot be
attributed to replay alone because the new arm also incorporates the
Experiment 23 fast-core choices. A same-seed comparison with the selected core
from Experiment 24 is the cleaner replay ablation if that archive is available.

## Mandatory local smoke

From the architecture repository root, with the Deep CFR repository at its
usual sibling location:

```bash
export SMOKE_OUTPUT="/tmp/exp26-replay-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_ucv_advantage_replay_36h.sh smoke-local
```

The smoke runs tiny Deep CFR, Original UCV and advantage-replay trajectories;
saves and reloads all smoke policies; checks the persistent replay and frozen
configuration contract; performs exact aggregation and head-to-head analysis;
and renders both joined charts. Smoke values are plumbing checks only.

## Full cloud run

Reuse the Experiment 21/24 service account and result bucket. The architecture
commit must be pushed before submission.

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export EXP21_RUN_ID="exp21-36h-20260830-141641"
export RUN_ID="exp26-replay-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_ucv_advantage_replay_36h.sh run
```

The local command submits a remote controller and returns. The controller runs
the cloud smoke, five production workers and aggregation, so the laptop may be
closed after submission. Full parallelism requires 40 available regional N2
vCPUs. Expect approximately 37--40 elapsed hours plus provisioning and about
190--200 N2 VM-hours including bootstrap, smoke, overshoot and aggregation.

```bash
./gcp/run_ucv_advantage_replay_36h.sh status
./gcp/run_ucv_advantage_replay_36h.sh resume
./gcp/run_ucv_advantage_replay_36h.sh dry-run
```

Completed outputs are stored under `$BUCKET/$RUN_ID/`. Download them from the
repository root with:

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "cloud_outputs/$RUN_ID/"
```

Key outputs are:

- `analysis/exploitability_by_training_time.png`;
- `analysis/exploitability_by_nodes_touched.png`;
- `analysis/combined_checkpoint_policy_metrics.csv`;
- `analysis/late_window_metrics_by_seed.csv`;
- `analysis/paired_candidate_comparisons.csv`;
- `analysis/final_same_seed_head_to_head.csv`;
- `analysis/training_checkpoint_curves.csv`, including replay occupancy, target
  scale, iteration-weight and recursive-target audit fields.
