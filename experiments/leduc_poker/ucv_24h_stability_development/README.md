# Experiment 23: 24-hour UCV stability development study

## Question and frozen design

This development experiment asks whether the evidence from Experiments 21 and
22 supports a faster, simpler and more stable UCV-ESCHER architecture. It runs
four paired arms:

1. **Original UCV**: adaptive beta, three cross-fitted critics, gated
   instantaneous predictor and constant `1e-3` learning rate.
2. **Fast core**: fixed `beta=1` and two cross-fitted critics, retaining
   calibrated residual-adaptive sampling and the gated predictor.
3. **Non-predictive fast core**: the fast core with the instantaneous predictor
   genuinely disabled and conservative DCFR+ accumulation retained.
4. **Stable non-predictive core**: the non-predictive core with cumulative-
   regret and average-policy gradient norms clipped at `5.0`; their learning
   rates remain `1e-3` to 15 million nodes, cosine-decay to `1e-4` by 45
   million nodes, and stay at `1e-4` thereafter. Critic and residual-
   calibration learning rates remain `1e-3`.

Each arm runs for the first completed iteration crossing 24 active training
hours over four fresh paired development seeds. Playable policies are saved at
the first completed iteration crossing every two-hour threshold and at the
first completed iteration crossing 15 million nodes. Snapshot serialisation is
excluded from the active-time clock. The 500-iteration setting is a failure
guard, not a stopping target.

The development seeds are generated before results are observed from the first
32 bits of `sha256("ucv-24h-stability-development-{index}")`. They do not
overlap Experiment 19/21 held-out labels or Experiment 22 development labels.
Four seeds are intended for effect estimation and architecture triage: the
minimum two-sided paired exact sign-flip p-value is `0.125`.

## Pre-specified interpretation

The primary development evidence is:

- mean exact exploitability across all 12--24-hour checkpoints;
- late-window adjacent-checkpoint RMSSD and worst deterioration;
- improvement from 12 to 24 hours, rather than a result-selected best point;
- exact exploitability and active time at the 15-million-node checkpoint;
- nodes processed by 24 hours and peak resident memory.

The aggregate analysis reports each seed and paired candidate-minus-original
effects. Lower is better for exploitability, volatility, time and memory;
higher is better for improvement and nodes processed. This is not confirmatory
evidence. Any architecture promoted after inspecting Experiment 23 requires a
fresh-seed confirmation against Original UCV with the design frozen.

The residual calibration network remains in every arm because it controls the
sampling proposal even when beta is fixed. The non-predictive arms remove the
instantaneous-regret model, not calibration. Online diagnostics retain beta,
prediction-gate, correction, calibration, critic-error and learning-rate
traces so the implementation and any performance change can be audited.

## Mandatory local smoke test

Use the repository's Python 3.9 environment from the repository root:

```bash
export SMOKE_OUTPUT="/tmp/exp23-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_ucv_24h_stability_development.sh smoke-local
```

The smoke runs all four real arms on seed `0`, forces the stable arm through
its annealing interval, verifies fixed beta and disabled predictors, saves and
reloads every playable checkpoint, performs exact OpenSpiel evaluation, and
renders every analysis chart. Smoke results have no scientific meaning.

## Fully remote GCP run

Reuse the project, region, bucket and Batch/controller service account already
configured for Experiments 19--22. Commit and push this experiment first, then:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp23-stab-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=16

./gcp/run_ucv_24h_stability_development.sh run
```

The command returns once the remote controller is accepted; the laptop may
then be closed. The controller runs a clean cloud smoke, submits production
only after it succeeds, and finally performs exact aggregation. Full
parallelism requires 128 regional N2 vCPUs. Budget approximately 400--440
`n2-standard-8` VM-hours including completed-iteration overshoot, bootstrap,
smoke and aggregation, with roughly 25--30 elapsed hours at full parallelism.
Every production worker has a 36-hour hard limit and no automatic task retry,
which caps accidental training cost and prevents silently replacing a failed
seed trajectory.

Operational commands, using the same exported values, are:

```bash
./gcp/run_ucv_24h_stability_development.sh status
./gcp/run_ucv_24h_stability_development.sh resume
./gcp/run_ucv_24h_stability_development.sh dry-run
```

## Download and reproduce analysis

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "cloud_outputs/$RUN_ID/"
```

After downloading the complete `workers/` tree, rerun analysis locally with:

```bash
python -m experiments.leduc_poker.ucv_24h_stability_development.run \
  aggregate --output-root "cloud_outputs/$RUN_ID"
```

Key outputs are `checkpoint_policy_metrics.csv`,
`development_metrics_by_seed.csv`, `paired_metrics_vs_original.csv`,
`exploitability_by_training_time.png`,
`exploitability_by_nodes_touched.png`, and
`late_window_performance_stability.png`. Every policy file is playable and has
a checksum, frozen configuration, commit, seed, completed iteration, active
time and node-count record.
