# Experiment 28: information-set-stratified policy distillation

## Question

Experiment 27 showed that soft-target cross-entropy materially improves the
average-policy network, but does not recover the much stronger empirical
reservoir policy. Experiment 28 asks whether ordinary row minibatching
under-trains rare information sets and whether deliberately increasing their
minibatch representation closes more of that remaining gap.

This is an offline development experiment. It reuses the frozen 24- and
36-hour states from the Experiment 25 `averaged_critic_target` arm and does not
rerun UCV-ESCHER training.

## Frozen comparison

All three arms use the same `3 x 64` average-policy network, Adam learning rate,
batch size, 5,000 updates, iteration weighting and soft-target cross-entropy
loss:

1. **Empirical-mass control:** sample information set (I) with
   (q(I)=p(I)), where (p(I)) is its empirical row frequency.
2. **Square-root stratification:** sample with
   (q(I)\propto\sqrt{p(I)}).
3. **Uniform stratification:** sample information sets uniformly.

After selecting an information set, every arm samples one of its reservoir
rows uniformly and samples with replacement. Every loss contribution is
multiplied by (p(I)/q(I)). The expected training objective is therefore the
same empirical-reservoir objective in all arms; only the stochastic allocation
of updates across information sets changes. The empirical-mass arm is exactly
uniform row sampling in distribution under this common implementation.

The study uses the three independent Experiment 25 source trajectories and
three nested optimiser replicates per trajectory at both 24 and 36 hours. The
source trajectory—not the nine fitted networks—is the inferential unit. The
primary outcome is the 36-hour neural-minus-exact exploitability gap. Secondary
outcomes are exact exploitability, reach-weighted policy error, fitting time and
the effective sample fraction induced by the importance weights.

## Local smoke

From the repository root:

```bash
./gcp/run_information_set_stratified_distillation.sh smoke-local
```

The smoke creates tiny compatible Experiment 25 states, runs all three arms at
both checkpoints, reload-validates all six policies, aggregates every table and
renders all three charts.

## Google Cloud run

Use the same project, region, bucket and service account as Experiments 19--27.
The Experiment 25 run must still contain each averaged-target worker's
`worker_result.json` and its `time_24h` and `time_36h` training states.

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP25_RUN_ID="exp25-fact-20260906-003836"
export RUN_ID="exp28-strat-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_information_set_stratified_distillation.sh run
```

The cloud controller first runs an independent smoke, then starts one on-demand
`n2-standard-8` worker per source seed and finally aggregates the results. Each
worker downloads only its own two source states and has a 12-hour hard limit.
The controller runs remotely, so the initiating laptop may be disconnected.

Check or resume a run by restoring the original `RUN_ID` and environment:

```bash
./gcp/run_information_set_stratified_distillation.sh status
./gcp/run_information_set_stratified_distillation.sh resume
```

## Outputs

The `analysis/` directory contains:

- `aggregate_summary.json`: frozen contract and primary paired effects;
- `metric_summary.csv`: source-seed summaries for both checkpoints;
- `paired_effect_summary.csv` and `paired_effects_by_seed.csv`: stratified arms
  minus empirical-mass control;
- `sampling_diagnostics.csv`: proposal ranges, importance-ratio ranges and
  effective sample fractions;
- `information_set_errors.csv`: per-information-set L1 and KL diagnostics;
- `policy_inventory.csv`: checksums and paths for every reload-validated fitted
  policy;
- `sampling_arm_comparison.png`, `checkpoint_comparison.png` and
  `frozen_policy_gap_decomposition.png`.

To download only the complete analysis:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp -r "$BUCKET/$RUN_ID/analysis/*" "cloud_outputs/$RUN_ID/analysis/"
```

Do not interpret optimizer replicates as additional independent seeds. With
three source trajectories the smallest attainable two-sided sign-flip p-value
is 0.25, so effect magnitude, direction across all three trajectories and
diagnostic consistency matter more than threshold significance.
