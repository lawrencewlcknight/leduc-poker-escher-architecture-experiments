# Experiment 24: selected UCV architecture at 36 hours

## Question and evidence status

Experiment 24 asks whether the simpler architecture selected after Experiment
23 retains its speed and stability through the 24--36-hour interval in which
Original UCV-ESCHER was volatile in Experiment 21. The candidate fixes
`beta=1`, uses two cross-fitted critics, disables the instantaneous predictor,
and retains residual calibration, residual-adaptive full-support sampling,
conservative DCFR+ accumulation and the original constant learning rate. It
does not use the unsuccessful learning-rate annealing and gradient-clipping
package from Experiment 23.

The run deliberately reuses Experiment 21 seeds `104729`, `130363`, `155921`,
`181081` and `205759`. This provides paired trajectories against immutable
Deep CFR and Original UCV results without rerunning them. Because Experiment
21 motivated the subsequent architectural work, this is post-selection
follow-up evidence, not a fresh held-out confirmation.

## Frozen execution contract

Five independent `n2-standard-8` workers each train the selected candidate for
36 active hours. A playable policy is saved at the first completed iteration
crossing every two-hour threshold and at the first completed iteration crossing
15 million nodes. Snapshot time is excluded from the active training clock.
The 800-iteration cap is a failure guard, not a stopping target. Workers have a
50-hour Batch limit, no automatic retry and resumable completed-artifact reuse.

The aggregate stage verifies the archived Experiment 21 checkpoint-metrics
checksum, seed set, checkpoint schedule and every final policy checksum. It
writes new Experiment 24 analysis files; it never modifies Experiment 21.

Pre-specified primary outcomes are mean exact exploitability over 24--36 hours
and exact exploitability at 36 hours. Secondary outcomes are 24--36-hour
improvement, adjacent-checkpoint RMSSD, worst deterioration, node throughput,
the 15-million-node candidate endpoint, and exact final same-seed head-to-head
value against both Experiment 21 algorithms. Training seed is the inferential
unit. With five seeds, the minimum two-sided exact sign-flip p-value is
`0.0625`, so effect sizes and intervals remain central.

## Mandatory local smoke test

From the repository root, with the sibling Deep CFR repository still in the
location used for Experiment 21:

```bash
export SMOKE_OUTPUT="/tmp/exp24-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_selected_ucv_36h_confirmation.sh smoke-local
```

Set `DEEP_CFR_REPO` only if that sibling repository is elsewhere. The smoke
runs tiny Deep CFR, Original UCV and selected-candidate trajectories; saves and
reloads every policy; verifies fixed beta, two critics, predictor removal and
the absence of annealing/clipping; performs exact evaluation and final
head-to-head analysis; and renders both combined charts. Smoke values have no
scientific meaning.

## Fully remote GCP run

Reuse the Experiment 21--23 project, region, bucket and service account. Commit
and push Experiment 24 before setting `REPO_REF`. The archived reference is
frozen as `exp21-36h-20260830-141641`.

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export EXP21_RUN_ID="exp21-36h-20260830-141641"
export RUN_ID="exp24-selected-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_selected_ucv_36h_confirmation.sh run
```

The local command returns after the remote controller is accepted. The laptop
may then be disconnected. The controller performs a clean cloud smoke, starts
production only after smoke succeeds, stages the immutable Experiment 21
archive for analysis, and uploads the combined results.

Five-way parallelism requires 40 regional N2 vCPUs. Training consumes 180
`n2-standard-8` VM-hours. Allow approximately 190--200 VM-hours including
completed-iteration overshoot, bootstrap, smoke and aggregation, and about
37--40 elapsed hours plus provisioning delay. Each training worker has an
independent 50-hour hard limit.

Operational commands are:

```bash
./gcp/run_selected_ucv_36h_confirmation.sh status
./gcp/run_selected_ucv_36h_confirmation.sh resume
./gcp/run_selected_ucv_36h_confirmation.sh dry-run
```

## Download and reproduce analysis

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "cloud_outputs/$RUN_ID/"
```

To recreate the analysis locally, the downloaded Experiment 21 directory must
also be present:

```bash
python -m experiments.leduc_poker.selected_ucv_36h_confirmation.run aggregate \
  --output-root "cloud_outputs/$RUN_ID" \
  --experiment-21-root \
    "cloud_outputs/leduc-escher-arch-exp21-36h-20260830-141641"
```

The principal outputs are `exploitability_by_training_time.png`,
`exploitability_by_nodes_touched.png`,
`combined_checkpoint_policy_metrics.csv`, `paired_candidate_comparisons.csv`,
`final_same_seed_head_to_head.csv`, and `candidate_node_endpoint_metrics.csv`.
