# Experiment 25: residual-regret by averaged-critic-target factorial

## Research question

Experiment 25 asks whether either of two causally motivated interventions makes
the Experiment 23 selected UCV-ESCHER architecture less volatile and more
convergent over long training runs:

1. replacing only the cumulative-regret MLPs with deeper residual-LayerNorm
   networks; and
2. replacing each critic's one-fit hard target with the parameter-wise mean of
   its four most recently completed online fits.

The control is the selected non-predictive fast core: fixed `beta=1`, two
cross-fitted critics, no instantaneous predictor, residual calibration,
residual-adaptive full-support sampling, constant learning rate and no gradient
clipping. The experiment does not alter the estimator or its sampling support.

## Frozen factorial contract

| Arm | Cumulative-regret network | Frozen critic target |
|---|---|---|
| `control` | original 3x64 MLP | latest completed fit |
| `residual_regret` | four two-layer width-64 pre-LayerNorm residual blocks | latest completed fit |
| `averaged_critic_target` | original 3x64 MLP | mean of four completed fits |
| `residual_regret_averaged_target` | residual-LayerNorm | mean of four completed fits |

All four arms use the same three fresh paired development seeds. Each worker
runs for 36 active hours, saving a playable average-policy snapshot at the first
completed iteration crossing every two-hour threshold and at the first
completed iteration crossing 15 million training nodes. Full continuation
states are written at 24 and 36 hours. A continuation state contains models,
target histories, optimisers, replay buffers, exact-average accumulators,
diagnostics, counters and RNG state.

The four-arm run consists of 12 independent `n2-standard-8` workers. At full
parallelism it requires 96 regional N2 vCPUs, approximately 432 active training
VM-hours, and roughly 450--480 N2 VM-hours after bootstrap, completed-iteration
overshoot, smoke and aggregation. Allow about 40--44 elapsed hours plus cloud
provisioning. Each training task has a 54-hour per-attempt hard limit and no
automatic retry.

## Mandatory local smoke test

From the repository root:

```bash
export SMOKE_OUTPUT="/tmp/exp25-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_ucv_residual_target_factorial.sh smoke-local
```

The smoke runs all four arms, saves and reloads every playable policy, creates
both continuation checkpoints, restores each final continuation checkpoint into
a new solver, completes another optimisation iteration, performs exact policy
aggregation and renders every chart. Smoke values have no scientific meaning.

## Fully remote GCP run

Reuse the project, region, bucket and service account configured for Experiments
19--24. Commit and push Experiment 25 before setting `REPO_REF`.

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp25-fact-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=12

./gcp/run_ucv_residual_target_factorial.sh run
```

The local command returns once the remote controller has been accepted. The
laptop may then be disconnected. The controller runs a clean cloud smoke before
submitting production, waits for all workers, aggregates exact policies, and
uploads the results.

Production diagnostics are merged with bounded memory: the aggregator streams
the large per-worker critic and exact-oracle CSVs directly into their combined
outputs rather than materialising all rows as Python dictionaries. Aggregation
memory therefore remains independent of the accumulated diagnostic row count.

At each 24- and 36-hour continuation checkpoint the worker immediately uploads
its partial task directory. If a task subsequently fails, retain the original
`RUN_ID` and run:

```bash
./gcp/run_ucv_residual_target_factorial.sh resume
```

The replacement task downloads the partial directory and continues from the
latest compatible full state. A failure before 24 hours necessarily restarts
that task. A changed commit or configuration is deliberately rejected rather
than silently resumed.

Operational commands are:

```bash
./gcp/run_ucv_residual_target_factorial.sh status
./gcp/run_ucv_residual_target_factorial.sh resume
./gcp/run_ucv_residual_target_factorial.sh smoke-cloud
./gcp/run_ucv_residual_target_factorial.sh dry-run
```

## Download and reproduce analysis

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "cloud_outputs/$RUN_ID/"

python -m experiments.leduc_poker.ucv_residual_target_factorial.run aggregate \
  --output-root "cloud_outputs/$RUN_ID"
```

## Analysis and decision rule

Primary outcomes are mean exact exploitability over hours 24--36 and exact
exploitability at 36 hours. Secondary outcomes are late-window slope,
adjacent-checkpoint RMSSD, worst two-hour rebound, 15-million-node
exploitability, time to 15 million nodes, final nodes and memory.

The analysis reports paired arm-versus-control effects and the two factorial
main effects plus their interaction. Negative effects are favourable for
exploitability, slope, RMSSD, deterioration and time; positive effects are
favourable for improvement and final nodes. With only three development seeds,
effect sizes, consistency and diagnostic agreement are more informative than
hypothesis-test thresholds.

The solver also enumerates the exact reach- and iteration-weighted tabular
average strategy. `average_policy_diagnostics.png` compares its exploitability
with the learned average-policy output and exposes the distillation gap. Exact
critic-oracle errors, target-update magnitude, residual calibration and local
regret diagnostics are retained centrally. This permits a rebound to be
attributed more carefully to underlying regret learning, critic movement or
average-policy approximation.

The principal outputs are:

- `exploitability_by_training_time.png`;
- `exploitability_by_nodes_touched.png`;
- `late_window_performance_stability.png`;
- `average_policy_diagnostics.png`;
- `checkpoint_policy_metrics.csv`;
- `development_metrics_by_seed.csv`;
- `factorial_effect_summary.csv`;
- `paired_metrics_vs_control.csv`;
- `exact_q_oracle_diagnostics.csv`; and
- `training_state_inventory.csv`.

This remains development evidence. Any promoted architecture requires a new
fresh-seed confirmatory comparison.
