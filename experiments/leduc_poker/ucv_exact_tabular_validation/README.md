# Experiment 20: exact tabular validation of the UCV estimator

This is a single three-seed experiment that validates the implemented
UCV-ESCHER estimator directly in tabular Leduc. Seeds `0`, `1`, and `2` run
strictly sequentially in isolated subprocesses. The experiment trains only the
full UCV learner; every estimator comparison is evaluated counterfactually from
the same frozen learned state.

This experiment validates the implementation of the conditional-moment result.
It does not replace or constitute a proof of the theorem.

## Frozen protocol

Each seed trains to the first completed outer iteration crossing 15 million
training nodes. Exact diagnostics run at the first completed iteration crossing:

| Checkpoint | Nodes |
|---|---:|
| Early | 1,500,000 |
| Middle | 7,500,000 |
| Late | 15,000,000 |

At each checkpoint, the current regret networks, three frozen-target Q critics,
residual-calibration target network, prediction gates, iteration and target
versions are frozen. The diagnostic enumerates every reachable Leduc
information-set/action pair separately for each cross-fitting fold.

The five counterfactual estimators are:

| Variant | Frozen change |
|---|---|
| `full_adaptive_ucv` | Implemented adaptive beta and adaptive full-support sampling |
| `fixed_beta_one` | Beta fixed to one; adaptive sampling retained |
| `prediction_gate_zero` | Instantaneous-regret prediction gate disabled |
| `residual_calibration_disabled` | Calibration disabled, beta fixed to one and uniform full-support sampling |
| `baseline_free` | Zero control value with the same adaptive sampling law as full UCV |

The exact oracle invokes the production `control_variate_advantage`,
`variance_optimal_beta`, and `residual_adaptive_sampling_policy` functions. It
computes exact conditional action-value and advantage means, biases, variances
and mean-squared errors by dynamic programming over the Leduc tree. It also:

- checks conditional bias against the declared numerical tolerance `1e-9`;
- verifies that all learned models, gates, folds, target versions and RNG states
  are unchanged by evaluation;
- audits that Q prediction, calibration, beta and sampling are selected before
  the current sampled target;
- audits that the one-iteration-lagged prediction gate is fixed before data
  collection and updated afterwards;
- saves replayable diagnostic network snapshots with SHA-256 digests.

## Experiment 20 mandatory local smoke test

Run this before submitting the production job:

```bash
python -m experiments.leduc_poker.ucv_exact_tabular_validation.run --smoke \
  --output-root outputs/smoke_tests
```

The smoke test uses development seed `99991`, tiny training settings and scaled
30/150/300-node checkpoints. Its estimator results have no scientific meaning.
It nevertheless exercises all three checkpoints, all five variants, every
cross-fitting fold, exact aggregation, snapshot creation, invariant checks and
plots. On the reference laptop it completed in approximately two minutes.

A successful smoke run has both of the following values in
`aggregate_summary.json`:

```json
{
  "all_conditional_unbiasedness_checks_pass": true,
  "predictability_audit_status": "pass"
}
```

## Experiment 20 production run locally

The production command needs no seed or checkpoint arguments because they are
part of the frozen configuration:

```bash
python -m experiments.leduc_poker.ucv_exact_tabular_validation.run \
  --output-root outputs/ucv_exact_tabular_validation
```

The runner starts seed `0`, waits for it to finish, then starts seed `1`, and
finally seed `2`. It never trains two seeds concurrently.

## Experiment 20 production run as one GCP Batch job

Push the tested commit, then set the same Batch environment variables used by
the other architecture experiments. This single-job launcher needs no
controller-specific IAM roles beyond the existing Batch, logging, and bucket
permissions:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

Submit one standard eight-vCPU VM with a 48-hour safety timeout:

```bash
JOB_NAME="leduc-ucv-exp20-tabular-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_exact_tabular_validation.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 172800 8000 32000 100
```

The measured UCV training time is 11.22 hours per 15-million-node seed, or
33.65 hours for three sequential seeds. Allow approximately 36 hours including
exact diagnostics and aggregation. The 48-hour Batch duration is a safety cap,
not the expected runtime.

The job is fully remote after submission. Closing the laptop does not affect
it. Monitor it with:

```bash
gcloud batch jobs describe "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --location "$REGION"
```

The Batch cleanup trap uploads outputs on both success and failure:

```text
$BUCKET/$JOB_NAME/
```

## Experiment 20 optional GCP smoke test

```bash
JOB_NAME="leduc-ucv-exp20-tabular-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_exact_tabular_validation.run --smoke \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

## Outputs

The top-level run directory contains:

- `aggregate_summary.json` and `summary.json`;
- `experiment_metadata.json` with the complete frozen contract;
- `predictability_audit.json` with source hashes and ordering checks;
- `estimator_diagnostics.csv`, retaining seed, checkpoint, fold,
  information-set and action rows;
- `checkpoint_summary.csv` with exact bias, variance and MSE summaries;
- `seed_summary.csv` and `training_checkpoint_curves.csv`;
- `maximum_conditional_bias.png`;
- `exact_estimator_variance.png`;
- `worker_inputs/`, `worker_logs/`, and `worker_results/` for provenance and
  partial failure diagnosis.

Each seed directory additionally contains `early.pt`, `middle.pt`, and
`late.pt` under `snapshots/`, plus their sizes and SHA-256 hashes in the
checkpoint summary.
