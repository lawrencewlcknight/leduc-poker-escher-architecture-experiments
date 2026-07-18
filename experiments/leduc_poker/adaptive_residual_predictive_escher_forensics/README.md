# Experiment 5: adaptive-ESCHER forensic diagnostics

Experiment 5 is a Leduc-only diagnostic and mechanism-ablation experiment. Its
purpose is to locate the bottleneck in Experiment 3 before designing another
architecture:

```text
regret estimator -> accumulated advantages -> current strategy
                                      |
                                      v
                         average-policy distillation
```

Every arm uses the Experiment 3 learning configuration and paired Experiment 1
node targets. Exact diagnostics enumerate Leduc only during evaluation and are
excluded from `nodes_touched`; training remains trajectory-sampled and
model-free.

## Six one-factor-at-a-time arms

| Variant | Difference from the Experiment 3 control |
|---|---|
| `scheduled_predictive_persistent` | None; forensic control |
| `lambda_one` | Lambda fixed to 1: unbiased residual correction |
| `lambda_zero` | Lambda fixed to 0: relative-Q estimate only |
| `residual_only_lambda` | Past-residual adaptation with no schedule floor |
| `nonpredictive_accumulator` | Instantaneous predictor disabled |
| `reinitialized_q` | Upstream reinitialised Q learner instead of persistent Q |

This is intentionally not a factorial configuration sweep. Each arm changes
one architectural mechanism, so its difference from the control has a clear
interpretation.

The experiment runs seeds `0`, `1`, and `2` to their paired Experiment 1 ESCHER
node totals: 942,635, 939,834, and 962,274 nodes.

## Exact diagnostic contract

At the untrained checkpoint, approximately 10,000 nodes, and every completed
outer iteration, the experiment records:

1. **Current predictive strategy exploitability.** Exact exploitability of
   cumulative plus instantaneous predictive regret matching.
2. **Current cumulative-only strategy exploitability.** The same cumulative
   advantage networks evaluated without the instantaneous predictor.
3. **Exact weighted-average exploitability.** A tabular average updated with
   exact own-policy reach and iteration weight `iteration**gamma`.
4. **Neural average-policy exploitability.** The existing learned average
   policy used by Experiments 1--4.
5. **Exact Q error.** Every legal Q output is compared with the exact
   continuation value under the checkpoint's current predictive policy.
6. **Estimator moments.** The recursive residual estimator's exact conditional
   mean, bias, variance, and MSE are enumerated and grouped by player,
   information set, and action while the Q snapshot and lambda controller are
   frozen.
7. **Predictor error.** Instantaneous-advantage predictor MSE on the newly
   collected samples both before and after fitting. The pre-update error is
   paired by seed and checkpoint with the exact current-strategy performance
   difference between the predictive control and the arm that actually
   disables predictive updates.

The detailed Q and estimator CSV files retain information-set/action rows; the
checkpoint CSV contains reach-weighted aggregate metrics.

### Interpretation

- exact average better than neural average: average-policy distillation is the
  bottleneck;
- current strategies better than both averages: averaging or weighting is the
  bottleneck;
- high Q-oracle error with high estimator bias: critic approximation is the
  bottleneck;
- low bias but high variance in `lambda_one`: residual sampling variance is the
  bottleneck;
- low predictor error and positive predictive improvement: prediction helps;
- high predictor error and negative predictive improvement: prediction should
  be gated or replaced with a stable optimistic update.

## Run locally

Full six-arm, three-seed experiment:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run
```

Run a selected subset with, for example:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run \
  --variants scheduled_predictive_persistent,lambda_one,nonpredictive_accumulator
```

## Local smoke test

The smoke test executes all six architectural branches for one seed with tiny
buffers and training counts. Its performance results have no scientific
meaning.

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run \
  --seeds 0 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --early-evaluation-nodes 10 \
  --output-root outputs/smoke_tests
```

## GCP Batch

Set the usual repository Batch variables:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

The full job is expected to take approximately 12 hours sequentially. This
command provides a 24-hour timeout:

```bash
JOB_NAME="leduc-escher-arch-exp5-forensics-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

GCP smoke test:

```bash
JOB_NAME="leduc-escher-arch-exp5-forensics-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run \
    --seeds 0 \
    --target-nodes 50 \
    --traversals 4 \
    --max-iterations 2 \
    --advantage-train-steps 1 \
    --policy-train-steps 1 \
    --q-train-steps 1 \
    --batch-size 2 \
    --buffer-size 128 \
    --early-evaluation-nodes 10 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

Monitor and retrieve either job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

- `checkpoint_curves.csv` and `seed_summary.csv`;
- `strategy_diagnostics.csv`;
- `q_oracle_diagnostics.csv`;
- `estimator_diagnostics.csv`;
- `predictor_ablation_diagnostics.csv`;
- `ablation_exploitability_by_nodes.png`;
- `control_strategy_decomposition_by_nodes.png`;
- `q_oracle_error_by_nodes.png`;
- `estimator_bias_variance_by_nodes.png`;
- `predictor_error_vs_strategy_improvement.png`;
- `final_ablation_exploitability.png`;
- aggregate summary, metadata, failures, worker inputs, results, and logs.
