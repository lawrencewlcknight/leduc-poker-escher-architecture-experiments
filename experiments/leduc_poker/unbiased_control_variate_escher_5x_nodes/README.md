# Experiment 6: always-unbiased control-variate ESCHER

Experiment 6 trains a new **Unbiased Control-Variate ESCHER** architecture for
seeds `0`, `1`, and `2` to the exact paired ESCHER node endpoints from
Experiment 2. It reuses the immutable Experiment 2 ESCHER, VR-DeepDCFR+, and
VR-DeepPDCFR+ curves and plots all four algorithms together. The three
comparison algorithms are not retrained.

## Architectural hypothesis

For sampled action `A ~ xi`, rollout return `G`, cross-fitted action value
`Q_hat`, and a coefficient selected before observing `G`, the estimator is

```text
Q_tilde_beta(a) = beta(I,a) Q_hat(a)
                  + 1{A=a}/xi(a) [G - beta(I,a) Q_hat(a)].
```

Conditioned on the information available before the sample,

```text
E[Q_tilde_beta(a)]
  = beta Q_hat(a) + Q(a) - beta Q_hat(a)
  = Q(a).
```

The result holds for every finite predictable `beta`; unlike Experiment 3's
shrinkage coefficient, `beta` does not need to converge to one. The action
values are centred under the current strategy before entering the regret
accumulator. With full-support sampling, bounded importance ratios, sublinear
local regret, and sublinear neural approximation error, the standard
no-regret-to-Nash route therefore remains available in the tabular/oracle
limit.

This does not claim a finite-network convergence proof. Cross-fitting,
calibration, and clipping are approximation mechanisms; the experiment tests
whether they preserve the estimator's theoretical advantage in practice.

## Five coupled mechanisms

### 1. Three-fold cross-fitted persistent Q ensemble

Every trajectory is assigned deterministically to one replay fold. Its
transitions are written only to that fold's critic, while the trajectory's
control variate is the mean of the other two critics. Consequently, neither
critic used for a trajectory can ever train on that trajectory. All critics
are persistent and use frozen target snapshots during collection and TD
training. The configured baseline replay capacity is divided across folds, so
the ensemble does not triple total replay capacity.

### 2. Information-set-conditioned residual calibration

A persistent calibration network receives:

- the traverser's information-state representation;
- an action one-hot vector;
- log iteration;
- cross-fitted ensemble disagreement;
- player identity.

It predicts the held-out Q residual mean and variance. The target snapshot is
frozen for a complete collection phase and is trained only after returns have
been collected, so its predictions are fixed before each sampled return.

### 3. Variance-adaptive beta

For this Horvitz--Thompson estimator, conditional variance is minimised when
`beta Q_hat = E[G | I,a]`. The frozen calibration model predicts
`E[G-Q_hat | I,a]`; a ridge-stabilised ratio estimates the corresponding beta
and clips it to `[0, 2]`. Clipping may sacrifice variance optimality but cannot
introduce estimator bias because the importance-correction identity holds for
every predictable beta.

### 4. Residual-adaptive full-support sampling

At traverser nodes, legal actions are sampled in proportion to predicted
residual standard deviation, mixed with 20% uniform sampling:

```text
xi = 0.8 xi_variance + 0.2 Uniform(legal actions).
```

Thus every legal action has probability at least `0.2 / |A(I)|`. The recorded
sampling probability is used in the estimator's importance correction.

### 5. Prediction-gated regret updates

The instantaneous predictor is evaluated on newly collected targets before it
is reset or fitted. Its error is compared with the zero-prediction error. The
next iteration's gate is

```text
clip(1 - predictor_MSE / zero_predictor_MSE, 0, 1).
```

The gate interpolates conservative DCFR+ accumulation/policy updates and
predictive PDCFR+ updates. It is lagged by one iteration, so the gate that
controls a trajectory is chosen before that trajectory's targets are known.
This follows the motivation of stable-predictive CFR: optimism is used only
when the predictor has demonstrated held-out skill.

The estimator direction is motivated by predictive-baseline variance results
in [Davis, Schmid and Bowling (ICML
2020)](https://proceedings.mlr.press/v119/davis20a.html); conditional optimism
has precedent in [Farina et al. (ICML
2019)](https://proceedings.mlr.press/v97/farina19a.html).

## Matched Experiment 2 protocol

| Seed | Training-node target |
|---:|---:|
| 0 | 4,700,205 |
| 1 | 4,701,540 |
| 2 | 4,684,695 |

Each run stops after the first complete outer iteration crossing its target.
There is an untrained checkpoint, an early checkpoint after approximately
10,000 nodes, and a checkpoint after every outer iteration. Evaluation nodes
are excluded from `nodes_touched`.

The reference file is the checksum-validated 323-row Experiment 2 curve file
already bundled with Experiment 4. It was produced by Batch job
`leduc-escher-arch-exp2-20260717-105458` and has SHA-256
`0bd4ace4ea2611a34971aaf7c6ab676c05e39faa3bb3069113d641fac3b53b85`.
Every reused row is labelled `saved_experiment_2`; new rows are labelled
`experiment_6_new_run`.

## Full local run

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run
```

The three sequential runs are projected to take approximately 14 hours on the
same eight-vCPU machine class. The estimate is deliberately conservative
because the architecture trains three critics and a calibration network.

## Local smoke test

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run \
  --seeds 0 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --calibration-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --early-evaluation-nodes 10 \
  --output-root outputs/smoke_tests
```

## GCP Batch

Push the current commit so the Batch VM can clone it, then set the standard
repository variables:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

Full job with a 36-hour timeout:

```bash
JOB_NAME="leduc-escher-arch-exp6-unbiased-cv-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 129600 8000 32000 100
```

GCP smoke test:

```bash
JOB_NAME="leduc-escher-arch-exp6-unbiased-cv-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run \
    --seeds 0 \
    --target-nodes 50 \
    --traversals 4 \
    --max-iterations 2 \
    --advantage-train-steps 1 \
    --policy-train-steps 1 \
    --q-train-steps 1 \
    --calibration-train-steps 1 \
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

- `candidate_checkpoint_curves.csv` and `candidate_seed_summary.csv`;
- `combined_checkpoint_curves.csv` and `combined_seed_summary.csv`;
- `combined_exploitability_by_nodes.png`, the primary four-algorithm chart;
- `combined_final_exploitability.png`;
- `paired_differences.csv` against every Experiment 2 algorithm;
- estimator, beta, disagreement, sampling-floor, fold-size, calibration, and
  prediction-gate diagnostics in every candidate checkpoint row;
- metadata, aggregate summary, partial recovery results, worker inputs, logs,
  results, and failure records.
