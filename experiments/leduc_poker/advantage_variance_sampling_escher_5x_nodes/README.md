# Experiment 11: centred-advantage variance sampling

Experiment 11 tests a single architectural change to the Experiment 6
always-unbiased control-variate ESCHER: traverser actions are sampled to reduce
the predicted Euclidean variance of the **centred advantage vector** consumed
by the regret learner. Experiment 6 instead samples from the predicted standard
deviation of the uncentred Q residual.

Seeds `0`, `1`, and `2` run to the exact paired Experiment 6 node budgets.
Checksum-protected Experiment 6 curves and summaries are added to every
comparison without retraining the baseline.

## Sampling objective

Experiment 6 forms the sampled all-action estimate

\[
\widetilde Q(a)=
\beta(a)\widehat Q(a)+
\frac{\mathbf 1\{A=a\}}{\xi(a)}
\left(G-\beta(a)\widehat Q(a)\right)
\]

and centres it with the current strategy:

\[
\widetilde A=(I-\mathbf 1\pi^\top)\widetilde Q.
\]

The stochastic correction caused by sampling action `a` therefore points in
the direction

\[
(I-\mathbf 1\pi^\top)e_a,
\]

whose squared Euclidean norm over `K` legal actions is

\[
1-2\pi(a)+K\pi(a)^2.
\]

The calibration model predicts the mean and variance of
`R = G - Q_hat(a)`. Because the actual control is `beta(a) Q_hat(a)`, the
required second moment is

\[
\widehat {\mathbb E}[(G-\beta(a)\widehat Q(a))^2]
=\widehat{\operatorname{Var}}(R)
+[\widehat{\mathbb E}(R)+(1-\beta(a))\widehat Q(a)]^2.
\]

Ignoring a support floor, minimising the conditional trace of the sampled
advantage correction's covariance gives

\[
\xi^*(a\mid I)\propto
\sqrt{\widehat {\mathbb E}[(G-\beta(a)\widehat Q(a))^2]}
\left\|(I-\mathbf 1\pi^\top)e_a\right\|_2.
\]

The implementation mixes this proposal with the unchanged Experiment 6
uniform mass `epsilon = 0.2`:

\[
\xi(a)=(1-\epsilon)\xi^*(a)+\epsilon/K.
\]

This ensures every legal action has probability at least `epsilon/K` and
controls importance weights even when the calibration model is inaccurate.

## Why unbiasedness and model-free scaling are retained

The proposal depends only on the information set, current strategy, frozen
cross-fitted Q predictions, beta and frozen calibration predictions. All are
available before the trajectory return `G` is observed. The estimator retains
the exact `1 / xi(a)` correction, so for every full-support predictable
proposal:

\[
\mathbb E[\widetilde Q(a)\mid I]=Q(a),
\qquad
\mathbb E[\widetilde A\mid I]
=(I-\mathbf 1\pi^\top)Q.
\]

The new sampler can change variance but not expected regret. Training remains
model-free because it uses only sampled trajectories, return observations and
learned information-set functions; it never queries a transition model or
enumerates the game tree. The standard sampled no-regret-to-Nash route remains
available in the tabular/oracle limit under full-support sampling and
sublinear regret/approximation error.

The computation is local to each sampled information set and consists of
small action-vector operations. Independent traversals, critic folds and
player updates retain the same parallelisation opportunities as Experiment 6.

## Controlled comparison

Every Experiment 6 setting is retained, including:

- three persistent cross-fitted critics;
- adaptive always-unbiased control-variate beta;
- residual calibration network;
- prediction-gated PDCFR+/DCFR+ regret accumulator;
- correctly weighted average-strategy learning;
- network sizes, buffers, training steps and evaluation schedule.

Only `_traverser_sampling_policy` is overridden. A regression test verifies
that the base Experiment 6 solver still returns its original
residual-standard-deviation proposal.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Paired targets: `4,700,205`, `4,701,540`, and `4,684,695` nodes.
- Evaluation: untrained policy, approximately 10,000 nodes, and every complete
  outer iteration.
- Exact evaluation nodes are excluded from `nodes_touched`.
- Each seed stops after the first complete outer iteration crossing its paired
  target.
- Only Experiment 11 is trained; Experiment 6 is an immutable reference.

## Immutable Experiment 6 comparison

The packaged references are byte-for-byte copies from Batch job
`leduc-escher-arch-exp6-20260718-230108`, run directory
`unbiased_control_variate_escher_5x_nodes_20260718_220419`:

- `experiment6_checkpoint_curves.csv` — SHA-256
  `7f0ecfca091130565275fc27c775cdcd4e96b62eb122759209d9d4f17b0e65b5`;
- `experiment6_seed_summary.csv` — SHA-256
  `10a43adeb4f415f34e45f2498cd25d85977bb53e0da13300ed7618071635daf9`.

The runner validates checksums, row counts, algorithm IDs and seeds before
training. Saved wall-clock results are useful context, but they are not a
simultaneous hardware benchmark.

## Diagnostics and interpretation

Each checkpoint retains all Experiment 6 diagnostics and adds:

- predicted second moment of the actual control residual;
- mean, minimum and maximum centering influence norm;
- minimum/maximum action probability and mean proposal entropy;
- the predicted centred-advantage variance proxy under Experiment 11;
- the same proxy evaluated counterfactually under Experiment 6 sampling;
- their ratio.

A ratio below one means the new proposal is predicted to reduce the conditional
second moment under the current calibration model. It is a mechanism check,
not proof of realised performance: calibration error, temporal correlations
and neural regret approximation can still dominate. The decisive outcomes are
paired final exploitability and exploitability-normalised AUC by nodes.

## Run locally

```bash
python -m experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes.run
```

## Local smoke test

This validates the exact centering geometry, support floor, importance
correction, baseline-reference checksums, diagnostics and plots. Its
exploitability values are not scientifically meaningful.

```bash
python -m experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes.run \
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

## Runtime estimate

Experiment 6 required approximately `10.59` hours for three sequential seeds.
Experiment 11 adds only small vector arithmetic at traverser information sets,
so allow approximately **12 hours** and set the single-Batch maximum to **1,440
minutes** (24 hours, `86400` seconds).

## GCP Batch environment

Push the current commit so the Batch VM can clone it, then define:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

## Full single GCP Batch job

```bash
JOB_NAME="leduc-escher-arch-exp11-adv-sampling-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

## GCP Batch smoke test

This is one Batch job for seed `0`; use its **360-minute** (`21600`-second)
timeout.

```bash
JOB_NAME="leduc-escher-arch-exp11-adv-sampling-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes.run \
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

Monitor and retrieve the job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

- candidate and combined checkpoint/seed CSV files;
- paired differences against Experiment 6;
- combined exploitability by nodes and wall-clock time;
- combined final exploitability;
- predicted variance-proxy and sampling-distribution diagnostic charts;
- aggregate summary, provenance metadata, worker inputs/results and logs.
