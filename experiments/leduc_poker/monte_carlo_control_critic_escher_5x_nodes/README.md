# Experiment 10: current-iteration Monte Carlo control critic

Experiment 10 tests whether TD bootstrapping and critic staleness limit the
Experiment 6 architecture. It replaces Experiment 6's bootstrapped transition
critic with direct supervision from the recursively unbiased sampled returns
already generated during traversal.

The new architecture runs seeds `0`, `1`, and `2` to the exact paired
Experiment 6 node budgets. Checksum-protected Experiment 6 results are included
in every performance comparison without rerunning the baseline.

## Frozen collection and fitting phases

Every outer iteration is divided into two phases.

### 1. Frozen collection

Before collection begins, the following remain fixed:

- both cumulative and instantaneous regret networks;
- all control-critic target snapshots;
- residual calibration and beta selection;
- adaptive sampling models;
- the prediction gate.

Both players complete all their traversals before either player's regret
network is fitted. This is stricter than Experiment 6, where player 0's regret
network was updated before player 1 collected its samples.

For every visited state and sampled action, recursive traversal returns a
sampled traverser value `G`. The target stored for the shared Q representation
is `G` for player 0 and `-G` for player 1, matching the existing convention that
the network represents player-0 utility and inference changes sign for player
1.

### 2. Independent fitting

Every trajectory is assigned to one of three folds. Its Monte Carlo targets are
written only to that fold, while its control predictions come only from the
other two folds. At the end of collection:

- each critic is fitted directly to `(history, action, G)` targets;
- one-step transition tuples are explicitly discarded;
- there is no TD continuation value, target-policy evaluation or bootstrapped
  target;
- all online fits complete before their frozen target snapshots are updated;
- regret, calibration and gate updates also occur only after frozen collection.

Critic networks and optimisers persist as warm starts, but their replay buffers
are cleared at the start of every iteration. Thus the fitted targets describe
the current frozen strategy rather than a mixture of historical strategies.

The three fold losses are completely independent once collection finishes, so
critic fitting is embarrassingly parallel. The reference runner executes them
sequentially for deterministic, like-for-like CPU measurements; the
architecture does not require sequential fitting and can distribute folds in a
large-game implementation.

## Why the estimator remains unbiased

The critic is still used only as a control variate. Experiment 10 retains
Experiment 6's estimator:

\[
\widetilde Q_t(a)=
\beta_t(a)\widehat Q_t(a)+
\frac{\mathbf 1\{A=a\}}{\xi_t(a)}
\left(G-\beta_t(a)\widehat Q_t(a)\right).
\]

The critic snapshot, beta and full-support sampling distribution are selected
before `G` is observed. Therefore

\[
\mathbb E[\widetilde Q_t(a)]=Q_t(a)
\]

regardless of Monte Carlo critic error. Removing TD bootstrapping can alter the
variance and generalisation of the control, but cannot bias expected regret.
Training remains model-free: it uses only sampled trajectories and their
returns, with no transition model or game-tree enumeration.

The standard sampled no-regret-to-Nash argument remains available in the
tabular/oracle limit under full-support sampling and sublinear approximation
error. Direct MC critic fitting is not itself required for convergence; it is a
variance-reduction mechanism whose imperfections are residual-corrected.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Paired targets: `4,700,205`, `4,701,540`, and `4,684,695` nodes.
- Evaluation: untrained policy, approximately 10,000 nodes, and every complete
  outer iteration.
- Exact evaluation nodes are excluded from `nodes_touched`.
- Experiment 6 network sizes, buffers, gradient steps and every non-critic
  setting are unchanged.
- Each run stops after the first complete outer iteration crossing its target.
- Only the new candidate is trained; Experiment 6 results are immutable inputs.

## Immutable Experiment 6 comparison

The packaged references are byte-for-byte copies from Batch job
`leduc-escher-arch-exp6-20260718-230108`, run directory
`unbiased_control_variate_escher_5x_nodes_20260718_220419`:

- `experiment6_checkpoint_curves.csv` — SHA-256
  `7f0ecfca091130565275fc27c775cdcd4e96b62eb122759209d9d4f17b0e65b5`;
- `experiment6_seed_summary.csv` — SHA-256
  `10a43adeb4f415f34e45f2498cd25d85977bb53e0da13300ed7618071635daf9`.

The runner verifies checksums, row counts, algorithm IDs and seeds before
training. The saved wall-clock curve is useful context but is not a
contemporaneous hardware benchmark.

## Diagnostics

In addition to all Experiment 6 metrics, each checkpoint records:

- current-iteration MC target count by fold;
- target mean, variance and maximum absolute magnitude;
- direct critic regression loss;
- minimum and maximum frozen critic target versions;
- agreement between MC-target accounting and estimator samples.

The target-scale charts are particularly important because importance-corrected
recursive returns can exceed terminal utility bounds. If performance regresses
while target variance or maximum magnitude grows, a robust but expectation-safe
critic loss would be a natural follow-up; the residual correction itself should
not be clipped.

## Run locally

```bash
python -m experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes.run
```

## Local smoke test

This verifies frozen-phase ordering, cross-fitting, player-value signs, direct
regression without TD, immutable references, diagnostics and plots. Its
exploitability comparison is not scientifically meaningful.

```bash
python -m experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes.run \
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
Experiment 10 retains the same Q gradient-step count but removes batched TD
continuation and next-strategy computation. Allow approximately **12 hours**
for the full three-seed experiment and set the single-Batch maximum to **1,440
minutes** (24 hours, `86400` seconds). The estimate is conservative because the
direct regression target is cheaper.

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
JOB_NAME="leduc-escher-arch-exp10-mc-critic-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

## GCP Batch smoke test

This is one Batch job for seed `0`; use its **360-minute** (`21600`-second)
timeout.

```bash
JOB_NAME="leduc-escher-arch-exp10-mc-critic-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes.run \
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
- Monte Carlo target variance/loss and target-scale charts;
- aggregate summary, provenance metadata, worker inputs/results and logs.
