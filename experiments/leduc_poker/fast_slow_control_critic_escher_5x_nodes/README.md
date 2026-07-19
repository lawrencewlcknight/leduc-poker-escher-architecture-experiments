# Experiment 9: fast/slow cross-fitted control critic

Experiment 9 tests whether critic staleness limits the Experiment 6 architecture.
It trains the new architecture for seeds `0`, `1`, and `2` to the exact paired
Experiment 6 node budgets and compares it with checksum-protected Experiment 6
results without retraining Experiment 6.

## Architecture

For every information set and action, the control critic is

\[
C_t(I,a)=\rho_t(I,a)C_{\mathrm{fast},t}(I,a)
+[1-\rho_t(I,a)]C_{\mathrm{slow},t}(I,a).
\]

Each of Experiment 6's three trajectory folds now contains two persistent,
frozen-target critics:

- **Fast critic.** Its network and optimiser persist, but its replay is cleared
  before every outer iteration. It is trained only from the current iteration's
  transitions and tracks the current strategy.
- **Slow critic.** Its network and optimiser persist and it is trained from a
  uniform reservoir over the lifetime transition stream. It retains long-run
  coverage rather than merely retaining the most recent full buffer.

A trajectory is assigned to exactly one fold. Both fast and slow transitions
are written only to that fold; both predictions come only from the other two
folds. The fast and slow predictions used to label the controller therefore did
not train on that trajectory.

### Held-out rho controller

The controller is an information-set/action-conditioned neural network. Its
inputs are:

- the information-state representation and action;
- iteration and traversing player;
- held-out fast and slow predictions;
- disagreement within each held-out critic ensemble;
- the absolute fast/slow prediction gap.

Its sigmoid output gives rho in `[0, 1]`. The final layer is zero-initialised,
so the initial mixture is exactly `rho=0.5`. During an iteration the controller
target is frozen. After a rollout return is observed, the pre-return features,
out-of-fold fast/slow predictions and sampled return are added to a recent
circular replay. At the end of the iteration the controller minimises the
squared error of the mixture and only then updates its frozen target. Thus rho
for a return is always selected before that return is observed.

The implementation caches each pre-return decision through the recursive
rollout, avoiding a second set of critic/controller forward passes.

## Unbiasedness and convergence route

Experiment 9 retains Experiment 6's estimator. With `C_t` denoting the mixed
control value,

\[
\widetilde Q_t(a)=
\beta_t(a)C_t(a)+
\frac{\mathbf 1\{A=a\}}{\xi_t(a)}
\left(G-\beta_t(a)C_t(a)\right).
\]

Conditional on the pre-return information, `rho`, `beta`, both critic
snapshots, and the full-support sampling policy are fixed. Therefore

\[
\mathbb E[\widetilde Q_t(a)]=Q_t(a)
\]

for every predictable rho and beta, regardless of either critic's error. The
mixture changes variance, not expectation. Experiment 6's clipped/discounted
regret accumulator, prediction gate, residual calibration, adaptive
full-support sampling and average-policy learner are otherwise unchanged. The
usual sampled no-regret-to-Nash route is therefore retained in the tabular/oracle
limit, subject to the same neural approximation assumptions as Experiment 6.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Targets: `4,700,205`, `4,701,540`, and `4,684,695` nodes respectively.
- Evaluation: untrained policy, approximately 10,000 nodes, and every complete
  outer iteration.
- Evaluation nodes are excluded from `nodes_touched`.
- All Experiment 6 settings remain unchanged except the control-critic
  architecture and its explicitly documented training work.
- Slow Q training retains Experiment 6's `10,000` steps per fold; fast Q uses
  `5,000` steps per fold and the controller uses `2,000` steps per iteration.
- The run stops after the first complete outer iteration crossing its paired
  node target.

## Immutable Experiment 6 comparison

The packaged reference files were copied without transformation from Batch job
`leduc-escher-arch-exp6-20260718-230108`, run directory
`unbiased_control_variate_escher_5x_nodes_20260718_220419`:

- `experiment6_checkpoint_curves.csv` — SHA-256
  `7f0ecfca091130565275fc27c775cdcd4e96b62eb122759209d9d4f17b0e65b5`;
- `experiment6_seed_summary.csv` — SHA-256
  `10a43adeb4f415f34e45f2498cd25d85977bb53e0da13300ed7618071635daf9`.

The runner verifies both hashes, row counts, algorithm IDs and seeds before
training. Because Experiment 6 is not rerun, its wall-clock curve is useful
context but is not a contemporaneous hardware benchmark.

## Diagnostics

In addition to the Experiment 6 metrics, every checkpoint records:

- mean/minimum/maximum rho;
- fast and slow held-out ensemble disagreement;
- fast/slow prediction gap;
- sampled-return MSE for fast, slow and controlled-mixture predictions;
- fast, slow and controller losses and frozen-target versions;
- recent fast-fold replay sizes;
- slow-fold reservoir sizes and lifetime transitions seen;
- controller replay size.

These distinguish three outcomes: a rho that moves toward fast as the strategy
changes supports the staleness hypothesis; a rho near slow rejects it; and a
mixture MSE no better than both components indicates controller failure even if
one critic is useful.

## Run locally

```bash
python -m experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.run
```

## Local smoke test

This one-seed test checks reference provenance, both replay timescales,
cross-fitting, the frozen rho controller, unbiasedness diagnostics and every
output chart. Its exploitability comparison is intentionally meaningless.

```bash
python -m experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.run \
  --seeds 0 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --fast-q-train-steps 1 \
  --calibration-train-steps 1 \
  --rho-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --fast-q-buffer-size 128 \
  --rho-buffer-size 128 \
  --early-evaluation-nodes 10 \
  --output-root outputs/smoke_tests
```

## Runtime estimate

Experiment 6 required approximately `10.59` hours for its three sequential
seeds. Experiment 9 performs 1.5 times as many Q optimisation steps, evaluates
two critic timescales, and trains the rho controller. Allow approximately
**24 hours** for the full three-seed experiment and set the single-Batch maximum
to **2,880 minutes** (48 hours, `172800` seconds).

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
JOB_NAME="leduc-escher-arch-exp9-fast-slow-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 172800 8000 32000 100
```

## GCP Batch smoke test

This is one Batch job for seed `0`; use its **360-minute** (`21600`-second)
timeout.

```bash
JOB_NAME="leduc-escher-arch-exp9-fast-slow-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.run \
    --seeds 0 \
    --target-nodes 50 \
    --traversals 4 \
    --max-iterations 2 \
    --advantage-train-steps 1 \
    --policy-train-steps 1 \
    --q-train-steps 1 \
    --fast-q-train-steps 1 \
    --calibration-train-steps 1 \
    --rho-train-steps 1 \
    --batch-size 2 \
    --buffer-size 128 \
    --fast-q-buffer-size 128 \
    --rho-buffer-size 128 \
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
- rho by nodes and critic sampled-MSE charts;
- aggregate summary, provenance metadata, worker inputs/results and logs.
