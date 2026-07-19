# Experiment 12: parallel multi-action residual correction

Experiment 12 tests a single architectural extension to the Experiment 6
always-unbiased control-variate ESCHER. At a traverser's information set it can
evaluate a nonempty subset of actions instead of exactly one action. The subset
expands when the calibrated regret-target noise is large, and sibling action
rollouts use common random numbers and can execute concurrently.

Seeds `0`, `1`, and `2` run to the exact paired Experiment 6 node budgets.
Checksum-protected Experiment 6 curves and summaries are included in every
performance comparison without rerunning the baseline.

## Multi-action estimator

Experiment 6 uses the control `C_a = beta(a) Q_hat(a)`. Experiment 12 samples a
nonempty action subset `S` and estimates every legal action value as

\[
\widetilde Q(a)=C_a+
\frac{\mathbf 1\{a\in S\}}{p_a}(G_a-C_a),
\]

where `p_a = P(a in S | I)` is the action's exact marginal inclusion
probability. The all-action vector is then centred exactly as in Experiment 6:

\[
\widetilde A=(I-\mathbf 1\pi^\top)\widetilde Q.
\]

Subset membership may be correlated; independence is not required by the
estimator. For every legal action with positive inclusion probability,

\[
\mathbb E\left[
\frac{\mathbf 1\{a\in S\}}{p_a}(G_a-C_a)
\middle|I\right]
=Q(a)-C_a.
\]

Therefore `E[Q_tilde(a)|I] = Q(a)` and the centred regret target is unbiased.
The critic, subset rule and random-stream coupling are selected before any of
the corresponding returns is observed.

## Exact nonempty subset sampling

For every legal action, the sampler first defines an independent raw inclusion
probability `r_a`. It rejects and redraws only when all actions are absent. If

\[
q_0=\prod_b(1-r_b),
\]

then the conditional marginal used by the estimator is known in closed form:

\[
p_a=\frac{r_a}{1-q_0}.
\]

This guarantees a nonempty rollout set without approximating inclusion
probabilities. The implementation's proof test enumerates all `2^K - 1`
nonempty subsets and verifies both these marginals and the expected corrected
advantage vector.

## When the subset expands

The calibration network predicts moments of `R_a = G_a - Q_hat(a)`. For the
actual control `C_a = beta(a) Q_hat(a)`, Experiment 12 estimates

\[
m_a=\widehat{\operatorname{Var}}(R_a)+
[\widehat{\mathbb E}(R_a)+(1-\beta(a))\widehat Q(a)]^2.
\]

The Euclidean noise scale entering the centred regret vector is

\[
s_a=\sqrt{m_a}\left\|(I-\mathbf 1\pi^\top)e_a\right\|_2,
\]

with

\[
\left\|(I-\mathbf 1\pi^\top)e_a\right\|_2^2
=1-2\pi(a)+K\pi(a)^2
\]

for `K` legal actions. Before conditioning on a nonempty subset, the sampler
uses

\[
r_a=\operatorname{clip}
\left(\frac{s_a}{\kappa},\frac{\epsilon}{K},1\right),
\]

where `epsilon = 0.2` is Experiment 6's support mass and `kappa = 2` is the
normalized rollout-cost scale. Without clipping, `r_a = s_a/kappa` minimises
the local cost-regularised diagonal objective

\[
\frac{s_a^2}{r_a}+\kappa^2r_a.
\]

Thus low predicted regret noise produces almost exclusively one-action
subsets, whereas high predicted noise spends more nodes on simultaneous
counterfactual rollouts. Every action retains full support.

## Common random numbers

Every top-level traversal owns three independent random streams:

- chance outcomes;
- opponent actions;
- nested action-subset decisions.

When multiple actions are included, every sibling rollout receives an exact
clone of all three streams after the current subset is selected. Consequently,
siblings reaching the same public-card chance event use the same chance
quantile, while opponent decisions use a separate shared quantile stream.
Nested subset draws cannot shift the chance or opponent streams.

Each rollout still has the correct marginal distribution. Correlation changes
variance, not expectation. In poker, shared public-card and opponent noise may
cancel when action values are differenced. The runner records paired squared
return differences so this mechanism can be inspected.

## Parallel execution and deterministic replay

The first multi-action frontier in each top-level traversal is submitted to a
three-worker thread pool. Neural policy, critic and calibration snapshots are
read-only throughout collection. Each worker accumulates local regret,
average-policy, critic and calibration events; the main thread merges those
events in action order only after every sibling finishes.

This avoids concurrent replay-buffer mutation and makes the parallel result
identical to serial branch evaluation for the same seed. A regression test
checks that serial and three-worker training produce the same nodes, samples,
exploitability and learned policy on a small run.

Nested multi-action frontiers are evaluated serially by this reference runner
to avoid recursive thread-pool deadlock, but their rollouts remain independent
tasks. A distributed large-game implementation can schedule every frontier on
a work-stealing executor. The runner records total rollout work, the ideal
fully parallel critical-path node span, realised parallel batches and their
implied ideal speedup.

## Why the architecture remains model-free

The algorithm uses only sampled legal actions, observed trajectory returns and
learned information-set functions. It neither queries transition probabilities
nor enumerates future game states. OpenSpiel supplies chance samples as the
environment, just as in Experiment 6.

Under full-support subset sampling, exact inclusion correction and predictable
controls, the local regret estimates remain unbiased. The standard sampled
no-regret-to-Nash argument therefore remains available in the tabular/oracle
limit, subject to the same sublinear regret and function-approximation
conditions as Experiment 6.

## Controlled comparison

Every Experiment 6 component is unchanged except action-subset traversal:

- three persistent cross-fitted critics;
- adaptive always-unbiased control-variate beta;
- residual calibration;
- prediction-gated PDCFR+/DCFR+ accumulation;
- correctly weighted average-strategy learning;
- network sizes, replay capacities, training steps and evaluation schedule.

Additional branches count every decision, chance and terminal node in
`nodes_touched`. This makes the node chart an explicit comparison between more
precise local updates and fewer top-level trajectories/outer iterations.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Paired targets: `4,700,205`, `4,701,540`, and `4,684,695` nodes.
- Evaluation: untrained policy, approximately 10,000 nodes, and every complete
  outer iteration.
- Evaluation-tree nodes are excluded from training `nodes_touched`.
- Each seed stops after the first complete outer iteration crossing its target.
- Action workers: `3`; normalized rollout-cost scale: `2.0`.
- Only Experiment 12 is trained; Experiment 6 is an immutable reference.

## Immutable Experiment 6 comparison

The packaged references are byte-for-byte copies from Batch job
`leduc-escher-arch-exp6-20260718-230108`, run directory
`unbiased_control_variate_escher_5x_nodes_20260718_220419`:

- `experiment6_checkpoint_curves.csv` — SHA-256
  `7f0ecfca091130565275fc27c775cdcd4e96b62eb122759209d9d4f17b0e65b5`;
- `experiment6_seed_summary.csv` — SHA-256
  `10a43adeb4f415f34e45f2498cd25d85977bb53e0da13300ed7618071635daf9`.

The runner validates both checksums, row counts, algorithm IDs and seeds before
training. Saved Experiment 6 wall-clock results are context rather than a
simultaneous hardware benchmark.

## Diagnostics and interpretation

Each checkpoint records all Experiment 6 metrics plus:

- realised and predicted mean subset size and maximum subset size;
- fraction of traverser information sets sampling multiple actions;
- raw and conditional inclusion-probability summaries;
- predicted centred-regret noise and diagonal variance proxy;
- common-random-number groups and paired squared return differences;
- executed parallel batches;
- ideal parallel node speedup and parallelisable-node fraction.

The first mechanism check is whether subset size increases with predicted noise
without collapsing inclusion probabilities. Performance should be assessed by
paired final exploitability and normalised exploitability AUC by nodes. The
wall-clock chart measures this three-worker implementation, while the ideal
span diagnostic estimates remaining distributed parallelism.

## Run locally

```bash
python -m experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run
```

## Local smoke test

```bash
python -m experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run \
  --seeds 0 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --calibration-train-steps 1 \
  --batch-size 2 \
  --buffer-size 256 \
  --subset-rollout-cost-scale 2.0 \
  --parallel-action-workers 3 \
  --early-evaluation-nodes 10 \
  --output-root outputs/smoke_tests
```

## Runtime estimate

Experiment 6 required approximately `10.59` hours for its three sequential
seeds. Experiment 12 pays for extra branches inside the same node budget and
executes its first counterfactual frontier concurrently, but event collection
and thread scheduling add overhead. Allow approximately **12 hours** for the
three-seed job and set the single-Batch maximum to **1,440 minutes** (24 hours,
`86400` seconds).

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
JOB_NAME="leduc-escher-arch-exp12-multi-action-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run \
    --parallel-action-workers 3 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

## GCP Batch smoke test

This is one Batch job for seed `0`; use its **360-minute** (`21600`-second)
timeout.

```bash
JOB_NAME="leduc-escher-arch-exp12-multi-action-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run \
    --seeds 0 \
    --target-nodes 50 \
    --traversals 4 \
    --max-iterations 2 \
    --advantage-train-steps 1 \
    --policy-train-steps 1 \
    --q-train-steps 1 \
    --calibration-train-steps 1 \
    --batch-size 2 \
    --buffer-size 256 \
    --subset-rollout-cost-scale 2.0 \
    --parallel-action-workers 3 \
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
- adaptive-subset and parallelism diagnostic charts;
- aggregate summary, provenance metadata, worker inputs/results and logs.
