# Experiment 13: fixed-beta lifetime-reservoir ESCHER

Experiment 13 tests the best post-Experiment-12 specification at the exact
paired node budgets used by Experiment 6. It trains seeds `0`, `1`, and `2`,
then automatically adds checksum-protected Experiment 6 curves and summaries
to the performance charts without retraining Experiment 6.

## Architecture

The candidate retains Experiment 6's always-unbiased estimator:

\[
\widetilde Q(a)=
\widehat Q(a)+
\frac{\mathbf 1\{A=a\}}{\xi(a)}
\left(G-\widehat Q(a)\right).
\]

Relative to Experiment 6 it makes exactly two connected changes:

1. **Fixed beta.** The control-variate coefficient is fixed to `beta=1`.
   Experiment 8 found approximately 10.8% lower final exploitability for this
   arm. It also removes noisy information-set/action coefficients whose mean
   was already approximately one.
2. **Lifetime critic replay.** Each of the three persistent, frozen-target,
   cross-fitted critics uses uniform reservoir sampling over its complete
   transition stream. Experiment 9's learned fast-critic weight converged close
   to zero, while its slow lifetime-reservoir critic had lower sampled-return
   error. Experiment 13 therefore retains the supported slow mechanism and
   removes the fast critics and rho controller.

The following Experiment 6 components are unchanged:

- three disjoint trajectory folds with held-out critic inference;
- persistent critic networks and optimisers with frozen collection targets;
- residual calibration for adaptive full-support action sampling;
- prediction-gated PDCFR+/DCFR+ regret accumulation;
- clipped and discounted cumulative advantages;
- correctly weighted average-policy learning;
- network sizes, optimisation work, buffers and evaluation schedule.

The calibration network is retained only because Experiment 6's action sampler
needs residual uncertainty. It no longer controls beta.

## Unbiasedness and convergence route

For every predictable critic and every full-support sampling distribution,

\[
\mathbb E[\widetilde Q(a)\mid I]=Q(a).
\]

Replacing circular replay with a lifetime reservoir changes critic variance and
generalisation, not estimator expectation. Fixing beta at one also preserves
exact unbiasedness. The candidate therefore retains the usual sampled
no-regret-to-Nash route in the tabular/oracle limit, subject to the same
function-approximation and optimisation conditions as Experiment 6.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Paired node targets: `4,700,205`, `4,701,540`, and `4,684,695`.
- The untrained policy is evaluated before training.
- An early evaluation occurs at approximately 10,000 nodes.
- Evaluation follows every complete outer iteration.
- Evaluation-tree nodes are excluded from `nodes_touched`.
- A run stops after the first complete iteration crossing its paired target.
- Only Experiment 13 is trained; Experiment 6 is immutable reference data.

The runner verifies fixed beta, policy centring, nonempty critic folds and
positive lifetime transition counts before accepting a seed result.

## Immutable Experiment 6 comparison

The packaged files are byte-for-byte copies from Batch job
`leduc-escher-arch-exp6-20260718-230108`, run directory
`unbiased_control_variate_escher_5x_nodes_20260718_220419`:

- `experiment6_checkpoint_curves.csv` — SHA-256
  `7f0ecfca091130565275fc27c775cdcd4e96b62eb122759209d9d4f17b0e65b5`;
- `experiment6_seed_summary.csv` — SHA-256
  `10a43adeb4f415f34e45f2498cd25d85977bb53e0da13300ed7618071635daf9`.

The runner validates checksums, row counts, algorithm IDs and seeds before
training.

## Run locally

```bash
python -m experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.run
```

## Local smoke test

```bash
python -m experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.run \
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

## Runtime and Batch timeout

Experiment 6 required approximately 10.59 hours for three sequential seeds.
Experiment 13 performs the same optimisation work, and reservoir insertion has
negligible cost at this scale. Allow approximately **12 hours** and set the
single-Batch maximum to **1,440 minutes** (`86400` seconds).

## Full single GCP Batch job

Use the GCP environment variables documented in the root README:

```bash
JOB_NAME="leduc-escher-arch-exp13-reservoir-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

## GCP Batch smoke test

The smoke test is one Batch job for seed `0`; use a **360-minute**
(`21600`-second) timeout.

```bash
JOB_NAME="leduc-escher-arch-exp13-reservoir-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.run \
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

## Outputs

- candidate and combined checkpoint/seed CSV files;
- paired differences against Experiment 6;
- aggregate and run summaries;
- exploitability by nodes and wall-clock time;
- final exploitability with standard-error bars;
- configuration, provenance and lifetime-replay diagnostics;
- isolated worker inputs, results and logs.
