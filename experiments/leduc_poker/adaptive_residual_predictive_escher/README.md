# Experiment 3: Adaptive Residual-Corrected Predictive ESCHER

This experiment evaluates one new model-free algorithm, **Adaptive Residual
Predictive ESCHER**, over three seeds (`0`, `1`, `2`). It trains to the paired
Experiment 1 ESCHER node budgets and then combines the new checkpoints with the
saved Experiment 1 ESCHER, VR-DeepDCFR+, and VR-DeepPDCFR+ curves. The three
reference algorithms are not rerun.

The reference data are an immutable, reduced-column copy of
`checkpoint_curves.csv` from GCP Batch job
`leduc-escher-arch-exp1-20260716-223327`, run directory
`escher_vs_vr_deep_cfr_matched_nodes_20260716_213639`. Keeping the reference
file with the experiment makes the comparison reproducible on a fresh Batch VM
without placing all raw Experiment 1 outputs under version control.

## Architecture

At a visited history, let the current strategy be `pi`, let the fixed
full-support sampling policy select `A ~ xi`, let `G` be the recursively
estimated sampled continuation return, and let the frozen history-Q snapshot
produce all legal-action estimates `Q_hat(a)`. The estimator is

```text
Q_tilde(a) = Q_hat(a)
             + lambda_t * 1{A=a} / xi(A) * (G - Q_hat(A))

A_tilde(a) = Q_tilde(a) - sum_b pi(b | I) Q_tilde(b).
```

The policy-weighted instantaneous advantage is therefore exactly zero, up to
floating-point precision, for every sample. Illegal action entries are exactly
zero.

The algorithm combines five mechanisms:

1. **Persistent frozen-target all-action Q.** The online history-Q network and
   optimiser persist across outer iterations. A separate target snapshot is
   fixed throughout both players' traversal collection in an outer iteration
   and throughout each Q optimisation phase, then hard-synchronised only after
   that phase. This avoids target drift within a collection iteration and the
   loss of learned history values caused by reinitialising Q.
2. **Adaptive residual correction.** The residual correction interpolates
   between direct relative-Q ESCHER (`lambda=0`) and the unbiased VR/DREAM
   control-variate estimator (`lambda=1`).
3. **Bootstrapped discounted cumulative advantages.** The cumulative network
   inherits the VR-DeepPDCFR+ update: positive previous advantages are
   bootstrapped from a frozen target, discounted, combined with the current
   advantage, and learned without reinitialising the cumulative network.
4. **PDCFR+ prediction.** A separate instantaneous-advantage network predicts
   the next update and is used by predictive regret matching.
5. **Correct weighted average strategy.** The average-strategy network retains
   the PDCFR+ `gamma=2` weighting used in Experiment 1.

As in ESCHER, sampling at the **updating player's** nodes is fixed uniform over
legal actions. It therefore has time-independent full support; in Leduc its
minimum action probability is at least one third. Opponent actions are sampled
from the current opponent strategy. This preserves the own-reach weighting of
average-strategy observations and avoids changing more of the ESCHER sampling
contract than the estimator requires. Chance outcomes retain their game-defined
probabilities.

## Lambda calibration and convergence route

Lambda is the maximum of a deterministic floor and a past-residual calibration:

```text
floor(t) = 1 - (1 - lambda_start)
                 / (1 + (t - 1) / half_life) ** power

uncertainty(p, a) = residual_ema(p, a)
                    / (residual_ema(p, a) + residual_scale)

lambda_t(p, a) = max(floor(t), uncertainty(p, a)).
```

The default floor uses `lambda_start=0.2`, `half_life=2`, and `power=1`.
It equals 0.2, 0.6, 0.7333, and 0.8 at iterations 1, 3, 5, and 7 and tends to
one. The uncertainty term starts at 0.8 and rises when past absolute Q
residuals are large. It may fall as Q becomes calibrated, but can never fall
below the increasing schedule floor.

Crucially, lambda for a sample is computed from the residual EMA **before** the
current return is observed. The current residual updates the EMA only after the
advantage has been constructed. Allowing the current return to select its own
lambda would generally bias the estimator in an uncontrolled way.

For fixed lambda, conditional expectation gives

```text
E[Q_tilde(a)] = (1 - lambda_t) Q_hat(a) + lambda_t Q(a).
```

Thus the per-step Q bias is `(1-lambda_t)(Q_hat-Q)`. With bounded Q error and
the default power-one floor, cumulative shrinkage is at most logarithmic, so

```text
sum_{t <= T} (1-lambda_t) ||Q_hat_t-Q_t|| = o(T).
```

Together with fixed full-support traverser sampling, sublinear local PDCFR+
regret, correctly weighted averaging, and sublinear function-approximation
error, this supplies a credible oracle/tabular route to Nash convergence. It is not an unconditional
convergence proof for finite neural networks; that limitation also applies to
other deep CFR approximations.

## Why this might be state of the art

The hypothesis is stronger than simply combining ESCHER and VR-DeepPDCFR+:

- Experiment 28 ESCHER eliminates importance-sampling variance but directly
  inherits Q approximation bias and repeatedly reconstructs regret from a
  capped historical reservoir. Its early plateau is consistent with persistent
  target error and stale regret, rather than insufficient MLP width alone.
- VR-DeepPDCFR+ has the stronger accumulator and an unbiased baseline-corrected
  estimator, but every sampled residual is divided by its sampling probability.
  This preserves correctness while potentially amplifying rare-action error.
- The new estimator treats the two methods as endpoints of a bias--variance
  continuum. It uses a large correction when the frozen Q snapshot has recently
  been inaccurate, permits a lower-variance correction when Q is calibrated,
  and asymptotically recovers the unbiased endpoint through the schedule floor.
- A persistent Q network should improve calibration and make residual shrinkage
  useful. Freezing its target during collection prevents the estimator from
  mixing values from different Q versions inside one data-collection phase.
- Discounting and clipping suppress early erroneous advantages, while the
  instantaneous predictor can exploit slowly changing advantages near an
  equilibrium.

These mechanisms give a concrete reason to expect improvement over Experiment
28 and a testable reason to hope for improvement over both VR variants. They do
not guarantee a state-of-the-art result. A publication-level claim would still
require larger games, additional seeds, estimator/accumulator ablations,
wall-clock and memory comparisons, and a formal regret analysis.

## Matched-node protocol

The exact paired Experiment 1 ESCHER targets are:

| Seed | Target training nodes |
|---:|---:|
| 0 | 942,635 |
| 1 | 939,834 |
| 2 | 962,274 |

The adaptive run stops after the first complete outer iteration that reaches or
exceeds its target. Overshoot is recorded because stopping part-way through an
outer iteration would change the algorithm. Exact exploitability evaluation
does not increment the training-node count, and its Python/NumPy/PyTorch RNG
use is isolated from subsequent training.

All neural-training settings match the Experiment 1 VR-DeepPDCFR+ arm unless
the architecture requires a change: 10,000 traversals per player/iteration,
three 64-unit hidden layers, learning rate 0.001, million-item buffers, batch
size 2,048, 750 cumulative/immediate-advantage steps, 5,000 average-policy
steps, 10,000 history-Q steps, `alpha=2.3`, and `gamma=2`.

There is an untrained uniform-policy checkpoint at zero nodes, a checkpoint
after crossing 10,000 nodes, and one checkpoint after every outer iteration.

## Run

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher.run
```

Only the adaptive algorithm is trained. The bundled Experiment 1 curves are
loaded automatically and included in the output plots and combined CSV files.

Fast end-to-end smoke test:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher.run \
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

Smoke-test performance metrics have no scientific meaning.

## GCP Batch

Full three-seed job:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"

JOB_NAME="leduc-escher-arch-exp3-adaptive-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 28800 8000 32000 100
```

The eight-hour timeout provides substantial headroom over the three sequential
adaptive runs while avoiding the cost of rerunning the nine Experiment 1 arms.

GCP smoke test:

```bash
JOB_NAME="leduc-escher-arch-exp3-adaptive-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher.run \
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

Monitor and download either job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

- `adaptive_seed_summary.csv` and `adaptive_checkpoint_curves.csv`: new runs;
- `combined_seed_summary.csv` and `combined_checkpoint_curves.csv`: saved
  Experiment 1 results plus the new adaptive results;
- `combined_exploitability_by_nodes.png`: the primary requested comparison;
- `combined_final_exploitability.png`;
- `paired_differences.csv`: adaptive final performance minus each Experiment 1
  algorithm for the same seed;
- `aggregate_summary.json`, `summary.json`, and complete experiment metadata;
- per-seed worker inputs, logs, and results.

Estimator-specific checkpoints include lambda floor/mean/min/max, absolute Q
residual, absolute residual correction, policy-weighted centering residual,
minimum full-support traverser sampling probability, and Q target version. The
centering residual should remain near machine precision; a material value is a
correctness alarm.
