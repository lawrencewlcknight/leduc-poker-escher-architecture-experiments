# Experiment 15: fixed-beta full fast/slow control-critic ESCHER

Experiment 15 tests the strongest hypothesis remaining after the Experiment 13
forensic audit. It combines Experiment 8's fixed control-variate coefficient
with the complete Experiment 9 fast/slow critic and learned controller. It
trains seeds `0`, `1`, and `2` to the same paired node budgets as Experiments 6,
9 and 13.

Checksum-protected results from Experiments 6, 9 and 13 are added to every
performance chart without retraining those algorithms.

## Architecture

For information set \(I\) and action \(a\), the control value is

\[
C_t(I,a)=
\rho_t(I,a)C_{\mathrm{fast},t}(I,a)
+[1-\rho_t(I,a)]C_{\mathrm{slow},t}(I,a).
\]

The action-value estimator fixes the control coefficient at one:

\[
\widetilde Q_t(a)=
C_t(a)+
\frac{\mathbf 1\{A=a\}}{\xi_t(a)}
\left(G-C_t(a)\right).
\]

The candidate contains:

- three persistent, frozen-target fast critics trained from only the current
  outer iteration;
- three persistent, frozen-target slow critics trained from uniform lifetime
  reservoirs;
- strict trajectory-level cross-fitting for both timescales;
- Experiment 9's frozen, one-iteration-lagged held-out \(\rho\) controller;
- `beta=1` for every information set and action;
- Experiment 6's residual calibration, adaptive full-support sampling,
  prediction-gated regret accumulation and weighted average-policy learner.

This is the complete Experiment 9 architecture. Unlike Experiment 13, the fast
critics and rho controller are not removed.

## RNG-isolation correction

The audit of Experiment 13 found that reservoir replacement and replay
minibatch selection shared Python's process-wide random generator with the
regret, calibration and average-policy learners. Filling a reservoir therefore
changed minibatches in otherwise unrelated components.

Experiment 15 gives every slow reservoir, fast replay and rho-controller replay
its own deterministic random stream. These streams are derived from the run
seed using fixed integer offsets. Consequently:

- reservoir replacement does not advance global Python RNG state;
- fast and slow critic minibatches do not advance it;
- rho-controller minibatches do not advance it;
- replay streams are reproducible and independent across folds and components.

The runner records and validates `control_replay_rng_isolated=1` at every
trained checkpoint. Earlier experiment definitions remain unchanged.

## Unbiasedness and convergence route

Conditional on the frozen critics, predictable rho and full-support sampling
policy,

\[
\mathbb E[\widetilde Q_t(a)\mid I]=Q_t(a).
\]

Neither critic accuracy nor rho accuracy affects estimator expectation. They
affect only variance. Fixing beta at one therefore preserves the sampled
no-regret-to-Nash route in the tabular/oracle limit under the same neural
approximation and optimisation assumptions as Experiment 6.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Paired targets: `4,700,205`, `4,701,540`, and `4,684,695` training nodes.
- Untrained-policy evaluation before training.
- Early evaluation at approximately 10,000 nodes.
- Evaluation after every complete outer iteration.
- Evaluation-tree nodes are excluded from `nodes_touched`.
- Training stops after the first complete iteration crossing the paired target.
- Only Experiment 15 is trained.
- Experiments 6, 9 and 13 are immutable reference data.

The runner rejects a seed result unless beta stayed exactly one, control replay
RNG isolation was active, the estimator remained policy-centred, and every
fast and slow critic fold received data.

## Immutable comparators

Experiment 9's combined files contain Experiment 6 and Experiment 9:

- `experiment9_combined_checkpoint_curves.csv` — SHA-256
  `b811edc29f6f50d92bba6763eba4a76df6864b6143985d877be3ddb293617994`;
- `experiment9_combined_seed_summary.csv` — SHA-256
  `583e9949c3c02ac781cdbc76c15951a8db26a30ca4d5dae929b52ea5083e47f4`.

Experiment 13 contributes:

- `experiment13_checkpoint_curves.csv` — SHA-256
  `586298bdc0453c6103ec7f3993f76a666dc3544c2b25a65f42d3124627c4a8fd`;
- `experiment13_seed_summary.csv` — SHA-256
  `d55a91eb855b506f78de45cb1817a4a063c89e8d53e210428a4e1d8c9af63f04`.

The runner validates all hashes, row counts, algorithm IDs and seeds before
starting candidate training.

## Run locally

```bash
python -m experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes.run
```

## Local smoke test

```bash
python -m experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes.run \
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

## Runtime and Batch timeout

Experiment 9 required 15.36 hours for three sequential seeds on
`n2-standard-8`. Fixed beta and local replay RNGs add negligible work. Allow
approximately **17 hours** and set the single-Batch maximum to **2,160
minutes** (36 hours, `129600` seconds).

## Experiment 15 full single GCP Batch job

Use the GCP environment variables documented in the root README:

```bash
JOB_NAME="leduc-escher-arch-exp15-fixed-beta-fast-slow-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 129600 8000 32000 100
```

## Experiment 15 GCP Batch smoke test

The smoke test runs seed `0` in one Batch job:

```bash
JOB_NAME="leduc-escher-arch-exp15-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes.run \
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

## Outputs

- candidate and combined checkpoint/seed CSV files;
- paired differences against Experiments 6, 9 and 13;
- aggregate and run summaries;
- exploitability by nodes and wall-clock time;
- final exploitability with standard-error bars;
- rho and held-out critic-error diagnostics;
- fixed-beta and replay-RNG-isolation diagnostics;
- configuration, provenance, worker inputs, results and logs.
