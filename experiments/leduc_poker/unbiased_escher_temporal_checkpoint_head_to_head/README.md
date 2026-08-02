# Experiment 16: Experiment 7 Temporal Checkpoint Head-to-Head

## Research question

Does the reduction in exploitability observed during long-horizon training of
Unbiased Control-Variate ESCHER correspond to progressively stronger direct
play against its own earlier policies?

## Design

Each of five fixed seeds trains one uninterrupted instance of the Experiment 7
candidate:

```text
checkpoint targets:  3M, 6M, 9M, 12M, 15M nodes
seeds:               1234, 2025, 31415, 27182, 16180
```

The candidate configuration is imported directly from Experiment 7. No
algorithm, optimiser, replay, sampling, network or training-budget setting is
changed. A lightweight snapshot of the fitted neural average-policy is saved
after the first complete outer iteration crossing each threshold. Actual nodes
touched are recorded and used on every temporal chart.

The callback does not stop, restart or reload training. It copies only the
average-policy state to CPU, does not serialize replay or optimiser state and
does not advance any random generator. The runner verifies that the final
snapshot is exactly equal to the uninterrupted solver's live final policy.

## Exact evaluation

Leduc permits exact policy evaluation. Every pair of checkpoints within a seed
is evaluated in both seat assignments. If A is the later policy and B the
earlier policy, the effect is:

```text
0.5 * (value of A as player 0 against B + value of A as player 1 against B)
```

There is no Monte Carlo match noise or arbitrary game-count choice. A positive
effect means that the later checkpoint has positive exact seat-averaged EV
against the earlier checkpoint.

This is a complement to exploitability, not a replacement for it. A policy can
be less exploitable without beating a particular earlier policy directly.

## Statistical protocol

The independent training seed, rather than the ten correlated checkpoint pairs
within a seed, is the primary inferential unit.

- Primary estimand: per-seed mean later-versus-earlier EV over all ten pairs.
- Adjacent estimand: per-seed mean EV against the immediately previous policy.
- Endpoint estimand: 15M-node policy EV against the 3M-node policy.
- Secondary tests: all ten checkpoint-pair effects.
- Inference: 95% t intervals and exact one-sided sign-flip tests.
- Multiplicity: Holm family-wise correction for the ten secondary tests.

With five seeds, the smallest possible one-sided exact sign-flip p-value is
`1 / 32 = 0.03125`. Effect size, confidence intervals and cross-seed
consistency remain primary interpretive evidence.

## Local run

```bash
python -m experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run
```

Re-run exact analysis without retraining:

```bash
python -m experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run \
  analyse \
  --run-dir outputs/unbiased_escher_temporal_checkpoint_head_to_head/RUN_DIRECTORY
```

## Local smoke test

The smoke test runs one seed to 50 nodes. All five logical stages may represent
the same completed outer iteration at this deliberately tiny budget; its
head-to-head values are not scientific.

```bash
python -m experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run \
  --seeds 1234 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --calibration-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --early-evaluation-nodes 5 \
  --output-root outputs/smoke_tests
```

## Runtime

The completed Experiment 7 candidate required a mean of 10.87 hours per
15M-node seed on `n2-standard-8`. Five sequential seeds project to 54.4 hours.
Allow approximately 55--65 hours and set the single-Batch maximum to
**5,760 minutes** (96 hours, `345600` seconds).

## Full GCP Batch job

Use the GCP environment variables documented in the root README:

```bash
JOB_NAME="leduc-escher-arch-exp16-temporal-h2h-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

## GCP Batch smoke test

```bash
JOB_NAME="leduc-escher-arch-exp16-temporal-h2h-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run \
    --seeds 1234 \
    --target-nodes 50 \
    --traversals 4 \
    --max-iterations 2 \
    --advantage-train-steps 1 \
    --policy-train-steps 1 \
    --q-train-steps 1 \
    --calibration-train-steps 1 \
    --batch-size 2 \
    --buffer-size 128 \
    --early-evaluation-nodes 5 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

## Principal outputs

| File | Contents |
| --- | --- |
| `training_stage_metrics.csv` | Target and actual nodes, iteration, elapsed time and replay sizes at each snapshot. |
| `training_checkpoint_curves.csv` | Every exact evaluation generated during uninterrupted training. |
| `checkpoint_exploitability_metrics.csv` | Exact NashConv/2 and policy value by seed and checkpoint. |
| `head_to_head_pairwise.csv` | Exact two-seat EV for all 25 ordered pairs per seed. |
| `head_to_head_primary_effect_by_seed.csv` | One independent primary effect per training seed. |
| `head_to_head_inference_summary.csv` | Primary, adjacent and endpoint estimates, intervals and exact p-values. |
| `head_to_head_pairwise_inference.csv` | Secondary pair effects with Holm-adjusted p-values. |
| `aggregate_summary.json` | Machine-readable estimands and actual checkpoint nodes. |
| `head_to_head_later_vs_earlier.png` | Exact-EV matrix labelled by mean nodes touched. |
| `head_to_head_strength_vs_earlier_by_nodes.png` | Mean EV against all earlier checkpoints. |
| `head_to_head_strength_vs_previous_by_nodes.png` | Adjacent-checkpoint EV. |
| `exploitability_by_nodes.png` | Exact checkpoint exploitability. |
| `average_policy_value_by_nodes.png` | Exact checkpoint average-policy value. |
| `strength_vs_exploitability.png` | Equilibrium quality against temporal head-to-head strength. |
| `snapshots/*.pkl` | Lightweight playable PyTorch average-policy snapshots. |
