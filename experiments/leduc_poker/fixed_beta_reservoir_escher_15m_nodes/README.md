# Experiment 14: fixed-beta reservoir ESCHER at 15 million nodes

Experiment 14 trains the Experiment 13 specification to approximately 15
million nodes for seeds `0`, `1`, and `2`. It reuses all nine checksum-protected
Experiment 7 runs—VR-DeepDCFR+, VR-DeepPDCFR+, and Unbiased Control-Variate
ESCHER—so no comparator is retrained.

## Purpose

Experiment 13 determines whether fixed beta and lifetime critic replay improve
the candidate at the Experiment 6 horizon. Experiment 14 asks whether any gain
persists or widens over the longer horizon where Experiment 7 showed the prior
candidate outperforming both VR-Deep algorithms by approximately 40%.

The architecture is identical to Experiment 13:

- always-unbiased residual correction with `beta=1`;
- three persistent frozen-target cross-fitted critics;
- uniform lifetime reservoir replay within every critic fold;
- Experiment 6 residual-adaptive full-support sampling;
- Experiment 6 gated predictive regret learner;
- correctly weighted average-policy learning;
- no fast critic and no rho controller.

Only the iteration safety cap is raised to `120`, matching Experiment 7. No
learning rule or optimisation setting is changed.

## Experimental contract

- Seeds: `0`, `1`, `2`.
- Common target: `15,000,000` training nodes.
- Each candidate seed stops after the first complete outer iteration crossing
  the target.
- Evaluation occurs before training, at approximately 10,000 nodes, and after
  every complete outer iteration.
- Evaluation-tree nodes are excluded from `nodes_touched`.
- Only the new candidate is trained.
- All three Experiment 7 algorithms are plotted from immutable saved results.

## Immutable Experiment 7 comparison

The packaged files are byte-for-byte copies from Batch job
`leduc-escher-arch-exp7-20260720-002431`, run directory
`unbiased_escher_vs_vr_deep_cfr_15m_nodes_20260719_232708`:

- `experiment7_checkpoint_curves.csv` — SHA-256
  `d0869cc7926525ddc7afd31b9a87c5d30929a10b556f69e79a6d943ebb6b9e38`;
- `experiment7_seed_summary.csv` — SHA-256
  `028d6f364613cee6211858d2792785957c8fd558e5435a8df976737236610853`.

The runner verifies both hashes, 862 checkpoint rows, nine summary rows, all
three algorithm IDs and all three seeds before training.

## Run locally

```bash
python -m experiments.leduc_poker.fixed_beta_reservoir_escher_15m_nodes.run
```

## Local smoke test

```bash
python -m experiments.leduc_poker.fixed_beta_reservoir_escher_15m_nodes.run \
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

The prior unbiased candidate required approximately 32.62 hours for its three
sequential Experiment 7 seeds. Experiment 14 performs the same optimisation
work. Allow approximately **36 hours** and set the single-Batch maximum to
**2,880 minutes** (`172800` seconds).

## Full single GCP Batch job

Use the GCP environment variables documented in the root README:

```bash
JOB_NAME="leduc-escher-arch-exp14-reservoir-15m-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_reservoir_escher_15m_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 172800 8000 32000 100
```

## GCP Batch smoke test

The smoke test is one Batch job for seed `0`; use a **360-minute**
(`21600`-second) timeout.

```bash
JOB_NAME="leduc-escher-arch-exp14-reservoir-15m-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_reservoir_escher_15m_nodes.run \
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
- paired differences against every Experiment 7 algorithm;
- aggregate and run summaries;
- four-algorithm exploitability charts by nodes and wall-clock time;
- final exploitability with standard-error bars;
- configuration, provenance and lifetime-replay diagnostics;
- isolated worker inputs, results and logs.
