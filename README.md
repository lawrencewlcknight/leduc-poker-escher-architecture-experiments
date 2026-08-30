# Leduc Poker ESCHER Architecture Experiments

This repository contains model-free ESCHER architecture experiments for Leduc
poker. It was created as a clean successor to the original ESCHER experiment
repository so that new architectural changes can be evaluated with the same
metrics, plots, seeds, and artifact conventions used in the MPhil thesis.

The only historical experiment retained is **Experiment 28**, the strongest
validated ESCHER configuration at the point this repository was created. It is
the control against which all new architecture experiments should be assessed.

## Baseline contract

The canonical baseline is defined in
`experiments/leduc_poker/escher_candidate_architecture_multiseed/config.py`.
Its important fixed properties are:

- OpenSpiel game: `leduc_poker`;
- seeds: `1234`, `2025`, `31415`, `27182`, and `16180`;
- 80 iterations, 500 regret traversals, and 500 value traversals per iteration;
- `(256, 256, 128)` policy, regret, and value trunks;
- a 64-unit per-action regret head;
- standardised legal-action regret targets;
- exact exploitability reported as `NashConv / 2`;
- node-touch and wall-clock accounting retained alongside exploitability.

Do not edit this baseline in place for a new hypothesis. Create a new experiment
from `experiments/leduc_poker/escher_architecture_base.py` and record only the
architectural difference. This keeps comparisons auditable and prevents
baseline drift.

## Repository layout

```text
escher_poker/                         Shared solver, networks, metrics, and plots
unbiased_escher/                      Experiment 6 architecture implementation
experiments/leduc_poker/
  escher_candidate_architecture_multiseed/  Experiment 28 baseline
  escher_vs_vr_deep_cfr_matched_nodes/      Three-seed matched-node comparison
  escher_vs_vr_deep_cfr_5x_nodes/           Five-times-longer comparison
  adaptive_residual_predictive_escher/      Experiment 3 adaptive architecture
  adaptive_residual_predictive_escher_5x_nodes/  Experiment 4 long adaptive run
  adaptive_residual_predictive_escher_forensics/ Experiment 5 diagnostics
  unbiased_control_variate_escher_5x_nodes/ Experiment 6 unbiased architecture
  unbiased_escher_vs_vr_deep_cfr_15m_nodes/ Experiment 7 15M-node comparison
  unbiased_control_variate_escher_lean_ablation/ Experiment 8 lean ablation
  fast_slow_control_critic_escher_5x_nodes/ Experiment 9 fast/slow critic
  monte_carlo_control_critic_escher_5x_nodes/ Experiment 10 direct MC critic
  advantage_variance_sampling_escher_5x_nodes/ Experiment 11 advantage sampler
  parallel_multi_action_residual_escher_5x_nodes/ Experiment 12 action subsets
  fixed_beta_reservoir_escher_5x_nodes/ Experiment 13 fixed-beta reservoir
  fixed_beta_reservoir_escher_15m_nodes/ Experiment 14 long reservoir run
  fixed_beta_fast_slow_escher_5x_nodes/ Experiment 15 corrected composition
  unbiased_escher_temporal_checkpoint_head_to_head/ Experiment 16 temporal H2H
  six_algorithm_final_policy_head_to_head/ Experiment 17 six-algorithm H2H
  ucv_escher_parallel_equivalence/      Experiment 18 parallel equivalence
  four_algorithm_heldout_benchmark/     Experiment 19 held-out benchmark
  ucv_exact_tabular_validation/          Experiment 20 exact UCV validation
  escher_architecture_base.py               Baseline-copy helper
  escher_variant_config_utils.py            Derived-config validation
  escher_variant_ablation_runner.py         Multi-variant experiment runner
  escher_single_seed_variant_runner.py      Single-seed diagnostic runner
tests/                                 Unit and baseline-contract tests
docs/                                  Output, cloud, and thesis conventions
scripts/promote_thesis_artifacts.py     Curates lightweight thesis artifacts
outputs/                               Untracked working output
thesis_artifacts/                      Tracked, curated result artifacts
```

## Setup

The code targets Python 3.9.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Run the Experiment 28 baseline

Full five-seed run:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run
```

Fast wiring smoke test:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 2 \
  --value-traversals 2 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-regret 2 \
  --batch-size-value 2 \
  --batch-size-average-policy 2 \
  --memory-capacity 128 \
  --output-root outputs/smoke_tests
```

The smoke test verifies the entry point and export pipeline; it is not a useful
performance estimate.

## Run Experiment 1: matched-node algorithm comparison

The comparison with VR-DeepDCFR+ and VR-DeepPDCFR+ uses the paper's Leduc
training settings, evaluates each VR outer iteration, and stops each VR seed at
the first iteration crossing the paired Experiment 28 node count:

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.run
```

See
`experiments/leduc_poker/escher_vs_vr_deep_cfr_matched_nodes/README.md` for the
comparison contract, upstream provenance, expected memory requirements, and a
fast wiring test.

### Experiment 1 GCP Batch smoke test for both VR algorithms

The following one-seed smoke job runs all three experiment arms: the ESCHER
baseline, VR-DeepDCFR+, and VR-DeepPDCFR+. It uses deliberately tiny buffers,
traversal counts, and training-step counts to verify installation,
orchestration, matched-node stopping, evaluation, plotting, and Cloud Storage
upload. Its performance results are not scientifically meaningful.

Push the current repository first so the Batch VM can clone it, then run this
from the repository root:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"

JOB_NAME="escher-vr-matched-nodes-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.run \
    --seeds 0 \
    --escher-iterations 2 \
    --escher-traversals 2 \
    --escher-value-traversals 2 \
    --escher-evaluation-interval 1 \
    --escher-policy-train-steps 1 \
    --escher-regret-train-steps 1 \
    --escher-value-train-steps 1 \
    --escher-batch-size 2 \
    --escher-memory-capacity 128 \
    --vr-traversals 2 \
    --vr-max-iterations 3 \
    --vr-advantage-train-steps 1 \
    --vr-policy-train-steps 1 \
    --vr-baseline-train-steps 1 \
    --vr-batch-size 2 \
    --vr-buffer-size 128 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

Monitor the job and download its outputs with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Run Experiment 2: five times as many nodes

Experiment 2 extends all three algorithms to the paired node budget produced by
405 ESCHER training cycles (five times Experiment 1's 81 cycles). It also adds
an untrained-policy evaluation at zero nodes for every algorithm and an
additional VR checkpoint immediately after crossing 10,000 training nodes:

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run
```

The complete protocol and 36-hour GCP Batch command are in
`experiments/leduc_poker/escher_vs_vr_deep_cfr_5x_nodes/README.md`.

### Experiment 2 full GCP Batch job

The projected sequential runtime is approximately 24 hours. This command uses
a 129,600-second (36-hour) timeout:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"

JOB_NAME="leduc-escher-arch-exp2-5x-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 129600 8000 32000 100
```

### Experiment 2 GCP Batch smoke test

This job runs all three Experiment 2 arms for one seed with tiny training
settings. The production VR early-evaluation threshold is lowered from 10,000
to 10 nodes so the smoke result verifies the zero-node, early-threshold, and
regular checkpoint pipeline.

```bash
JOB_NAME="leduc-escher-arch-exp2-5x-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run \
    --seeds 0 \
    --escher-iterations 2 \
    --escher-traversals 2 \
    --escher-value-traversals 2 \
    --escher-evaluation-interval 1 \
    --escher-policy-train-steps 1 \
    --escher-regret-train-steps 1 \
    --escher-value-train-steps 1 \
    --escher-batch-size 2 \
    --escher-memory-capacity 128 \
    --vr-traversals 2 \
    --vr-max-iterations 3 \
    --vr-advantage-train-steps 1 \
    --vr-policy-train-steps 1 \
    --vr-baseline-train-steps 1 \
    --vr-batch-size 2 \
    --vr-buffer-size 128 \
    --vr-early-evaluation-nodes 10 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

Use the monitoring and download commands in the Experiment 2 README. The smoke
test's performance metrics have no scientific meaning.

## Run Experiment 3: adaptive residual-corrected predictive ESCHER

Experiment 3 trains only the new adaptive architecture to the three paired
Experiment 1 ESCHER node budgets. It reuses a provenance-recorded copy of the
Experiment 1 checkpoint curves to produce a four-algorithm exploitability chart
without rerunning ESCHER, VR-DeepDCFR+, or VR-DeepPDCFR+:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher.run
```

The architecture, convergence argument, exact configuration, diagnostic
invariants, local smoke test, and GCP Batch commands are documented in
`experiments/leduc_poker/adaptive_residual_predictive_escher/README.md`.

### Experiment 3 local smoke test

This one-seed, two-iteration run verifies the adaptive estimator, initial and
early evaluation checkpoints, Experiment 1 reference-data merge, CSV exports,
and comparison plots. Its performance metrics have no scientific meaning.

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

### Experiment 3 GCP Batch smoke test

Set the Batch environment variables, then submit the same reduced run from the
repository root:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"

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

Monitor the job and download its artifacts with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Run Experiment 4: adaptive architecture at Experiment 2 node budgets

Experiment 4 changes only the training horizon of the Experiment 3 adaptive
architecture. It trains seeds `0`, `1`, and `2` to the paired Experiment 2
ESCHER node totals (approximately 4.7 million nodes each), then combines the
new curves with the immutable saved Experiment 2 ESCHER, VR-DeepDCFR+, and
VR-DeepPDCFR+ curves:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run
```

### Experiment 4 local smoke test

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run \
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

### Experiment 4 GCP Batch smoke test

Use the environment variables defined in the Experiment 3 section above:

```bash
JOB_NAME="leduc-escher-arch-exp4-adaptive-5x-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run \
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

The complete provenance contract, projected runtime, 18-hour full Batch job,
monitoring commands, and output inventory are in
`experiments/leduc_poker/adaptive_residual_predictive_escher_5x_nodes/README.md`.

## Run Experiment 5: adaptive-ESCHER forensic diagnostics

Experiment 5 runs six one-factor architectural-mechanism arms for seeds `0`,
`1`, and `2` at their paired Experiment 1 node budgets (approximately one
million nodes per run). It separates the current regret-matched strategy, an
exact tabular weighted average, and the learned average-policy network; it also
measures exact all-action Q error, estimator bias and variance, and predictor
error against predictive-strategy improvement.

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run
```

### Experiment 5 local smoke test

This executes all six mechanism branches for one seed:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run \
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

### Experiment 5 GCP Batch smoke test

Use the environment variables defined in the Experiment 3 section above:

```bash
JOB_NAME="leduc-escher-arch-exp5-forensics-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher_forensics.run \
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

The full 18-run job is expected to take about 12 hours sequentially and uses a
24-hour Batch timeout. The full submission command, diagnostic definitions,
interpretation guide, monitoring commands, and output inventory are in
`experiments/leduc_poker/adaptive_residual_predictive_escher_forensics/README.md`.

## Run Experiment 6: unbiased control-variate ESCHER

Experiment 6 trains the always-unbiased, three-fold cross-fitted
control-variate architecture for seeds `0`, `1`, and `2` to the exact paired
Experiment 2 ESCHER node budgets. It reuses the saved Experiment 2 ESCHER,
VR-DeepDCFR+, and VR-DeepPDCFR+ curves and produces a single four-algorithm
exploitability-by-nodes chart.

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_5x_nodes.run
```

### Experiment 6 local smoke test

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

### Experiment 6 GCP Batch smoke test

Use the environment variables defined in the Experiment 3 section above:

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

The full job is projected to take about 14 hours sequentially and is configured
with a 36-hour Batch timeout. The proof sketch, cross-fitting contract, full
Batch command, provenance, diagnostics, and output inventory are in
`experiments/leduc_poker/unbiased_control_variate_escher_5x_nodes/README.md`.

## Experiments 7–12: single-Batch schedule

The recommended workflow is one complete GCP Batch job per experiment. The
default runner for each row executes every algorithm or architecture arm and
all three seeds sequentially inside that one job, then produces the combined
outputs before the job exits.

| Experiment | Work inside one Batch job | Best completion estimate | Planning allowance | Set Batch maximum to |
|---|---|---:|---:|---:|
| 7 | 3 algorithms × 3 seeds at 15M nodes | 64.5 hours | 78 hours | **5,760 minutes** |
| 8 | 8 ablation arms × 3 seeds | 72 hours | 72 hours | **5,760 minutes** |
| 9 | Fast/slow critic × 3 seeds | 24 hours | 24 hours | **2,880 minutes** |
| 10 | Monte Carlo critic × 3 seeds | 12 hours | 12 hours | **1,440 minutes** |
| 11 | Advantage sampler × 3 seeds | 12 hours | 12 hours | **1,440 minutes** |
| 12 | Parallel multi-action candidate × 3 seeds | 12 hours | 12 hours | **1,440 minutes** |

The maximum is deliberately larger than the expected duration; a successful
job stops as soon as the runner completes. The Batch submission helper accepts
seconds, so the corresponding arguments are `345600`, `345600`, `172800`,
`86400`, `86400`, and `86400`. Every smoke test below is also a single Batch
job. Use its documented `21600`-second (**360-minute**) timeout.

The full single-job submissions are:

```bash
JOB_NAME="leduc-escher-arch-exp7-15m-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100

JOB_NAME="leduc-escher-arch-exp8-lean-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100

JOB_NAME="leduc-escher-arch-exp9-fast-slow-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 172800 8000 32000 100

JOB_NAME="leduc-escher-arch-exp10-mc-critic-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100

JOB_NAME="leduc-escher-arch-exp11-adv-sampling-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100

JOB_NAME="leduc-escher-arch-exp12-multi-action-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run \
    --parallel-action-workers 3 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

Run the relevant smoke job first. Experiments 7 and 8 also support split-job
recovery, but no splitting is required for the single-Batch workflow above.

## Run Experiment 7: 15-million-node long-horizon comparison

Experiment 7 trains VR-DeepDCFR+, VR-DeepPDCFR+, and the Experiment 6 Unbiased
Control-Variate ESCHER candidate for seeds `0`, `1`, and `2` to a common target
of approximately 15 million training nodes:

```bash
python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run
```

The measured-throughput estimate is approximately 64.5 hours for all nine runs.
For the requested single-Batch workflow, allow 78 hours operationally and set
the maximum to **5,760 minutes** (`345600` seconds). The runner also supports
partial-job recovery, but the default command completes and aggregates all nine
runs in one job.

### Experiment 7 local smoke test

```bash
python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
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

### Experiment 7 GCP Batch smoke test

Use the GCP environment variables defined in the Experiment 3 section above:

```bash
JOB_NAME="leduc-escher-arch-exp7-15m-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
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

The full sequential Batch command, parallel-by-algorithm commands, aggregation
workflow, runtime derivation, configuration contract, and output inventory are
in
`experiments/leduc_poker/unbiased_escher_vs_vr_deep_cfr_15m_nodes/README.md`.

## Run Experiment 8: lean Experiment 6 ablation

Experiment 8 runs the full Experiment 6 architecture and seven simplification
arms for paired seeds `0`, `1`, and `2` at the Experiment 6 per-seed node
budgets. It isolates fixed beta, predictor use and removal, critic count, and
sampling, then directly tests the combined lean candidate: beta-one unbiased
residual correction, two cross-fitted critics, non-predictive DCFR+, uniform
sampling, and no calibration network.

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run
```

### Experiment 8 local smoke test

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
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

### Experiment 8 GCP Batch smoke test

Use the GCP environment variables defined in the Experiment 3 section above:

```bash
JOB_NAME="leduc-escher-arch-exp8-lean-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
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

The complete 24-run job is estimated at about 72 hours. For the requested
single-Batch workflow, set the maximum to **5,760 minutes** (`345600` seconds).
Optional split-job recovery, the proof rationale, interpretation rule and
output inventory are in
`experiments/leduc_poker/unbiased_control_variate_escher_lean_ablation/README.md`.

## Run Experiment 9: fast/slow cross-fitted control critic

Experiment 9 replaces Experiment 6's single-timescale critic folds with paired
fast and slow critics. Fast replay contains only the current outer iteration;
slow replay is a uniform lifetime reservoir. A frozen held-out controller
selects an information-set/action-conditioned convex mixture before each
return is observed. The unbiased residual correction is unchanged.

The new architecture runs seeds `0`, `1`, and `2` to the Experiment 6 paired
node budgets and automatically adds checksum-validated Experiment 6 results to
the performance charts.

```bash
python -m experiments.leduc_poker.fast_slow_control_critic_escher_5x_nodes.run
```

### Experiment 9 local smoke test

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

### Experiment 9 GCP Batch smoke test

Use the GCP environment variables defined in the Experiment 3 section above:

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

Allow approximately 24 hours for the complete three-seed run and set the
single-Batch maximum to **2,880 minutes** (`172800` seconds). The proof
argument, exact replay semantics,
Experiment 6 checksums, full Batch command, diagnostics and output inventory
are in
`experiments/leduc_poker/fast_slow_control_critic_escher_5x_nodes/README.md`.

## Run Experiment 10: current-iteration Monte Carlo control critic

Experiment 10 replaces Experiment 6's bootstrapped TD critic with direct
supervision from the recursively unbiased sampled returns generated during
traversal. Both players collect against one frozen strategy before any regret,
critic, calibration or gate update. Each trajectory writes returns to one
critic fold and uses predictions only from the other folds.

Seeds `0`, `1`, and `2` run to the Experiment 6 paired node budgets, and
checksum-validated Experiment 6 results are automatically included in the
performance charts.

```bash
python -m experiments.leduc_poker.monte_carlo_control_critic_escher_5x_nodes.run
```

### Experiment 10 local smoke test

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

### Experiment 10 GCP Batch smoke test

Use the GCP environment variables defined in the Experiment 3 section above:

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

Allow approximately 12 hours for the complete three-seed experiment and set
the single-Batch maximum to **1,440 minutes** (`86400` seconds). The
frozen-phase contract, unbiasedness proof,
parallelisation properties, Experiment 6 checksums, full Batch command,
diagnostics and output inventory are in
`experiments/leduc_poker/monte_carlo_control_critic_escher_5x_nodes/README.md`.

## Run Experiment 11: centred-advantage variance sampling

Experiment 11 retains the complete Experiment 6 architecture but replaces its
residual-standard-deviation action proposal with one aligned to the Euclidean
variance of the centred advantage vector. The score for each action combines
the predicted second moment of `G - beta * Q`, the current strategy, and the
exact norm of that action's column in the policy-centering operator. The
unchanged uniform floor and exact importance correction preserve full support
and unbiasedness.

Seeds `0`, `1`, and `2` run to the exact Experiment 6 node budgets. Immutable,
checksum-validated Experiment 6 results are automatically included in the
performance charts.

```bash
python -m experiments.leduc_poker.advantage_variance_sampling_escher_5x_nodes.run
```

### Experiment 11 local smoke test

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

### Experiment 11 GCP Batch smoke test

Use the GCP environment variables defined in the Experiment 3 section above:

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

Allow approximately 12 hours for the complete three-seed experiment and set
the single-Batch maximum to **1,440 minutes** (`86400` seconds). The full Batch
command, derivation,
unbiasedness argument, Experiment 6 checksums, diagnostics and output inventory
are in
`experiments/leduc_poker/advantage_variance_sampling_escher_5x_nodes/README.md`.

## Run Experiment 12: parallel multi-action residual correction

Experiment 12 retains Experiment 6's critic, calibration, beta, regret
accumulator and average-policy architecture. At traverser information sets it
replaces the single sampled action with an adaptive nonempty subset. Exact
conditional inclusion probabilities preserve unbiasedness, while sibling
actions share coupled chance/opponent random streams and the first
multi-action frontier executes on three workers.

Seeds `0`, `1`, and `2` run to the exact Experiment 6 node budgets. Immutable,
checksum-validated Experiment 6 results are automatically included in all
performance charts.

```bash
python -m experiments.leduc_poker.parallel_multi_action_residual_escher_5x_nodes.run
```

### Experiment 12 local smoke test

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

### Experiment 12 GCP Batch smoke test

Use the GCP environment variables defined in the Experiment 3 section above:

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

Allow approximately 12 hours for the three-seed job and set the single-Batch
maximum to **1,440 minutes** (`86400` seconds). The exact inclusion derivation,
common-random-number contract,
parallel event-merging design, full Batch command, Experiment 6 checksums and
output inventory are in
`experiments/leduc_poker/parallel_multi_action_residual_escher_5x_nodes/README.md`.

## Experiments 13–14: recommended fixed-beta reservoir candidate

Experiments 13 and 14 combine the strongest supported mechanisms from
Experiments 8 and 9:

- the always-unbiased residual correction is fixed at `beta=1`;
- all three persistent frozen-target critic folds use uniform lifetime
  reservoir replay;
- Experiment 6's calibrated full-support sampler and gated predictor remain;
- Experiment 9's fast critics and rho controller are removed.

Only the new candidate is trained. Experiment 13 imports immutable Experiment
6 results at the paired 4.7M-node budgets. Experiment 14 imports all three
immutable Experiment 7 algorithms at approximately 15M nodes.

| Experiment | Work inside one Batch job | Expected completion | Set Batch maximum to |
|---|---|---:|---:|
| 13 | Candidate × 3 seeds at Experiment 6 budgets | 12 hours | **1,440 minutes** (`86400` seconds) |
| 14 | Candidate × 3 seeds at 15M nodes | 36 hours | **2,880 minutes** (`172800` seconds) |

### Experiment 13 full single GCP Batch job

```bash
JOB_NAME="leduc-escher-arch-exp13-reservoir-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_reservoir_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 86400 8000 32000 100
```

### Experiment 13 GCP Batch smoke test

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

### Experiment 14 full single GCP Batch job

```bash
JOB_NAME="leduc-escher-arch-exp14-reservoir-15m-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_reservoir_escher_15m_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 172800 8000 32000 100
```

### Experiment 14 GCP Batch smoke test

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

The architecture rationale, convergence route, exact comparator checksums,
local smoke tests, runtime derivations and output inventories are documented
in:

- `experiments/leduc_poker/fixed_beta_reservoir_escher_5x_nodes/README.md`;
- `experiments/leduc_poker/fixed_beta_reservoir_escher_15m_nodes/README.md`.

## Experiment 15: fixed-beta full fast/slow control critic

The Experiment 13 audit showed that the Experiment 9 improvement could not be
attributed to its slow lifetime reservoir alone. Experiment 15 therefore keeps
the complete Experiment 9 architecture—fast critics, slow critics and learned
rho controller—and fixes the always-unbiased control-variate coefficient at
`beta=1`.

It also corrects the replay RNG confound found during the audit. Every fast
replay, slow reservoir and rho-controller replay uses a deterministic
component-local Python RNG, so control-side replacement and minibatch sampling
cannot perturb the regret, calibration or average-policy learners.

Only the new candidate is trained. Checksum-protected Experiment 6, 9 and 13
results are included in the charts. Seeds `0`, `1` and `2` use the same paired
approximately 4.7M-node budgets as Experiment 13.

Expected completion is approximately **17 hours** for the three sequential
seeds. Set the single-Batch maximum to **2,160 minutes** (`129600` seconds).

### Experiment 15 full single GCP Batch job

```bash
JOB_NAME="leduc-escher-arch-exp15-fixed-beta-fast-slow-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.fixed_beta_fast_slow_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 129600 8000 32000 100
```

### Experiment 15 GCP Batch smoke test

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

The architectural contract, RNG-isolation correction, comparator checksums,
local smoke command and output inventory are in
`experiments/leduc_poker/fixed_beta_fast_slow_escher_5x_nodes/README.md`.

## Experiment 16: Experiment 7 temporal checkpoint head-to-head

Experiment 16 trains the Experiment 7 Unbiased Control-Variate ESCHER
configuration to approximately 15 million nodes for five independent seeds.
One uninterrupted run per seed saves fitted average policies after the first
complete outer iteration crossing approximately 3M, 6M, 9M, 12M and 15M
nodes. Every pair is then evaluated exactly in both seats; training seed is the
inferential unit, so no Monte Carlo game-count choice is required.

The five sequential seeds are projected to require 54.4 hours from measured
Experiment 7 throughput. Allow **55--65 hours** and set the Batch maximum to
**5,760 minutes** (`345600` seconds).

### Experiment 16 local smoke test

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

### Experiment 16 full single GCP Batch job

```bash
JOB_NAME="leduc-escher-arch-exp16-temporal-h2h-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_escher_temporal_checkpoint_head_to_head.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

### Experiment 16 GCP Batch smoke test

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

The exact estimands, sign-flip inference, snapshot invariants, analysis-only
command and output inventory are documented in
`experiments/leduc_poker/unbiased_escher_temporal_checkpoint_head_to_head/README.md`.

## Experiment 17: six-algorithm final-policy head-to-head

Experiment 17 compares the best approximately 15-million-node configurations
of Deep CFR, DREAM, ESCHER, VR-DeepDCFR+, VR-DeepPDCFR+ and UCV-ESCHER across
the common seeds `1234`, `2025`, `31415`, `27182`, and `16180`. It reuses and
archives the existing final snapshots for four algorithms and trains the two
VR algorithms with the authors' parameterisation imported from Experiment 7.

Every policy pair is evaluated exactly in both seats. No games are sampled;
the paired training seed is the inferential unit. The secondary league also
evaluates all 25 cross-seed policy combinations for each algorithm pair without
treating those correlated matchups as independent samples. With five seeds,
the smallest possible two-sided exact sign-flip p-value is `0.0625`, so effect
sizes and consistency can be reported but conventional two-sided significance
requires more training seeds.

Measured Experiment 7 times project **65.3 hours** for the ten sequential VR
training runs on `n2-standard-8`. Allow **70--80 hours** including staging and
exact analysis. Use the standard project Batch allocation and 96-hour limit.
The wrapper reads the four audited inputs from a versioned bundle in the DREAM
results bucket, to which the Batch service account already has access; `BUCKET`
remains the destination for the new Experiment 17 outputs.

```bash
JOB_NAME="leduc-escher-arch-exp17-six-algorithm-h2h-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "bash gcp/run_experiment_17.sh \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

GCP smoke test using the same Batch configuration:

```bash
JOB_NAME="leduc-escher-arch-exp17-six-algorithm-h2h-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "bash gcp/run_experiment_17.sh \
     --smoke \
     --seeds 1234 \
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
  n2-standard-8 345600 8000 32000 100
```

The complete design, local smoke command, snapshot staging contract, runtime
derivation and output inventory are in
`experiments/leduc_poker/six_algorithm_final_policy_head_to_head/README.md`.

## Experiment 18: parallel UCV-ESCHER equivalence

Experiment 18 compares the exact Experiment 7 UCV-ESCHER learner under its
existing sequential execution and a synchronous three-worker Ray backend. The
parallel arm partitions, rather than multiplies, the 15-million-node traversal
budget. Persistent replay and all authoritative optimisation remain in one
driver; global trajectory IDs preserve the three cross-fitted Q folds. The
independent Q-fold and residual-calibration learners also run concurrently with
a bounded CPU-thread budget.

The three paired seeds are assessed with pre-declared practical-equivalence
margins of `0.02` final exploitability and `0.01` final policy value. The
experiment is expected to take approximately **64 hours** on `n2-standard-8`.
Use an **84-hour / 5,040-minute** Batch timeout.

```bash
JOB_NAME="leduc-escher-arch-exp18-ucv-parallel-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_escher_parallel_equivalence.run \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 302400 8000 32000 100
```

### Experiment 18 GCP Batch smoke test

```bash
JOB_NAME="leduc-escher-arch-exp18-ucv-par-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_escher_parallel_equivalence.run \
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
     --evaluation-frequency 1 \
     --early-evaluation-nodes 10 \
     --parallel-num-workers 2 \
     --parallel-ray-object-store-memory 268435456 \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

The architecture, equivalence estimand, resource controls, local smoke command
and output inventory are documented in
`experiments/leduc_poker/ucv_escher_parallel_equivalence/README.md`.

## Experiment 19: frozen four-algorithm held-out benchmark

This benchmark trains Deep CFR, VR-DeepDCFR+, VR-DeepPDCFR+, and UCV-ESCHER
over eight frozen held-out seeds. Every run saves playable policies at the first
completed iteration crossing 15 million nodes and at the first completed
iteration crossing 11 active hours. The cloud launcher segments production into
32 independent tasks (four algorithms by eight seeds) and runs the exact
head-to-head analysis after training.

### Experiment 19 mandatory local smoke test

Place the Deep CFR repository at the normal sibling workspace location:

```text
deep_cfr_v3/
  leduc_poker_escher_architecture/leduc-poker-escher-architecture-experiments/
  leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments/
```

Then, from this repository, run:

```bash
./gcp/run_four_algorithm_heldout_benchmark.sh smoke-local
```

The smoke test uses development seed `0`, not a held-out seed. It runs all four
training implementations with tiny budgets, writes both endpoint snapshots,
reloads every snapshot as a playable OpenSpiel policy, and completes both exact
head-to-head pipelines. Its numerical results are not scientifically meaningful.

### Experiment 19 GCP prerequisites

Experiment 19 uses a remote controller because its cloud smoke, 32 training
workers, and aggregate analysis are separate Batch jobs. The project must have
the Batch, Compute Engine, Cloud Logging, and Cloud Storage APIs enabled. In
addition to the permissions used for earlier experiments, the service account
must be able to create the controller's child Batch jobs and act as the service
account attached to those jobs:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-results-bucket/heldout-benchmarks"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/batch.jobsEditor"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"
```

These grants are one-time setup operations. The account must also retain its
existing Cloud Logging and selected-bucket read/write permissions. With the
default `PARALLELISM=32`, confirm that the selected region has quota for 256
concurrent N2 vCPUs. Use `PARALLELISM=16` below if only 128 are available.

### Experiment 19 full GCP Batch run

Commit and push the repository before launching, then set immutable repository
commits and the Google Cloud configuration:

```bash
export ARCH_REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export RUN_ID="leduc-heldout-$(date -u '+%Y%m%d-%H%M%S')"

./gcp/run_four_algorithm_heldout_benchmark.sh run
```

`ARCH_REPO_REF` and `DEEP_CFR_REPO_REF` must resolve to pushed commits; they
make all 32 workers use identical source. The single `run` command submits a
lightweight remote controller job and then returns. Once Google Cloud confirms
that submission, the laptop can be closed, disconnected, or switched off. The
controller submits a clean-environment cloud
smoke job, submits the 32-task production job only if smoke succeeds, waits for
training remotely, and finally submits exact aggregation. Standard training VMs
and 32-way parallelism are the defaults.

Useful operational commands are:

```bash
# Inspect the controller and three child Batch definitions without submitting.
./gcp/run_four_algorithm_heldout_benchmark.sh dry-run

# Check jobs associated with RUN_ID.
./gcp/run_four_algorithm_heldout_benchmark.sh status

# Submit a remote recovery controller; validated completed workers are skipped.
./gcp/run_four_algorithm_heldout_benchmark.sh resume

# Reduce simultaneous N2 quota use.
PARALLELISM=16 ./gcp/run_four_algorithm_heldout_benchmark.sh run
```

See the
[complete benchmark protocol and artifact guide](experiments/leduc_poker/four_algorithm_heldout_benchmark/README.md)
for endpoint semantics, frozen seeds and configurations, runtime estimates,
Spot VM trade-offs, service-account requirements, output structure, and the
confirmatory versus descriptive analyses.

## Experiment 20: exact tabular UCV estimator validation

Experiment 20 is independent of Experiment 19. It trains only UCV-ESCHER for
seeds `0`, `1`, and `2`, strictly sequentially on one VM. For each seed it
freezes the first completed iterations crossing 1.5M, 7.5M and 15M nodes, then
enumerates exact conditional action-value and advantage moments for five
estimator configurations over every reachable Leduc information-set/action
pair and all three cross-fitting folds.

### Experiment 20 mandatory local smoke test

Run the development smoke before allocating the production VM:

```bash
python -m experiments.leduc_poker.ucv_exact_tabular_validation.run --smoke \
  --output-root outputs/smoke_tests
```

The smoke succeeds only if snapshot reloads, frozen-state invariants,
predictability checks, exact enumeration, aggregation, and plotting all
complete. Its numerical estimates are not scientific results.

### Experiment 20 GCP prerequisites

Experiment 20 uses the standard single-job Batch launcher and therefore needs
no controller-specific IAM roles. Set the same values used by earlier single-VM
experiments. `REPO_URL` must name the pushed repository containing Experiment
20; the launcher clones its default branch when the VM starts.

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

The project must have the Batch, Compute Engine, Cloud Logging, and Cloud
Storage APIs enabled. The configured Batch service account must retain its
existing permission to write logs and upload objects beneath `$BUCKET`.

### Experiment 20 GCP Batch smoke test

This optional clean-environment smoke checks cloud checkout, installation,
execution, and result upload as well as the experiment itself:

```bash
JOB_NAME="leduc-ucv-exp20-tabular-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_exact_tabular_validation.run --smoke \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

Confirm that the Batch job is `SUCCEEDED` and that `aggregate_summary.json`
reports `all_conditional_unbiasedness_checks_pass: true` and
`predictability_audit_status: pass` before submitting production.

### Experiment 20 full GCP Batch run

Submit one standard eight-vCPU VM with a 48-hour safety timeout:

```bash
JOB_NAME="leduc-ucv-exp20-tabular-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_exact_tabular_validation.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 172800 8000 32000 100
```

The job runs all three seeds sequentially and is fully remote once Batch accepts
it. Closing the laptop does not affect it. The measured training requirement is
33.65 VM-hours; allow approximately 36 hours including diagnostics and
aggregation. The 48-hour limit is a safety cap.

Monitor the job and inspect its uploaded artifacts with:

```bash
gcloud batch jobs describe "$JOB_NAME" \
  --project "$PROJECT_ID" \
  --location "$REGION"

gcloud storage ls "$BUCKET/$JOB_NAME/"
```

The cleanup trap uploads outputs on success or failure. Detailed protocol,
output definitions, validation criteria, and download interpretation are in
`experiments/leduc_poker/ucv_exact_tabular_validation/README.md`.

## Add an architecture experiment

Start every new experiment by calling:

```python
from experiments.leduc_poker.escher_architecture_base import make_default_config

config = make_default_config("leduc_poker_escher_my_architecture")
```

Then change only the fields required by the hypothesis, give each variant a
stable ID and human-readable label, and reuse the shared runner and plotting
utilities. New architectural mechanisms belong in `escher_poker/`; experiment
packages should contain configuration and orchestration rather than forked
solver implementations.

See `docs/OUTPUT_CONVENTIONS.md` before adding metrics or figures. See
`TESTING.md` for verification and `docs/GCP_BATCH_EXPERIMENTS.md` for cloud runs.

## Thesis artifacts

Raw outputs and cloud downloads remain outside Git. Promote reviewed plots,
tables, aggregate summaries, and provenance metadata with:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

The selected files are copied under
`thesis_artifacts/<experiment_name>/<run_directory_name>/` with a promotion
manifest.
