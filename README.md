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
