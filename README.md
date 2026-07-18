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
experiments/leduc_poker/
  escher_candidate_architecture_multiseed/  Experiment 28 baseline
  escher_vs_vr_deep_cfr_matched_nodes/      Three-seed matched-node comparison
  escher_vs_vr_deep_cfr_5x_nodes/           Five-times-longer comparison
  adaptive_residual_predictive_escher/      Experiment 3 adaptive architecture
  adaptive_residual_predictive_escher_5x_nodes/  Experiment 4 long adaptive run
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

## Run the matched-node algorithm comparison

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

Fast local smoke test:

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

GCP Batch smoke test, using the environment variables defined above:

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

### Full Experiment 2 GCP Batch job

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
