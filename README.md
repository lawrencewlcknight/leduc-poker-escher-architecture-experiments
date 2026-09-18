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
  deep_cfr_ucv_36h_plateau/              Experiment 21 36-hour convergence study
  ucv_three_arm_15m_simplification/      Experiment 22 long-horizon simplification
  ucv_24h_stability_development/          Experiment 23 UCV stability development
  selected_ucv_36h_confirmation/          Experiment 24 selected UCV 36-hour follow-up
  ucv_residual_target_factorial/           Experiment 25 residual/target 2x2 factorial
  ucv_advantage_replay_36h/                 Experiment 26 direct-advantage replay
  average_policy_redistillation/            Experiment 27 offline average-policy distillation
  information_set_stratified_distillation/  Experiment 28 information-set policy sampling
  promoted_ucv_cross_entropy_36h/            Experiment 29 promoted UCV 36-hour comparison
  average_policy_optimization_horizon/       Experiment 41 continued policy-fitting horizon
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

## Experiment 21: Deep CFR and UCV-ESCHER 36-hour convergence

Experiment 21 extends Deep CFR and UCV-ESCHER to 36 active training hours over
five seeds each. It saves a playable policy at the first completed iteration
crossing every two-hour threshold, then computes exact exploitability for all
180 policies. The analysis plots exploitability against both active training
time and nodes touched and reports late-window changes over 24--30, 30--36 and
24--36 hours.

### Experiment 21 mandatory local smoke test

Keep the Deep CFR repository at the sibling location described for Experiment
19, then run:

```bash
./gcp/run_deep_cfr_ucv_36h_plateau.sh smoke-local
```

This runs tiny versions of both learners, creates and reloads every smoke
checkpoint, performs exact policy evaluation and final head-to-head analysis,
and renders both requested charts. Smoke values are not scientific results.

### Experiment 21 GCP prerequisites

Experiment 21 uses the same remote-controller service-account permissions as
Experiment 19. No additional IAM roles are required if Experiment 19 completed
successfully. Set the existing project, region, bucket and service account,
plus immutable pushed commits for both repositories:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export ARCH_REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export RUN_ID="exp21-36h-$(date -u '+%Y%m%d-%H%M%S')"
```

The architecture commit must include Experiment 21 and be pushed before
submission. With the default ten-way parallelism, the selected region needs 80
available N2 vCPUs. The workers use standard `n2-standard-8` VMs; Spot capacity
is deliberately disabled because the playable checkpoints do not contain the
optimizer and replay-buffer state required to resume a preempted trajectory.

### Experiment 21 full GCP Batch run

Submit the complete remote workflow with one command:

```bash
./gcp/run_deep_cfr_ucv_36h_plateau.sh run
```

After Batch accepts the controller, the laptop can be closed or switched off.
The controller runs a clean cloud smoke, ten independent training tasks, and
exact aggregation. At full parallelism, allow about 37--40 elapsed hours plus
any VM provisioning delay. Successful training consumes 360 VM-hours; a
prudent allowance including bootstrap, completed-iteration overshoot, smoke and
aggregation is approximately 375--390 VM-hours. Each training task has a
50-hour hard Batch limit, so an unexpectedly slow worker cannot run without
bound.

Useful commands are:

```bash
# Inspect all four Batch definitions without submitting them.
./gcp/run_deep_cfr_ucv_36h_plateau.sh dry-run

# Check jobs and print the artifact location for RUN_ID.
./gcp/run_deep_cfr_ucv_36h_plateau.sh status

# Recover a failed workflow; validated completed workers are reused.
./gcp/run_deep_cfr_ucv_36h_plateau.sh resume

# Lower simultaneous quota demand without changing successful VM-hours.
PARALLELISM=5 ./gcp/run_deep_cfr_ucv_36h_plateau.sh run
```

Results are stored below `$BUCKET/$RUN_ID/`. Download them from the repository
root with:

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "cloud_outputs/$RUN_ID/"
```

The two requested figures are
`analysis/exploitability_by_training_time.png` and
`analysis/exploitability_by_nodes_touched.png`. Exact checkpoint-level values,
seed trajectories, summaries, late-window diagnostics and final 36-hour
head-to-head results are retained as CSV and JSON files. See the
[complete Experiment 21 protocol](experiments/leduc_poker/deep_cfr_ucv_36h_plateau/README.md)
for the frozen seed/configuration contract and interpretation limits.

## Experiment 22: three-arm UCV simplification at 15 million nodes

Experiment 22 compares complete UCV-ESCHER, fixed `beta=1`, and two
cross-fitted critics over six fresh paired development seeds. Every worker
stops at the first completed iteration crossing 15 million training nodes and
saves a playable final policy. The 18 algorithm/seed runs are independent
standard `n2-standard-8` Batch tasks.

Run the mandatory local smoke first:

```bash
./gcp/run_ucv_three_arm_15m_simplification.sh smoke-local
```

For the fully remote GCP workflow, reuse the Experiment 19/21 project, bucket,
region and controller service account, then set the pushed Experiment 22
commit:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp22-simpl-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=18

./gcp/run_ucv_three_arm_15m_simplification.sh run
```

Once Batch accepts the remote controller, the laptop may be disconnected or
switched off. Full parallelism requires 144 N2 vCPUs and should complete in
approximately 10--13 elapsed hours. Budget approximately 155--170 N2 VM-hours
including the dense mechanism diagnostics, bootstrap, smoke and aggregation.
Every production task has a 20-hour hard limit.

Operational commands are:

```bash
./gcp/run_ucv_three_arm_15m_simplification.sh dry-run
./gcp/run_ucv_three_arm_15m_simplification.sh status
./gcp/run_ucv_three_arm_15m_simplification.sh resume
```

The aggregate analysis reports paired exploitability, runtime and memory
effects; exact sign-flip tests; a predeclared `0.01` non-inferiority margin;
trajectory plots; calibration reliability by information-set/action; beta and
prediction-gate behaviour; correction magnitude; realised target variance
relative to fixed `beta=1`; and the association between critic error and the
next iteration's local regret. See the
[complete Experiment 22 protocol](experiments/leduc_poker/ucv_three_arm_15m_simplification/README.md).

## Experiment 23: 24-hour UCV stability development study

Experiment 23 uses four fresh paired development seeds to compare Original
UCV, a fixed-`beta=1` two-critic fast core, that core with the instantaneous
predictor removed, and a stable non-predictive core with late cosine learning-
rate decay and gradient clipping. Each of the 16 independent
`n2-standard-8` workers saves playable policies every two active hours through
24 hours and at the first completed iteration crossing 15 million nodes.

Run the mandatory local smoke first:

```bash
./gcp/run_ucv_24h_stability_development.sh smoke-local
```

Then reuse the existing GCP configuration and set the pushed Experiment 23
commit:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp23-stab-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=16

./gcp/run_ucv_24h_stability_development.sh run
```

The remote controller owns smoke, production and exact aggregation, so the
laptop may be disconnected after submission. Full parallelism needs 128 N2
vCPUs and should take approximately 25--30 elapsed hours. Budget roughly
400--440 N2 VM-hours; each training worker has an independent 36-hour hard
limit and no automatic retry.

Use the same environment with `status`, `resume`, or `dry-run`. The analysis
pre-specifies mean exploitability over 12--24 hours, checkpoint volatility,
continued late improvement, 15-million-node performance and throughput. It
also produces exact exploitability-by-time and exploitability-by-node charts.
See the [complete Experiment 23 protocol](experiments/leduc_poker/ucv_24h_stability_development/README.md).

## Experiment 24: selected UCV architecture at 36 hours

Experiment 24 trains the non-predictive fast core selected after Experiment 23
for 36 active hours on the same five seeds used in Experiment 21. The selected
architecture uses fixed `beta=1`, two cross-fitted critics and no instantaneous
predictor while retaining residual calibration and residual-adaptive
full-support sampling. It retains the original constant learning rate and does
not use the unsuccessful annealing/clipping package.

Each of five independent `n2-standard-8` workers saves playable policies every
two active hours through 36 hours and at the first completed iteration crossing
15 million nodes. Aggregation verifies and imports the immutable Experiment 21
archive, then creates new three-algorithm charts for Deep CFR, Original UCV and
the selected architecture. Experiment 21 itself is not modified.

Run the mandatory local smoke first:

```bash
./gcp/run_selected_ucv_36h_confirmation.sh smoke-local
```

Then reuse the existing cloud configuration and immutable Deep CFR reference:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export EXP21_RUN_ID="exp21-36h-20260830-141641"
export RUN_ID="exp24-selected-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_selected_ucv_36h_confirmation.sh run
```

The controller owns cloud smoke, production and aggregation, so the laptop may
be disconnected after submission. Full parallelism requires 40 regional N2
vCPUs and should take approximately 37--40 elapsed hours. Budget approximately
190--200 N2 VM-hours including bootstrap, overshoot, smoke and aggregation;
each training worker has a 50-hour hard limit and no automatic retry.

Use the same environment with `status`, `resume`, or `dry-run`. The analysis
reports the pre-specified 24--36-hour mean and final exploitability, late-window
stability, throughput, 15-million-node performance and final exact head-to-head
effects. The use of already examined Experiment 21 seeds is declared as paired
post-selection follow-up evidence, not a new held-out confirmation. See the
[complete Experiment 24 protocol](experiments/leduc_poker/selected_ucv_36h_confirmation/README.md).

## Experiment 25: residual-regret by averaged-critic-target factorial

Experiment 25 starts from the non-predictive fast core selected in Experiment
23 and independently tests two stability interventions: deeper
residual-LayerNorm cumulative-regret networks and four-fit temporally averaged
critic targets. The paired 2x2 design comprises four arms and three fresh
development seeds, giving 12 independent 36-active-hour workers.

Each worker saves playable policies every two active hours and at 15 million
nodes. It also saves complete resumable training states at 24 and 36 hours.
Exact aggregation reports hours 24--36 performance, convergence slope,
checkpoint RMSSD, worst rebound, equal-node performance, throughput, factorial
main effects and their interaction. An exact tabular-average diagnostic
separates regret-learning behaviour from average-policy distillation error.

Run the mandatory local smoke first:

```bash
./gcp/run_ucv_residual_target_factorial.sh smoke-local
```

Then reuse the existing cloud configuration:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp25-fact-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=12

./gcp/run_ucv_residual_target_factorial.sh run
```

The remote controller owns smoke, production and aggregation, so the laptop may
be disconnected after submission. Full parallelism requires 96 regional N2
vCPUs. Budget approximately 450--480 N2 VM-hours and allow about 40--44 elapsed
hours plus provisioning. Training tasks use on-demand `n2-standard-8` VMs, have
a 54-hour per-attempt hard limit and no automatic retry. Continuation states are
uploaded at 24 and 36 hours; use the same `RUN_ID` with `resume` after a failure.

See the
[complete Experiment 25 protocol](experiments/leduc_poker/ucv_residual_target_factorial/README.md).

## Experiment 26: UCV direct-advantage replay at 36 hours

Experiment 26 is the bolder UCV redesign motivated by the long-horizon
volatility in Experiment 21. It starts from Experiment 23's non-predictive fast
core, retains the model-free UCV external-sampling estimator, and replaces the
recursively fitted cumulative-regret target with a persistent Deep-CFR-style
reservoir of instantaneous UCV advantages. Player-specific networks fit those
utility-normalised observations directly with linear iteration weighting; the
previous fitted network is never used as the next target.

Five independent `n2-standard-8` workers use the same five seeds as Experiment
21. Each saves playable policies every two active hours through 36 hours and at
the first completed iteration crossing 15 million nodes. Aggregation imports
the immutable Experiment 21 archive and produces joined exact-exploitability
charts against Deep CFR and Original UCV by training time and nodes touched,
plus late-window, throughput and final head-to-head summaries.

Run the mandatory local smoke first:

```bash
./gcp/run_ucv_advantage_replay_36h.sh smoke-local
```

Then reuse the existing cloud configuration and immutable Experiment 21 Deep
CFR reference:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export EXP21_RUN_ID="exp21-36h-20260830-141641"
export RUN_ID="exp26-replay-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_ucv_advantage_replay_36h.sh run
```

The remote controller owns smoke, production and aggregation, so the laptop may
be disconnected after submission. Full parallelism requires 40 regional N2
vCPUs and should take approximately 37--40 elapsed hours. Budget approximately
190--200 N2 VM-hours; each training worker has a 50-hour hard limit and no
automatic retry. Use the same environment with `status`, `resume`, or
`dry-run`.

The joined reuse of Experiment 21 seeds is paired architecture-development
evidence, not a new held-out confirmation. Because the new arm also includes
the Experiment 23 fast-core changes, its difference from Original UCV is not a
replay-only causal effect. See the
[complete Experiment 26 protocol](experiments/leduc_poker/ucv_advantage_replay_36h/README.md).

## Experiment 27: isolated average-policy redistillation

Experiment 27 reuses the frozen 24- and 36-hour continuation states from the
Experiment 25 averaged-critic-target arm. It does not rerun UCV training. It
first separates exact-average, empirical-reservoir and neural-fitting error,
then compares weighted MSE with soft-target cross-entropy and reset fitting with
warm-started fitting in a paired `2x2` design. Three optimiser fits are nested
within each of the three source trajectories; the source trajectory remains the
inferential unit. An exact-table-supervised neural arm is retained strictly as
a diagnostic capacity bound.

Run the mandatory local smoke first:

```bash
./gcp/run_average_policy_redistillation.sh smoke-local
```

Then reuse the existing cloud configuration and Experiment 25 archive:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP25_RUN_ID="exp25-fact-20260906-003836"
export RUN_ID="exp27-distill-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_average_policy_redistillation.sh run
```

The remote controller owns smoke, production and aggregation, so the laptop
may be disconnected after submission. Three on-demand `n2-standard-8` workers
run in parallel. Each downloads only its own two Experiment 25 source states
(approximately 4.6--5.2 GB), not the full archive, and has a 12-hour hard
limit. This is an offline fitting study and should be much cheaper than a new
36-hour training experiment.

See the
[complete Experiment 27 protocol](experiments/leduc_poker/average_policy_redistillation/README.md).

## Experiment 28: information-set-stratified policy distillation

Experiment 28 follows the cross-entropy result from Experiment 27 and tests
whether rare information sets are under-trained by ordinary reservoir-row
minibatches. It reuses the same frozen Experiment 25 averaged-target states and
compares empirical-mass, square-root-frequency and uniform information-set
sampling. All arms use identical soft-target cross-entropy fitting and exact
importance correction, so they optimize the same empirical objective. Three
optimizer replicates are nested within each of the three source trajectories;
the source trajectory remains the inferential unit.

Run the mandatory local smoke first:

```bash
./gcp/run_information_set_stratified_distillation.sh smoke-local
```

Then reuse the existing cloud configuration and Experiment 25 archive:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP25_RUN_ID="exp25-fact-20260906-003836"
export RUN_ID="exp28-strat-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_information_set_stratified_distillation.sh run
```

The remote controller owns smoke, production and aggregation, so the laptop
may be disconnected after submission. Three on-demand `n2-standard-8` workers
run in parallel; each downloads only its own two Experiment 25 source states
and has a 12-hour hard limit. Every fitted policy is saved, checksum recorded
and reload-validated before a worker can succeed.

See the
[complete Experiment 28 protocol](experiments/leduc_poker/information_set_stratified_distillation/README.md).

## Experiment 29: promoted UCV cross-entropy candidate at 36 hours

Experiment 29 trains the complete revised candidate selected by Experiments
23, 25, 27 and 28. It combines fixed `beta=1`, two cross-fitted critics, no
instantaneous predictor, residual-adaptive full-support sampling, the original
3x64 cumulative-regret MLP, a four-fit temporally averaged critic target, and
reset average-policy fits using iteration-weighted soft-target cross-entropy.
Ordinary empirical-reservoir minibatching is retained.

Five `n2-standard-8` workers use the same seeds as Experiments 21 and 24 and
train for 36 active hours. Policies are saved every two hours and at 15 million
nodes; full resumable states are saved at 24 and 36 hours. Aggregation imports,
rather than retrains, Deep CFR and Original UCV from Experiment 21 and
Simplified UCV from Experiment 24. It creates four-algorithm trajectories by
time and nodes, stability summaries, exact head-to-head tables, and a central
exact-average versus neural-policy distillation diagnostic.

Run the mandatory local smoke first:

```bash
./gcp/run_promoted_ucv_cross_entropy_36h.sh smoke-local
```

Then reuse the existing cloud configuration and frozen reference archives:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export EXP21_RUN_ID="exp21-36h-20260830-141641"
export EXP24_RUN_ID="exp24-selected-20260905-222025"
export RUN_ID="exp29-ce-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_promoted_ucv_cross_entropy_36h.sh run
```

The remote controller owns smoke, production and aggregation, so the laptop
may be disconnected after submission. Full parallelism requires 40 regional
N2 vCPUs. Budget approximately 190--205 N2 VM-hours and allow 38--42 elapsed
hours plus provisioning. Workers use on-demand instances, a 54-hour hard cap,
no automatic retry, and durable 24-hour continuation states.

The joined comparison is explicitly post-selection paired development
evidence, not a new held-out confirmation. See the
[complete Experiment 29 protocol](experiments/leduc_poker/promoted_ucv_cross_entropy_36h/README.md).

## Experiment 30: causal rare-state audit

Experiment 30 is a frozen post-training audit of the promoted Experiment 29
candidate. It reuses the five 24- and 36-hour continuation states and retrains
nothing. At every Leduc information set, it replaces only the deployed neural
average-policy row with its exact reach- and iteration-weighted tabular target
and recomputes exact exploitability. It also constructs cumulative repair
curves ordered by rarity, policy error, measured repair effect, a combined
rarity/error/consequence score, and 20 random rankings.

The audit tests whether rare states causally account for a disproportionate
share of the deployed-policy distillation gap. The intervention effect is
exact; associations between rarity and approximation error remain descriptive.
Run the mandatory local smoke first:

```bash
./gcp/run_causal_rare_state_audit.sh smoke-local
```

Then reuse the existing cloud configuration and identify the completed
Experiment 29 archive:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="your-completed-experiment-29-run-id"
export RUN_ID="exp30-audit-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_causal_rare_state_audit.sh run
```

The remote controller owns smoke, audit and aggregation, so the laptop may be
disconnected after submission. Five on-demand `n2-standard-4` tasks each
download their own two source states; no training compute is repeated. See the
[complete Experiment 30 protocol](experiments/leduc_poker/causal_rare_state_audit/README.md).

## Experiment 31: tabular CFR+ live resolving

Experiment 31 freezes the five 36-hour promoted UCV-ESCHER policies from
Experiment 29 and tests forward search at the start of Leduc's second betting
round. Each solve retains all compatible private deals, weights them by
blueprint reach, and runs full-tree tabular CFR+ to terminal at nine search
budgets. Exact whole-game exploitability, head-to-head value against the
blueprint, nodes, decision latency and playable resolved policies are saved.

Run the mandatory smoke test:

```bash
./gcp/run_tabular_cfr_live_resolving.sh smoke-local
```

Then reuse the existing cloud configuration:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="your-completed-experiment-29-run-id"
export RUN_ID="exp31-resolve-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_tabular_cfr_live_resolving.sh run
```

Five `n2-standard-4` tasks run in parallel and the cloud controller completes
all stages without the laptop. The resolver is public-range-conditioned, but
does not contain a theoretically safe opt-out gadget; exact exploitability is
therefore the required non-degradation check. See the
[complete Experiment 31 protocol](experiments/leduc_poker/tabular_cfr_live_resolving/README.md).

## Experiment 32: UCV-sampled live resolving

Experiment 32 replaces full-tree local traversal with sampled traversal and
tests the UCV contribution causally. It compares a baseline-free
external-sampling resolver with an otherwise matched resolver using predictable
pre-update tabular action-value control variates. Eight paired resolver seeds
are nested within each of the same five Experiment 29 blueprint seeds.
Experiment 31 is imported for combined compute--quality and latency--quality
charts rather than rerun.

Run the mandatory smoke test:

```bash
./gcp/run_ucv_live_resolving.sh smoke-local
```

After Experiment 31 succeeds, run:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="your-completed-experiment-29-run-id"
export EXP31_RUN_ID="your-completed-experiment-31-run-id"
export RUN_ID="exp32-resolve-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_ucv_live_resolving.sh run
```

Five `n2-standard-4` tasks run in parallel under a remote controller. The five
blueprint seeds remain the inferential units; the 40 local resolver runs are
not treated as independent training seeds. See the
[complete Experiment 32 protocol](experiments/leduc_poker/ucv_live_resolving/README.md).

## Experiment 33: consequence-proxy selection and weighted distillation

Experiment 33 asks whether strategically concentrated average-policy error can
be reduced using a scalable consequence signal. It reuses the five frozen
Experiment 29 trajectories at 24 and 36 hours and Experiment 30's exact
single-information-set repair labels; it reruns no UCV regret learning. Proxy
selection is restricted to 24-hour sources and written as an immutable artifact
before redistillation begins. The 36-hour endpoint is the temporally held-out
primary evaluation.

The six arms separate optimisation from objective changes: standard
soft-target cross-entropy, importance-corrected proxy-prioritised sampling,
consequence weighting, policy-error by consequence weighting, targeted
fine-tuning with ordinary-data rehearsal, and an explicitly non-scalable exact
repair-gain upper bound. Three optimiser replicates are nested within each of
the five source trajectories.

Run the mandatory local smoke first:

```bash
./gcp/run_consequence_weighted_distillation.sh smoke-local
```

Then reuse the existing cloud configuration and identify both completed source
runs:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="your-completed-experiment-29-run-id"
export EXP30_RUN_ID="your-completed-experiment-30-run-id"
export RUN_ID="exp33-cwd-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_consequence_weighted_distillation.sh run
```

The cloud controller owns smoke, proxy calculation, selection, redistillation
and aggregation, so the laptop may be disconnected after submission. Five
on-demand `n2-standard-8` workers run in parallel in each array stage. Allow
approximately 1.5--2.5 elapsed hours and budget approximately 7--12 N2
VM-hours. See the
[complete Experiment 33 protocol](experiments/leduc_poker/consequence_weighted_distillation/README.md).

## Experiment 34: Rao--Blackwellised, structure-aware policy distillation

Experiment 34 tests whether the remaining average-policy approximation gap is
caused by repeated-target noise, insufficient capacity, or interference across
player and betting-round subproblems. It reuses the frozen Experiment 29
24-hour and 36-hour training states; it does not rerun UCV training.

The six-arm design crosses individual replay rows versus exactly aggregated
information-set targets with the current shared `3 x 64` network, a
parameter-matched shared `3 x 136` network, and four hard-routed player-by-round
experts. The 24-hour states are used for development selection and all choices
are frozen before the 36-hour validation. Exact-tabular-teacher fits are
diagnostic only.

Run the complete local smoke test first:

```bash
./gcp/run_rao_blackwellised_policy_distillation.sh smoke-local
```

Then, using the same `PROJECT_ID`, `REGION`, `BUCKET`, and `SA_EMAIL` already
configured for Experiments 29--33:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp34-rb-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_rao_blackwellised_policy_distillation.sh smoke-cloud
./gcp/run_rao_blackwellised_policy_distillation.sh run
```

The `run` command submits a cloud-owned controller, so the laptop may be
disconnected afterwards. Check progress or resume a failed stage with:

```bash
./gcp/run_rao_blackwellised_policy_distillation.sh status
./gcp/run_rao_blackwellised_policy_distillation.sh resume
```

With five-way parallelism, allow approximately 3--5 elapsed hours and budget
approximately 12--20 `n2-standard-8` VM-hours, plus short jobs on smaller VMs.

See the
[complete Experiment 34 protocol](experiments/leduc_poker/rao_blackwellised_policy_distillation/README.md).

## Experiment 35: fresh grouped-wide policy confirmation

Experiment 35 integrates the best average-policy architecture selected in
Experiment 34 into the complete UCV-ESCHER training loop. Five previously
unused seeds are trained for 36 active hours on separate on-demand
`n2-standard-8` VMs, with playable checkpoints every two hours and at 15
million nodes.

The candidate uses grouped information-set targets, a shared `3 x 136` policy
network, soft-target cross-entropy, learning rate `0.003`, and 20,000 fitting
updates. At every checkpoint it also refits the former row-wise `3 x 64`
policy from the same reservoir. This supplies a paired contemporaneous control
without a second training trajectory; its cost is excluded from active
training time and its RNG use is isolated.

The final analysis also imports all four frozen Experiment 29 series: Deep
CFR, original UCV-ESCHER, simplified UCV, and the revised cross-entropy UCV.
They are plotted beside the fresh candidate over time and nodes touched, but
are reported as independent-cohort historical comparisons rather than paired
tests because Experiment 35 deliberately retains new seed labels.

Run the mandatory local smoke test first:

```bash
./gcp/run_grouped_wide_policy_confirmation.sh smoke-local
```

After pushing that exact tested commit, reuse the cloud configuration already
used for the preceding experiments:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp35-confirm-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_grouped_wide_policy_confirmation.sh run
```

The cloud-owned controller runs smoke, training and aggregation, so the laptop
may be disconnected after submission. Check or resume with:

```bash
./gcp/run_grouped_wide_policy_confirmation.sh status
./gcp/run_grouped_wide_policy_confirmation.sh resume
```

The nominal production budget is 180 N2 VM-hours. With five-way parallelism,
allow approximately 38--45 elapsed hours including setup and checkpoint
overhead. Each training attempt has a hard 54-hour runtime ceiling. The task
identity is filtered and validated before training, continuation-state uploads
are retried five times, and Batch permits one automatic task retry. A retried
task restores the durable 24-hour continuation state rather than intentionally
starting again; `resume` provides the same recovery path if the whole array is
ultimately reported as failed.

See the
[complete Experiment 35 protocol](experiments/leduc_poker/grouped_wide_policy_confirmation/README.md).

## Experiment 36: best-response-guided supervised repair

Experiment 36 starts from each frozen 36-hour Experiment 29 policy and
fine-tunes only its average-policy network. Exact opponent best responses
identify the information sets reached when exploiting the network; supervised
cross-entropy towards the empirical reservoir policy is then upweighted at
those states. Three development arms use best-response weights 1, 10 and 100.

```bash
./gcp/run_best_response_guided_repair.sh smoke-local
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp36-brrepair-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_best_response_guided_repair.sh run
```

## Experiment 37: KL-constrained exploitability descent

Experiment 37 alternates policy-gradient improvement against exact current best
responses while penalising KL movement from the Experiment 29 blueprint. It
compares KL coefficients 0.01, 0.1 and 1.0. Exact Leduc responses isolate the
objective and make this development evidence rather than a scalable final
algorithm.

```bash
./gcp/run_kl_exploitability_descent.sh smoke-local
export RUN_ID="exp37-kled-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_kl_exploitability_descent.sh run
```

## Experiment 38: NeuRD fine-tuning

Experiment 38 applies exact counterfactual advantages directly to the policy
logits using the NeuRD/Hedge update, comparing learning rates 0.0001, 0.0003
and 0.001.

```bash
./gcp/run_neurd_fine_tuning.sh smoke-local
export RUN_ID="exp38-neurd-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_neurd_fine_tuning.sh run
```

## Experiment 39: vanilla PPO self-play

Experiment 39 is the conventional-RL control. The frozen Experiment 29 network
is fine-tuned by clipped PPO using sampled self-play returns and a learned value
baseline, again comparing learning rates 0.0001, 0.0003 and 0.001.

```bash
./gcp/run_ppo_self_play.sh smoke-local
export RUN_ID="exp39-ppo-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_ppo_self_play.sh run
```

All four experiments use the same five Experiment 29 sources, exact evaluation
schedule and output schema. They are exploratory because exact exploitability
is used to identify each arm's best checkpoint. See the
[complete post-training protocol](experiments/leduc_poker/policy_post_training/README.md).

## Experiment 40: approximate-exploiter-guided policy repair

Experiment 40 converts Experiment 36's exact-best-response intervention into a
sample-only candidate suitable for larger poker variants. Independently trained
one-sided exploiters play against each frozen Experiment 29 policy. A neural
occupancy-density classifier identifies information states reached
disproportionately under adversarial play, and that signal weights additional
supervised fitting towards the original empirical average-policy target.

The experiment compares uniform additional repair, the exact Leduc oracle, a
single approximate exploiter, an exploiter ensemble, and a reliability-gated
ensemble. Two additional exploiters per seat are held out for scalable
validation. Exact best response and exploitability are analysis-only and do not
select checkpoints.

```bash
./gcp/run_approximate_exploiter_guided_repair.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp40-aer-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_approximate_exploiter_guided_repair.sh run
```

Five `n2-standard-4` workers run one source seed each. Every production task has
a six-hour ceiling and one retry; the cloud-owned controller performs smoke,
training and aggregation without requiring the laptop to remain connected. See
the [complete Experiment 40 protocol](experiments/leduc_poker/approximate_exploiter_guided_repair/README.md).

## Experiment 41: average-policy optimisation horizon

Experiment 41 isolates Experiment 40's strongest observation: uniform
additional supervised fitting was more effective than approximate-exploiter
weighting. It loads each frozen Experiment 29 `time_36h` policy and replay
reservoir, constructs one iteration-weighted soft target per observed
information state, and continues one full-batch Adam trajectory to 2,000
updates. The same network is evaluated every 100 updates and saved at 400, 800,
1,200, 1,600 and 2,000 updates.

The fitting path operates only on replay tensors and does not enumerate the
game tree. Exact exploitability and tabular comparators are Leduc-only post-hoc
diagnostics. All five fixed horizons are completed for all five Experiment 29
seeds; no measured checkpoint controls stopping or selection.

```bash
./gcp/run_average_policy_optimization_horizon.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp41-fit-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_average_policy_optimization_horizon.sh run
```

Five `n2-standard-4` workers run in parallel. The remote controller performs
smoke, fitting and aggregation without requiring the laptop to remain
connected. See the [complete Experiment 41 protocol](experiments/leduc_poker/average_policy_optimization_horizon/README.md).

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
