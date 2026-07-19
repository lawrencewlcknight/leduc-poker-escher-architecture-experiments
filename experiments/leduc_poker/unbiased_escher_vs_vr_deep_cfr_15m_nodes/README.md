# Experiment 7: 15-million-node long-horizon comparison

Experiment 7 trains the following algorithms to a common target of approximately
15,000,000 training nodes for seeds `0`, `1`, and `2`:

- VR-DeepDCFR+;
- VR-DeepPDCFR+;
- the Unbiased Control-Variate ESCHER architecture introduced in Experiment 6.

All three algorithms are trained again. No Experiment 2 or Experiment 6 curves
are reused because those runs stop at approximately 4.7 million nodes.

## Experimental contract

The VR arms retain the Experiment 2 configurations, which reproduce the
authors' reported Leduc settings and the existing correctness fixes. The
candidate retains the complete Experiment 6 architecture and configuration:

- always-unbiased residual-corrected control-variate estimator;
- three-fold cross-fitted persistent Q ensemble;
- held-out residual calibration and adaptive beta;
- residual-adaptive full-support action sampling;
- prediction-gated DCFR+/PDCFR+ regret accumulation;
- correctly weighted average-policy learning.

The only production configuration change is the training horizon. The outer
iteration safety cap is raised from 100 to 120 because the measured 4.7M-node
runs imply approximately 89--99 iterations will be required to cross 15M.

Every arm:

- evaluates the untrained policy at zero training nodes;
- evaluates after crossing approximately 10,000 nodes;
- evaluates after every complete outer iteration;
- stops after the first complete iteration that reaches or exceeds 15M nodes;
- excludes exact-evaluation nodes from `nodes_touched`.

## Runtime estimate

The estimate is derived from measured Experiment 2 and Experiment 6 throughput
on the same eight-vCPU machine class, scaled linearly to 15M nodes:

| Algorithm | Measured hours per 15M-node seed | Three sequential seeds |
|---|---:|---:|
| VR-DeepDCFR+ | 4.63 | 13.90 |
| VR-DeepPDCFR+ | 5.66 | 16.97 |
| Unbiased Control-Variate ESCHER | 11.22 | 33.65 |
| **All runs** |  | **64.53** |

Allowing for VM variation, longer evaluation histories and operational
headroom, plan for approximately **78 hours** if all nine runs execute
sequentially. The full single-job command therefore uses a **5,760-minute
timeout** (96 hours, `345600` seconds).

The runner accepts algorithm and seed subsets and can aggregate their worker
results later. If the three algorithms are submitted concurrently, with each
job processing its three seeds sequentially, expected elapsed time is about
**34 hours measured / 42 hours conservatively**. If all nine algorithm/seed
pairs are submitted independently, expected elapsed time is approximately
**12--15 hours**, dominated by the candidate.

These estimates assume performance remains approximately linear in nodes.
Replay buffers are capacity-bounded, so there should not be a threefold memory
increase, but exact duration can only be confirmed by the run.

## Local full run

```bash
python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run
```

## Local smoke test

This runs all three algorithms for one seed using tiny buffers and training
counts. It verifies orchestration, initial/early/outer checkpoints, candidate
diagnostics, aggregation and plots; its performance values are meaningless.

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

This is the simplest submission and automatically produces the combined
results, but it is expected to occupy one VM for roughly 65--78 hours.

```bash
JOB_NAME="leduc-escher-arch-exp7-15m-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

## Optional split-job recovery

The requested and recommended path is the single job above. If that job is
interrupted after producing partial worker results, these algorithm-specific
commands can recover the missing work and the aggregator can rebuild the
combined outputs.

```bash
JOB_DCFR="leduc-escher-arch-exp7-15m-dcfr-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_DCFR" \
  "python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
    --algorithms vr_deep_dcfr_plus \
    --output-root outputs/cloud/$JOB_DCFR" \
  n2-standard-8 86400 8000 32000 100

JOB_PDCFR="leduc-escher-arch-exp7-15m-pdcfr-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_PDCFR" \
  "python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
    --algorithms vr_deep_pdcfr_plus \
    --output-root outputs/cloud/$JOB_PDCFR" \
  n2-standard-8 86400 8000 32000 100

JOB_CANDIDATE="leduc-escher-arch-exp7-15m-candidate-$(date -u +%Y%m%d-%H%M%S)"
./gcp/submit_batch_experiment.sh \
  "$JOB_CANDIDATE" \
  "python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
    --algorithms unbiased_control_variate_escher \
    --output-root outputs/cloud/$JOB_CANDIDATE" \
  n2-standard-8 172800 8000 32000 100
```

Individual seeds can be split further with `--seeds 0`, `--seeds 1`, or
`--seeds 2`. Use a 24-hour timeout for each single-seed job.

After downloading the partial jobs, create the combined result by repeating
`--aggregate-run-dir` for each downloaded output directory:

```bash
python -m experiments.leduc_poker.unbiased_escher_vs_vr_deep_cfr_15m_nodes.run \
  --aggregate-run-dir cloud_outputs/DCFR_JOB \
  --aggregate-run-dir cloud_outputs/PDCFR_JOB \
  --aggregate-run-dir cloud_outputs/CANDIDATE_JOB \
  --output-root outputs/experiment_7_aggregated
```

The aggregator recursively finds `worker_results/*.json`, rejects duplicate
algorithm/seed pairs, and regenerates all combined summaries and plots.

## GCP Batch smoke test

The smoke test is one Batch job covering all three algorithms for seed `0`.
Its timeout is **360 minutes** (`21600` seconds).

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

Monitor and retrieve any job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

- `seed_summary.csv`;
- `checkpoint_curves.csv`;
- `paired_differences.csv`;
- `aggregate_summary.json` and `summary.json`;
- `exploitability_by_nodes.png`;
- `exploitability_by_wall_clock.png`;
- `final_exploitability.png`;
- experiment metadata, partial recovery files, worker inputs, results and logs.
