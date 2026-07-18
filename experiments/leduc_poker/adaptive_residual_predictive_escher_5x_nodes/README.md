# Experiment 4: adaptive predictive ESCHER at Experiment 2 node budgets

Experiment 4 is a horizon-only extension of Experiment 3. It trains the
**Adaptive Residual Predictive ESCHER** architecture over seeds `0`, `1`, and
`2` until each run crosses the paired ESCHER node total from Experiment 2.
It does not rerun ESCHER, VR-DeepDCFR+, or VR-DeepPDCFR+.

The saved Experiment 2 checkpoint curves are bundled with this experiment and
are combined with the new adaptive curves to produce one four-algorithm
exploitability-by-nodes chart. This preserves the original measurements and
avoids approximately 24 hours of duplicate training compute.

## Scientific contract

This experiment changes exactly one scientific variable relative to
Experiment 3: the training horizon. The estimator, solver, network structure,
buffers, optimisers, train-step counts, traversals per outer iteration,
evaluation schedule, adaptive-lambda rule, target-Q update protocol, sampling
policy, and average-strategy weighting are unchanged.

The paired node targets are the exact final ESCHER training-node totals from
Experiment 2:

| Seed | Experiment 2 ESCHER node target |
|---:|---:|
| 0 | 4,700,205 |
| 1 | 4,701,540 |
| 2 | 4,684,695 |

The adaptive solver stops after the first complete outer iteration that reaches
or exceeds the target. Because an outer iteration is indivisible, a small
overshoot is expected and is reported in both absolute and relative terms.
`max_num_iterations=100` remains a safety cap; Experiment 3's measured node
rate projects roughly 30--31 outer iterations for these targets.

There is an untrained-policy evaluation at zero nodes, a checkpoint after
crossing 10,000 nodes, and one checkpoint after each outer iteration. Exact
exploitability evaluation nodes are excluded from `nodes_touched`, as in the
earlier experiments.

## Reused Experiment 2 results

The immutable reference file is `experiment2_checkpoint_curves.csv`. It is a
byte-for-byte copy of the 323-row `checkpoint_curves.csv` produced by:

- Batch job:
  `projects/clever-overview-399515/locations/europe-west1/jobs/leduc-escher-arch-exp2-20260717-105458`
- Run directory: `escher_vs_vr_deep_cfr_5x_nodes_20260717_095755`
- SHA-256:
  `0bd4ace4ea2611a34971aaf7c6ab676c05e39faa3bb3069113d641fac3b53b85`

The runner refuses to load a reference file with a different checksum, row
count, algorithm set, seed set, or paired ESCHER endpoint. Every reused row is
marked `result_source=saved_experiment_2`; every newly trained row is marked
`result_source=experiment_4_new_run`.

## Run locally

Full three-seed experiment:

```bash
python -m experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run
```

Only three adaptive runs are trained. Based on Experiment 3's observed
per-iteration runtime, the sequential full run is expected to take about eight
hours on the same eight-vCPU machine class.

Fast one-seed end-to-end smoke test:

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

The smoke run verifies reference validation, training, initial and early
evaluation, CSV combination, summaries, and plots. Its adaptive performance
metrics have no scientific meaning.

## GCP Batch

Push the current commit so the Batch VM can clone it, then set:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

Full three-seed job with an 18-hour timeout:

```bash
JOB_NAME="leduc-escher-arch-exp4-adaptive-5x-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.adaptive_residual_predictive_escher_5x_nodes.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 64800 8000 32000 100
```

GCP smoke test:

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

Monitor and retrieve either job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

- `adaptive_5x_seed_summary.csv` and
  `adaptive_5x_checkpoint_curves.csv`: newly trained Experiment 4 results;
- `combined_seed_summary.csv` and `combined_checkpoint_curves.csv`: Experiment
  2 reference results plus the new adaptive results;
- `combined_exploitability_by_nodes.png`: the primary four-algorithm chart;
- `combined_final_exploitability.png`;
- `paired_differences.csv`: adaptive final exploitability minus each Experiment
  2 algorithm for the same seed;
- `aggregate_summary.json`, `summary.json`, and `experiment_metadata.json`;
- per-seed worker inputs, logs, results, partial results, and failure records.

