# ESCHER vs VR-DeepCFR+ at matched node budgets

This experiment compares the Experiment 28 ESCHER baseline with
VR-DeepDCFR+ and VR-DeepPDCFR+ over three paired seeds (`0`, `1`, `2`). The VR
implementation is adapted from `rpSebastian/DeepPDCFR` at commit
`9f156c9fcdac7f8c9bd0debf94c9432d222858d3`; detailed provenance and integration
corrections are in `vr_deep_cfr/UPSTREAM.md`.

The source paper is *Deep (Predictive) Discounted Counterfactual Regret
Minimization* (AAAI 2026, DOI `10.1609/aaai.v40i20.38780`). Table 2 of the
extended arXiv version (`2511.08174`) is the configuration authority.

## Comparison protocol

The ESCHER run for a seed executes first with the exact Experiment 28 baseline
configuration. Its final training-node count becomes the target for both VR
runs with that seed. Each VR run stops after the first complete outer iteration
that reaches or exceeds the target. The overshoot and relative overshoot are
reported explicitly because VR node consumption cannot be stopped part-way
through an iteration without changing the algorithm.

VR evaluation occurs after every outer iteration (`evaluation_frequency=1`).
With 10,000 traversals per player, this gives checkpoint spacing close to
ESCHER's roughly 100,000-node spacing. Exact exploitability evaluation does not
increment either algorithm's training-node counter. The Python, NumPy, and
PyTorch RNG states are restored after every VR average-policy fit/evaluation so
denser evaluation cannot change subsequent training samples.

The VR learning settings are the Leduc settings in Table 2 of the paper:

- 10,000 traversals per player and outer iteration;
- three hidden layers of 64 units;
- learning rate 0.001 and epsilon 0.6;
- 1,000,000-item advantage, average-policy, and history-value buffers;
- 750 advantage, 5,000 average-policy, and 10,000 history-value training steps;
- batch size 2,048 for all three networks;
- alpha/gamma 2/2 for VR-DeepDCFR+ and 2.3/2 for VR-DeepPDCFR+;
- immediate-regret reinitialisation for VR-DeepPDCFR+.

The released repository YAML differs from the paper table: it specifies a
150,000 advantage buffer and 1,000 history-value steps. The paper settings are
the experiment defaults because the requested comparison is against the
authors' reported Leduc results. Both values are preserved in metadata.

## Run

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.run
```

The default job is computationally and memory intensive. Each algorithm/seed is
run in a fresh subprocess so TensorFlow and PyTorch allocations are released.
Use a machine with at least 32 GiB RAM; 64 GiB is safer for the million-entry
history-value replay buffer and network training batches.

Fast end-to-end wiring check:

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.run \
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
  --output-root outputs/smoke_tests
```

Smoke-test metrics have no scientific meaning.

### GCP Batch smoke test

First push this repository to a Git remote that the Batch VM can clone. Set
`REPO_URL` when using a fork or a repository URL other than the launcher's
default. Then run the following from the repository root:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/OWNER/leduc-poker-escher-architecture-experiments.git"

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

The final five arguments select the machine type, maximum runtime in seconds,
CPU milli, memory MiB, and boot-disk size in GiB. After submission, inspect and
download the smoke-test result with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

The run exports the same core thesis artifacts as the earlier repositories:

- `seed_summary.csv` and `aggregate_summary.json`;
- `checkpoint_curves.csv`;
- paired final differences against ESCHER;
- exploitability, average-policy value, and value-error curves by nodes touched;
- final exploitability by algorithm;
- complete configurations, provenance, node-matching rules, and failures.
