# Experiment 2: ESCHER vs VR-DeepCFR+ at 5x node budgets

This experiment extends the three-seed Experiment 1 comparison to approximately
five times as many training nodes. It compares the Experiment 28 ESCHER
architecture with VR-DeepDCFR+ and VR-DeepPDCFR+ over paired seeds `0`, `1`, and
`2`.

The VR implementation is adapted from `rpSebastian/DeepPDCFR` at commit
`9f156c9fcdac7f8c9bd0debf94c9432d222858d3`. The Leduc learning hyperparameters
remain the Table 2 settings used in Experiment 1; only the training horizon and
evaluation observations change.

## Training and node-matching protocol

Experiment 28's solver executes `num_iterations + 1` training cycles. The
Experiment 1 setting of 80 therefore gives 81 cycles. This experiment uses
`num_iterations=404`, giving exactly 405 cycles, or five times the ESCHER
training cycles. It does not change ESCHER's architecture, traversals per cycle,
network training steps, buffer sizes, or other baseline learning settings.

For each seed, ESCHER runs first. Its actual final training-node count becomes
the target for both paired VR runs. Each VR run stops after the first complete
outer iteration that reaches or exceeds the target; its absolute and relative
overshoot are reported. The VR safety cap remains 100 outer iterations, well
above the approximately 31--32 iterations projected for this budget.

## Early evaluation protocol

The experiment adds two evaluation observations without adding to the training
node counters:

- all three algorithms are evaluated before training, at zero training nodes;
- both VR algorithms are additionally evaluated after the first complete
  trajectory that crosses 10,000 training nodes.

The zero-node point measures each implementation's actual untrained policy.
VR's zero-initialised output head is uniform over legal actions; ESCHER reports
its built but untrained policy network. The 10k VR point fits the average-policy
network using the replay data available at that point and then computes exact
Leduc exploitability. Python, NumPy, and PyTorch RNG states are restored after
the fit/evaluation, so this checkpoint does not alter later traversal samples.
Regular VR evaluation still occurs after every outer iteration. Exact evaluation
tree nodes are excluded from `nodes_touched` for every algorithm.

Checkpoint rows include `checkpoint_kind`, `checkpoint_target_nodes`,
`is_initial_policy_evaluation`, and `is_final_policy_evaluation` so the zero,
10k, regular, and final observations remain distinguishable in thesis artifacts.

## Run locally

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run
```

The run executes nine training processes sequentially (three algorithms by
three seeds), with each seed isolated in a fresh subprocess.

## Full GCP Batch run: 36-hour timeout

Experiment 1 used 5 hours 8 minutes of Batch running time. Scaling its measured
per-algorithm node rates projects approximately 24 hours for this sequential 5x
job. The command below configures a 129,600-second (36-hour) timeout, providing
12 hours of headroom, on the same `n2-standard-8` class used for Experiment 1.

Push the current commit so the Batch VM can clone it, then run from the
repository root:

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

The final arguments are machine type, maximum runtime in seconds, CPU milli,
memory MiB, and boot-disk size in GiB. `129600` is exactly 36 hours.

Monitor and retrieve the job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## GCP Batch smoke test

This one-seed smoke job runs ESCHER, VR-DeepDCFR+, and VR-DeepPDCFR+ through
the Experiment 2 orchestration and Cloud Storage upload path. It reduces the VR
early-evaluation threshold from the production value of 10,000 nodes to 10
nodes so the smoke output verifies all three checkpoint types: zero-node,
early-threshold, and regular outer-iteration evaluation.

After pushing the current repository, run from the repository root with the
same environment variables used above:

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

Inspect and download the smoke results with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

In `checkpoint_curves.csv`, each VR algorithm should have rows with
`checkpoint_kind` values `initial_untrained_policy`, `early_node_threshold`,
and `outer_iteration`. Smoke-test performance values have no scientific meaning.

## Local fast wiring test

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_5x_nodes.run \
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
  --output-root outputs/smoke_tests
```

This validates the zero-node and lowered early-threshold checkpoints plus the
end-to-end local export path. Smoke-test metrics have no scientific meaning.

## Outputs

The experiment exports `seed_summary.csv`, `aggregate_summary.json`,
`checkpoint_curves.csv`, paired differences against ESCHER, the standard
nodes-touched plots, full metadata, worker logs, and failure records using the
same conventions as Experiment 1 and the earlier thesis repositories.
