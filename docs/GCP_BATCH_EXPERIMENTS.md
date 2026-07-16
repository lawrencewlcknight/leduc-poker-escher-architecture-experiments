# Running Architecture Experiments on Google Cloud Batch

The Batch launcher creates an ephemeral VM, clones this repository, installs the
Python 3.9 environment, runs one command, records resource diagnostics, and
uploads `outputs/` to Cloud Storage even when the experiment fails.

## Prerequisites

Configure a Google Cloud project with Batch, Compute Engine, Cloud Logging, and
Cloud Storage enabled. The Batch service account needs permission to run jobs,
write logs, and upload to the selected bucket.

Set these values in the submitting shell:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
```

The launcher defaults to the expected public GitHub URL for this repository.
Until that remote exists, or when using a fork, override it:

```bash
export REPO_URL="https://github.com/OWNER/leduc-poker-escher-architecture-experiments.git"
```

The VM must be able to clone `REPO_URL`; for a private repository, supply an
appropriately authenticated URL or adapt the clone step using your organisation's
secret-management policy.

## Submit a baseline smoke test

```bash
JOB_NAME="escher-architecture-exp28-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run \
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
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

The final arguments are machine type, maximum runtime in seconds, CPU milli,
memory MiB, and boot-disk size in GiB. Keep the machine specification and runtime
in the experiment record when comparing wall-clock performance.

## Submit the full Experiment 28 baseline

```bash
JOB_NAME="escher-architecture-exp28-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 86400 4000 16000 100
```

Use the same resource specification for paired architecture runs unless resource
scaling is itself part of the experiment.

## Monitor and inspect

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
./gcp/read_batch_task_logs.sh "$JOB_NAME" ERROR Traceback Killed
```

List and download uploaded output:

```bash
gcloud storage ls "$BUCKET/$JOB_NAME/"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

The job output includes `batch_run.log`, periodic resource snapshots, and
`batch_status.json`. Check the exit code and experiment summaries before
promoting thesis artifacts.

## New architecture experiments

Pass the new module command to the same launcher. Keep the repository URL,
machine allocation, output root convention, seeds, and baseline budget unchanged
for like-for-like comparisons. Use unique job and experiment names so cloud
uploads cannot collide.

Delete completed Batch job records when no longer needed; do not delete the
Cloud Storage copy until local outputs and thesis artifacts have been verified.
