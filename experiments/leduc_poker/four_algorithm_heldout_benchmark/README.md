# Frozen four-algorithm held-out benchmark

## Purpose and confirmatory contract

This experiment compares Deep CFR, VR-DeepDCFR+, VR-DeepPDCFR+, and
UCV-ESCHER on eight new Leduc training seeds. It trains every algorithm for the
same eleven-hour active wall-clock horizon and also captures the first completed
iteration crossing 15,000,000 training nodes.

The production seed tuple is frozen in `config.py`:

```text
104729, 130363, 155921, 181081, 205759, 230969, 256019, 281117
```

The runner refuses a production invocation with any other seed set or order.
The smoke test uses seed `0` and refuses held-out seeds. Do not inspect
production results and then change the algorithms, configurations, endpoints,
primary comparisons, or exclusion rules.

The algorithm configurations are imported from the selected prior
configurations rather than restated:

- Deep CFR uses the selected final-candidate configuration, extended only by a
  6,000-iteration safety cap;
- both VR algorithms use the prior paper-parameterised comparison configuration,
  extended only by a 500-outer-iteration safety cap;
- UCV-ESCHER uses its selected comparison configuration, also with a
  500-outer-iteration safety cap.

Those caps are failure guards, not experimental budgets. A worker stops only
after both required policies have been saved. If eleven hours is reached before
15 million nodes, it continues until the node snapshot exists.

## Endpoint semantics

Both endpoints are defined at completed algorithm iterations; training is never
interrupted halfway through an update.

- `node_15m`: the policy available after the first completed iteration with at
  least 15,000,000 recorded training-node touches;
- `time_11h`: the policy available after the first completed iteration whose
  cumulative active training wall time is at least 39,600 seconds.

Consequently, both endpoints can overshoot their threshold slightly. The exact
completed iteration, observed nodes, observed seconds, frozen configuration,
repository commit, file size, and SHA-256 digest are recorded in the snapshot
and worker manifest. Deep CFR's average-policy network retains its selected
ten-iteration fitting cadence; an endpoint does not introduce an extra fit.

Every endpoint is a playable policy snapshot, not just a metric row. Before a
worker can write `SUCCESS.json`, it reloads both files and exhaustively converts
them to OpenSpiel tabular policies. The aggregate stage verifies every digest
again before evaluation.

## Segmentation and expected duration

The production job is one Batch task group with 32 stable array tasks:

```text
4 algorithms × 8 seeds = 32 independent n2-standard-8 VMs
```

`taskCountPerNode=1` prevents co-location, and `BATCH_TASK_INDEX` maps to a
single algorithm/seed pair. With the default parallelism of 32, all training
trajectories run concurrently. The estimated successful training allocation is
352 VM-hours (32 × 11 hours), approximately 360 VM-hours after bootstrap and
endpoint overshoot. A prudent budget is roughly 390 VM-hours. Allow 12–14 hours
of elapsed time for production plus up to two hours for preflight and exact
aggregation. Reducing `PARALLELISM` lowers quota pressure but not successful
VM-hours: 16-way parallelism is expected to take roughly 24–28 elapsed hours.

Standard VMs are the default. Spot is optional, but a preempted worker restarts
its training trajectory because only endpoint policies—not full optimiser and
replay-buffer states—are persisted. Completed task archives are reused by
checksum when the same `RUN_ID` is resumed.

## Mandatory smoke test

From this repository, with the sibling Deep CFR repository present:

```bash
./gcp/run_four_algorithm_heldout_benchmark.sh smoke-local
```

This tiny run executes all four training implementations, saves both endpoint
formats, reloads all eight snapshots, recomputes exploitability, and completes
both exact head-to-head pipelines. Its numerical results have no scientific
meaning. It normally takes about one minute after dependencies are installed.

The default cloud `run` command submits a lightweight remote controller and
returns. The controller submits a one-VM cloud smoke job and waits for it inside
Google Cloud. Production is not submitted if smoke fails. This additionally
tests repository checkout, clean-environment dependency installation,
service-account permissions, and Cloud Storage upload.

## One-command cloud run

First commit and push these experiment files in the architecture repository.
Use immutable commit SHAs for both repositories:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-results-bucket/heldout-benchmarks"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export ARCH_REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export RUN_ID="leduc-heldout-$(date -u '+%Y%m%d-%H%M%S')"

./gcp/run_four_algorithm_heldout_benchmark.sh run
```

That single command submits one remote controller job and returns as soon as
Google Cloud accepts it. The laptop can then be closed, disconnected, or
switched off. The controller submits and waits for, in order:

1. the mandatory cloud smoke job;
2. the 32-task production training job;
3. the exact aggregation and head-to-head job.

The controller uses the configured service account to create the child jobs.
Consequently, `SA_EMAIL` needs `roles/batch.jobsEditor` on the project and
`roles/iam.serviceAccountUser` on itself, as well as permission to write Cloud
Logging entries and read/write the chosen bucket. An administrator can grant
the two controller-specific roles with:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/batch.jobsEditor"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --project="$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser"
```

The project also needs sufficient regional quota for 256 concurrent N2 vCPUs
when `PARALLELISM=32`. The controller itself uses one small standard `e2-small`
VM and remains alive only to submit and monitor the child jobs.

Useful controls:

```bash
# Generate the controller and three child Batch JSON files without submitting.
./gcp/run_four_algorithm_heldout_benchmark.sh dry-run

# Inspect jobs created with an existing RUN_ID.
./gcp/run_four_algorithm_heldout_benchmark.sh status

# Submit a remote recovery controller; completed workers are skipped.
./gcp/run_four_algorithm_heldout_benchmark.sh resume

# Lower simultaneous quota demand.
PARALLELISM=16 ./gcp/run_four_algorithm_heldout_benchmark.sh run

# Optional, interruption-prone cheaper capacity.
PROVISIONING_MODEL=SPOT MAX_RETRIES=2 \
  ./gcp/run_four_algorithm_heldout_benchmark.sh run
```

Job names and artifacts share `RUN_ID`. Results are written below:

```text
gs://.../RUN_ID/
  smoke/
  workers/
    task_000_deep_cfr_seed_104729/
      worker_result.json
      SUCCESS.json
      snapshots/
    ... 31 further task directories ...
  analysis/
    worker_manifest.csv
    aggregate_manifest.json
    node_15m/
    time_11h/
```

## Exact policy evaluation

No poker hands are sampled. At each endpoint, the primary analysis evaluates
all six unordered algorithm pairs in both seats for every paired training seed
(48 paired effects). The inferential unit is the training seed. It reports
effect sizes, 95% t intervals, exact two-sided paired sign-flip tests, and Holm
adjustment across the six pairwise tests.

The secondary descriptive league evaluates all 64 seed combinations for every
algorithm pair (384 exact matchups per endpoint). Those combinations are not
treated as independent observations. Endpoint exploitability and self-play
value are also recomputed from each archived policy.

Principal endpoint outputs are:

| File | Contents |
| --- | --- |
| `snapshot_inventory.csv` | Policy paths, endpoint observations, commits and checksums. |
| `endpoint_policy_metrics.csv` | Exact exploitability and self-play value. |
| `head_to_head_same_seed_pairwise.csv` | Primary six pairs × eight seeds, both seats. |
| `head_to_head_pairwise_inference.csv` | Seed-level intervals, exact tests and Holm adjustment. |
| `head_to_head_cross_seed_league.csv` | Secondary six pairs × 64 seed combinations. |
| `algorithm_strength_summary.csv` | Descriptive mean exact EV against the other algorithms. |
| `head_to_head_mean_ev_heatmap.png` | Antisymmetric paired-seed mean-EV matrix. |
| `endpoint_exploitability.png` | Mean endpoint exploitability with seed-level standard errors. |

To repeat aggregation after downloading the `workers/` directory:

```bash
python -m experiments.leduc_poker.four_algorithm_heldout_benchmark.run aggregate \
  --output-root /path/to/downloaded/RUN_ID
```
