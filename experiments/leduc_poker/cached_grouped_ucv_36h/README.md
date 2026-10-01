# Experiment 48: cached-critic grouped UCV, 36 hours

This belongs to the **Leduc ESCHER architecture repository**, not standard
ESCHER Experiment 48. It is an integrated efficiency follow-up to Experiment
35, using the per-fit target cache validated by Experiment 47. No historical
results, source policies, or default caching settings are modified.

## Frozen design

- Five fresh training runs, but **reused seed labels**: 470892, 385626, 145871,
  902492, 318362. This is not new held-out confirmation.
- One on-demand n2-standard-8 per seed, five simultaneously; eight Torch
  fitting threads. Sequential collection, not the parallel UCV implementation.
- 36 recorded active hours, stopping at completed outer iterations. The
  engineering iteration cap rises from 800 to 2,000 so it does not prematurely
  truncate the faster run; hitting the cap is an error, not completion.
- Experiment 35's fixed-beta, two-critic, non-predictive core, four-fit target
  averaging, residual calibration, sampling, replay capacities, network
  architectures, losses, optimisers and update budgets are unchanged.
- The only learning-path change is `cache_frozen_critic_targets=True`: targets
  are rebuilt from occupied replay rows at the start of every eligible critic
  fit, and discarded after it. Skipped early fits remain skipped.
- Reset grouped iteration-weighted soft-target CE, shared 136 x 136 x 136
  policy network, 20,000 updates, learning rate 0.003, **every outer iteration**.
  The paired legacy row-wise fit and exact-average diagnostics remain present.

The clock formula is deliberately inherited from Experiment 35: raw solver
time minus snapshot persistence, cumulative factorial diagnostics and paired
legacy/confirmation diagnostics. Candidate policy fitting is charged. Do not
reinterpret this as a newly defined pure optimisation clock or claim every
evaluation operation was excluded. Raw clocks, component times, excluded
durations and worker elapsed time are reported separately. Component timers
labelled inclusive/subset overlap and must not be added together.

## Checkpoints and storage

Playable policies are retained at the first completed iteration crossing
2, 4, ..., 36 active hours and 15 million training nodes: **19 per seed**, 95
total. Multiple thresholds may refer to one completed iteration; actual
counts and times are retained. These are real networks, not interpolated
policies. Each is reload-validated and checksum-protected.

There are **no full training states or retained replay reservoirs**, including
at 24/36 hours. A completed worker can be reused after hash verification. An
interrupted training worker fails clearly on reuse: restarting from its policy
alone would not continue training correctly. Use a new run ID to restart such
a run. The cloud jobs have zero automatic retries, preventing silent costly
retraining. Completed source experiments are never altered.

Metrics, diagnostics and playable policies upload at milestones; a marker is
uploaded before training. A final upload failure is a job failure. Temporary
files and `.pt` archives are excluded from publication. The smoke's small
in-memory equivalence states are never written as training-state artefacts.

## Historical comparison and analysis

`frozen_reference.json` is a small, checksum-pinned extraction of the existing
Experiment 35 and selected **uniform** SD-CFR (Deep CFR repository 29)
analytical outputs. Experiment 35's earlier UCV/Deep CFR comparator curves
are retained too. Source GCS locations and original file hashes are embedded.
Production needs **no historical model or replay downloads**. The extraction
script is provided for audit, not automatic reselection of comparators.

Record the first completed crossing of each seed's original Experiment 35
final iteration and node count separately. `matched_workload.csv` reports
observed time savings. If the workload is not reached by 36 hours, it reports
`reached=False`; it does not extrapolate. The historical workload comparison
does not assume bitwise equality across different hardware/software runs.
`matched_workload_summary.csv` reports five-seed time savings only when all
seeds reach the milestone, avoiding an average biased towards faster runs.

Outputs under `analysis/` include:

- `checkpoint_policy_metrics.csv`, `checkpoint_policy_summary.csv`: neural,
  paired legacy, empirical replay and exact tabular policy diagnostics.
- `paired_policy_effects.csv`, `paired_policy_summary.csv`, policy-gap chart.
- `endpoint_seed_metrics.csv`, `endpoint_summary.csv`: saved 15M crossing and
  36-hour results. Node endpoints are not interpolated.
- Exploitability-by-time and by-node charts, with individual seeds and one
  standard error. Node means interpolate only within each algorithm's common
  observed seed range; time summaries use common crossed thresholds. The node
  chart includes a 15M view. No pre-first-point extrapolation is performed.
- Average-policy diagnostic charts, including distillation gaps in the tables.
- `all_algorithm_checkpoint_policy_metrics.csv`, checkpoint summaries,
  late-window means and adjacent-checkpoint RMSSD on the common two-hour grid.
- `historical_comparisons.csv`: paired seed differences against Experiment 35;
  independent-cohort Welch intervals against the other historical controls.
- `matched_workload.csv`, `component_timings.csv`, complete source provenance,
  snapshot inventory, runtime/commit metadata and aggregation manifests.

All inference is exploratory and seed-level. A successful job is not a claim
of improved exploitability. Historical VM/runtime variation is not controlled
by sharing a machine class. A 1.69x critic-fit speedup does not imply a 1.69x
whole-run speedup. A full 36-hour allocation is still used; the hypothesis is
more training within it, not shorter jobs. No new head-to-head league is run
here; retained networks use the existing policy loader and support later
temporal/cross-algorithm exact evaluation without retraining.

## Smoke and GCP run

Optional local smoke (use the repository's Python environment):

```bash
./gcp/run_cached_grouped_ucv_36h.sh smoke-local
# Or: PYTHON=/path/to/venv/bin/python ./gcp/run_cached_grouped_ucv_36h.sh smoke-local
```

The mandatory cloud smoke repeats three complete cached/uncached iterations,
including the initial and early-node evaluations, with bitwise comparisons of
weights, optimisers, replay, RNG, exact averages and policy evaluations. It
also tests a 2,048-example critic minibatch with a partial cache tail, saves
real playable policies and runs aggregation/reload checks. Production starts
only after this succeeds. The small smoke fixture is not performance evidence.

After committing and pushing the new code, run from the repository root:

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-leduc-poker-dream-results"
# Keep SA_EMAIL set to the existing working experiment service account.
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp48-cache36h-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_cached_grouped_ucv_36h.sh run
```

Reset REPO_REF when switching repositories; an old commit is rejected before
submission. `BUCKET` is the destination (the historical DREAM-named shared
Leduc bucket is intentional). SA_EMAIL must exist, write outputs/logs, create
Batch child jobs and act as its child-job service account. Preflight checks
the standard explicit IAM bindings, not arbitrary custom/inherited roles.

The controller runs cloud smoke -> five training workers -> aggregation. The
laptop can disconnect after submission. Five simultaneous training VMs need
40 available N2 vCPUs. Budget 180 active N2 VM-hours plus diagnostic/setup and
controller overhead; elapsed time exceeds 36 hours. Each training job has a
54-hour hard limit with no automatic retry. Smoke has 2h and aggregation 6h
limits. Controller limit accounts for configured parallelism. Limits are
safety ceilings, not estimates; controller failure does not cancel children.

```bash
./gcp/run_cached_grouped_ucv_36h.sh status
./gcp/run_cached_grouped_ucv_36h.sh dry-run
# Only if training succeeded but aggregation failed; keep RUN_ID/REPO_REF:
./gcp/run_cached_grouped_ucv_36h.sh reaggregate

mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage rsync --recursive \
  "${BUCKET%/}/$RUN_ID/analysis" "cloud_outputs/$RUN_ID/analysis"
```

Tests: `python -m pytest tests/test_cached_grouped_ucv_36h.py`. The end-to-end
test has the existing `smoke` marker; fast checks can use `-m 'not smoke'`.
