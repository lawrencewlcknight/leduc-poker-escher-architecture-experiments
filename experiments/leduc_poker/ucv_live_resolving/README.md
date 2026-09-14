# Experiment 32: UCV-sampled live resolving

## Question

Can UCV sampling deliver a better exploitability--online-compute trade-off, or
lower stochastic variability, than otherwise matched baseline-free sampling
and the full-tree CFR+ resolver from Experiment 31?

The five frozen 36-hour Experiment 29 blueprints, trigger point, public-belief
construction and whole-game evaluation are identical to Experiment 31.

## Resolver arms

Both sampled arms retain tabular CFR+ regrets and enumerate every action at a
traverser's information set. At opponent nodes they sample one action from a
full-support mixture containing 5% uniform exploration.

1. **Baseline-free external sampling:** applies the sampling correction without
   a control variate.
2. **UCV external sampling:** uses a running tabular action-value estimate as a
   control variate. The baseline is read before the current target is formed
   and updated afterwards, preserving the estimator's predictability.

This matched baseline-free arm is essential: comparing only against full-tree
CFR+ would confound the effect of UCV with the effect of sampling.

Search budgets are 64 through 16,384 updates in powers of two. These were
chosen to overlap Experiment 31's actual node-count range. Eight paired local
resolver seeds are nested within each of the five blueprint seeds; the five
blueprints, not the 40 resolver runs, remain the inferential units.

As in Experiment 31, this is range-conditioned local re-solving without a safe
resolving gadget. Exact whole-game exploitability is measured and no formal
non-degradation guarantee is claimed.

## Run

Experiment 31 must complete first because its aggregate is imported for the
joined comparison. Run the mandatory local smoke test:

```bash
./gcp/run_ucv_live_resolving.sh smoke-local
```

Then retain the normal cloud variables and identify both source runs:

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

The remote controller owns smoke, search and aggregation, so the laptop may be
disconnected after submission. Five `n2-standard-4` tasks process one blueprint
each. Each task has a six-hour hard limit and no automatic worker retry.

## Outcomes

In addition to the Experiment 31 outputs, aggregation creates joined CFR+,
baseline-free-sampling and UCV-sampling charts by nodes and decision latency.
It reports within-blueprint variability across the eight paired resolver seeds
and retains root-level control-variate correction diagnostics. Every resolved
policy table and its checksum are saved.
