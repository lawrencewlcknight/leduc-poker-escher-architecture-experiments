# Experiment 31: tabular CFR+ live resolving

## Question

Does public-state forward search improve the deployed 36-hour UCV-ESCHER
blueprint when the continuation is solved by an established full-tree tabular
CFR+ resolver?

This is a post-selection development experiment. It reuses the five promoted
UCV cross-entropy policies from Experiment 29 and performs no blueprint
training.

## Intervention

The resolver is triggered at the start of Leduc's second betting round. For a
given public card and first-round action sequence it reconstructs every
compatible private deal and weights it by its chance probability and reach
under the frozen blueprint. CFR+ then solves that public-belief continuation
to terminal. Before the trigger the agent follows the neural blueprint; after
the trigger both players use the resolved tabular continuation.

The nine budgets are 1, 2, 4, 8, 16, 32, 64, 128 and 256 complete CFR+
updates. The experiment records actual nodes and latency at every public root,
so conclusions do not rely on treating an iteration as a hardware-independent
unit of work.

This implementation is range-conditioned but does **not** implement a safe
resolving opt-out gadget or opponent counterfactual-value constraint. It must
therefore not be described as theoretically safe. Exact whole-game
exploitability tests whether stitching the local policies makes each frozen
blueprint better or worse in practice.

## Replication and outcomes

- Five independent Experiment 29 blueprint seeds are the inferential units.
- CFR+ is deterministic; it is run once per blueprint and budget.
- The primary curve is exact exploitability against total online nodes.
- Secondary outputs include median and 95th-percentile decision latency,
  exact seat-averaged value against the unmodified blueprint, root-level work,
  and reloadable resolved-policy tables.

## Run

Run the local smoke test first:

```bash
./gcp/run_tabular_cfr_live_resolving.sh smoke-local
```

Then set the standard cloud variables plus the completed Experiment 29 run:

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

The controller runs entirely in Google Cloud after submission. Five
`n2-standard-4` tasks process one blueprint each. The hard task limit is three
hours, although Leduc resolving is expected to complete much sooner.

## Outputs

The `analysis/` directory contains:

- `aggregate_summary.csv` and `seed_level_curves.csv`;
- `exploitability_by_online_nodes.png`;
- `exploitability_by_decision_latency.png`;
- root-level node and latency diagnostics;
- source hashes and the frozen protocol manifest.

Each worker also stores its resolved tabular policies as JSON so every reported
point remains playable and independently checkable.
