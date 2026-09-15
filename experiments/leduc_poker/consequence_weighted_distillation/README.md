# Experiment 33: consequence-proxy selection and weighted distillation

## Research question

Can a scalable measure of strategic consequence identify the average-policy
errors that matter for exploitability, and can that measure improve neural
average-policy fitting without changing UCV regret learning?

This is an offline architecture-development experiment. It reuses the five
Experiment 29 promoted-candidate trajectories at 24 and 36 hours and the exact
single-information-set repair labels produced by Experiment 30. It does not
rerun poker training.

## Evidence separation

The workflow has a mandatory stage boundary:

1. proxy workers calculate counterfactual reach, exact-average action-value
   span and expected absolute advantage;
2. the proxy is selected using only the 24-hour checkpoint;
3. the selected proxy identifier and complete ranking are written to a
   checksummed `selected_proxy.json`;
4. redistillation workers refuse to run without that artifact; and
5. 36-hour results provide the temporally held-out primary evaluation.

The selectable proxies contain no neural policy error or exact repair gain.
Policy-error products use the empirical reservoir teacher and are reported as
diagnostic rankings, while exact repair
gain is used only as an oracle label and in a clearly marked non-scalable upper
bound arm. This prevents the deployable consequence score from containing its
own evaluation target.

The locked selection rule maximises mean positive single-repair gain captured
by the top 10% of information sets at 24 hours. Mean Spearman association and
then proxy identifier provide deterministic tie breaks.

## Redistillation arms

All fits use the original network, optimiser, reservoir, iteration weighting
and 5,000-step soft-target cross-entropy budget unless explicitly stated.
Three optimiser replicates are nested inside each source trajectory.

- standard soft-target cross-entropy;
- proxy-prioritised sampling with exact importance correction, preserving the
  original empirical objective;
- clipped consequence-weighted cross-entropy;
- clipped policy-error by consequence-weighted cross-entropy;
- standard cross-entropy followed by 2,500 targeted fine-tuning steps with 25%
  ordinary-batch rehearsal; and
- exact positive repair-gain weighting as a Leduc-only diagnostic upper bound.

The action-value quantities are enumerated exactly in Leduc to test the proxy
without critic noise; the same quantities can be approximated by a value critic
in a larger game. Weights use exponent `0.5`, are clipped around one to
`[0.25, 4.0]`, and are
renormalised. The primary outcome is the paired 36-hour neural-minus-exact
exploitability gap. Secondary outcomes include exact exploitability, error by
information set, 24-hour effects, reach-weighted divergence, effective sample
fraction, fit time and seed consistency. Source trajectory—not optimiser
replicate—is the inferential unit.

## Local smoke test

From the repository root:

```bash
./gcp/run_consequence_weighted_distillation.sh smoke-local
```

The smoke creates tiny compatible Experiment 29 states, runs the real
Experiment 30 repair oracle, locks a proxy, fits every arm at both endpoints,
reload-validates every saved policy, aggregates the results and renders both
charts. Smoke outputs default to
`/tmp/consequence-weighted-distillation-smoke`.

## Google Cloud run

The service account needs the same Batch, storage and logging permissions used
for Experiments 29 and 30. Set the completed source run identifiers explicitly:

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

The remote controller runs cloud smoke, five proxy tasks, proxy selection, five
redistillation tasks and final aggregation in order. The laptop may be shut
down after controller submission. Both array stages use on-demand
`n2-standard-8` VMs with four-hour per-task hard limits; the expected complete
elapsed time is approximately 1.5--2.5 hours and the expected budget is 7--12
N2 VM-hours.

Status and resumable recovery use the original `RUN_ID`:

```bash
./gcp/run_consequence_weighted_distillation.sh status
./gcp/run_consequence_weighted_distillation.sh resume
```

## Principal outputs

The final `analysis/` directory contains:

- `aggregate_summary.json` and `aggregate_manifest.json`;
- `proxy_repair_metrics.csv` and `proxy_validation_36h.png`;
- `fit_metrics.csv`, `seed_level_metrics.csv`, paired effect tables and
  `distillation_arm_comparison_36h.png`;
- information-set errors, sampling diagnostics and source decomposition;
- a worker manifest and complete policy inventory.

The earlier `selection/selected_proxy.json` records the immutable selection
decision and should be retained with the analysis.
