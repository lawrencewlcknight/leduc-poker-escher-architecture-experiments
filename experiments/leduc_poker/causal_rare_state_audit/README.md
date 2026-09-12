# Experiment 30: causal rare-state audit

## Question and evidence status

Experiment 30 asks whether the residual average-policy distillation gap in the
promoted UCV-ESCHER candidate is disproportionately caused by rare but
strategically consequential information sets. It is a frozen, post-training
diagnostic: it reuses the five Experiment 29 trajectories at 24 and 36 hours,
does no training, and never changes a source checkpoint.

The audit enumerates tabular Leduc exactly. For every information set it
replaces only the deployed neural policy row with the corresponding exact
reach- and iteration-weighted average-policy row, then recomputes exact
exploitability. The difference from the unmodified neural policy is the
single-information-set repair effect. This exact policy surgery measures the
causal effect of that narrowly defined deployment intervention; it does not
claim that rarity caused the neural approximation error or training dynamics.
Exact enumeration is used only for evaluation, so this experiment does not
alter the model-free status of the training algorithm.

## Frozen audit contract

The source states are the Experiment 29 `time_24h` and `time_36h` full
continuation states for seeds `104729`, `130363`, `155921`, `181081` and
`205759`. For every information set the audit records:

- raw and iteration-weighted reservoir frequency;
- exact-average reach weight;
- exact and neural action probabilities;
- policy L1, maximum absolute error and exact-to-neural KL;
- exact exploitability after repairing only that information set;
- the repair effect, decision depth and exact-policy entropy.

It then repairs increasing fractions of information sets in five orders:
rarest first, largest policy KL, largest single-repair gain, a combined
rarity/error/consequence score, and random order. The random comparator uses
20 independently shuffled rankings per source state. Curves report both the
fraction of information sets and reservoir mass repaired, exact hybrid-policy
exploitability, and the fraction of the neural-to-tabular distillation gap
recovered. Complete repair must reproduce exact-average exploitability to
absolute tolerance `1e-12` before a worker may succeed.

Single-state effects are not additive: repairing several policy rows can alter
the best response and interaction between rows. Consequently, cumulative
curves are recalculated exactly for each repaired set rather than constructed
by summing individual effects. The five training trajectories—not information
sets or random permutations—remain the inferential units.

## Mandatory local smoke test

From the repository root:

```bash
export SMOKE_OUTPUT="/tmp/exp30-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_causal_rare_state_audit.sh smoke-local
```

The smoke creates tiny Experiment 29-compatible 24- and 36-hour source roles,
audits every information set, checks complete-repair equivalence, aggregates
the seed and random-ranking hierarchy, and renders all three charts. Smoke
values have no scientific meaning.

## Fully remote GCP run

Commit and push Experiment 30 before setting `REPO_REF`. Reuse the established
project, region, bucket and service account, and provide the run ID containing
the completed Experiment 29 worker archives:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="your-completed-experiment-29-run-id"
export RUN_ID="exp30-audit-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_causal_rare_state_audit.sh run
```

The command submits a remote controller and returns. The laptop may then be
disconnected. The controller runs a clean cloud smoke, launches five
independent `n2-standard-4` audit workers, and aggregates only after all audit
workers succeed. Each worker downloads its own two Experiment 29 continuation
states. No algorithm is retrained.

Operational commands are:

```bash
./gcp/run_causal_rare_state_audit.sh status
./gcp/run_causal_rare_state_audit.sh resume
./gcp/run_causal_rare_state_audit.sh dry-run
```

The 12-hour task limit is a safety ceiling. Runtime should be dominated by
downloading approximately two large continuation states and repeated exact
Leduc best-response evaluations; inspect the cloud smoke before relying on a
more precise cost estimate.

## Outputs and interpretation

Download the compact analysis after completion:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp -r "$BUCKET/$RUN_ID/analysis/*" \
  "cloud_outputs/$RUN_ID/analysis/"
```

Principal outputs are:

- `causal_repair_curves.png`: exploitability and recovered distillation gap by
  the fraction of information sets repaired;
- `repair_mass_efficiency.png`: recovered gap by repaired reservoir mass;
- `rarity_error_consequence.png`: single-state effect against rarity, coloured
  by policy KL;
- `information_set_audit.csv`: the complete intervention-level evidence;
- `repair_curve_summary.csv`: seed-level means and intervals for all curves;
- `correlation_summary.csv`: descriptive seed-level Spearman associations;
- `top_rare_important_information_sets.csv`: the highest combined-score rows;
- `worker_manifest.csv`: source hashes and repository provenance.

Evidence for a rare-state bottleneck requires more than a rarity correlation.
The strongest pattern would be that the combined or consequence ranking
recovers substantially more of the gap than random repair after changing a
small fraction of information sets and a small fraction of reservoir mass,
consistently across source seeds and at both 24 and 36 hours. If only the
largest-error ranking helps, the result supports targeted distillation but not
the stronger claim that rarity identifies the failures. If rarest-first does
not beat random, rarity-aware training should not be promoted on this evidence.
