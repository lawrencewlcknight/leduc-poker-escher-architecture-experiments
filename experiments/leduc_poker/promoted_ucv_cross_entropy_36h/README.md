# Experiment 29: promoted UCV cross-entropy candidate at 36 hours

## Question and evidence status

Experiment 29 asks whether the complete architecture promoted by the staged
development studies improves long-horizon deployed-policy performance. The
candidate combines the Experiment 23 non-predictive fast core, Experiment 25's
four-fit temporally averaged critic target, and Experiment 27's
iteration-weighted soft-target cross-entropy average-policy objective. It uses
the ordinary empirical reservoir sampler retained after Experiment 28. The
average-policy network is reset at every evaluation fit, matching the stronger
pre-specified Experiment 27 regime without adding its inconclusive warm start.

The run reuses Experiment 21 seeds `104729`, `130363`, `155921`, `181081` and
`205759` to support paired trajectory comparisons. Deep CFR and Original UCV
are imported from Experiment 21 and Simplified UCV from Experiment 24; none is
retrained. Because all five labels and the architecture have already informed
development, this is post-selection paired development evidence rather than a
new held-out confirmation.

## Frozen execution contract

Five independent on-demand `n2-standard-8` workers train the revised candidate
for 36 active hours. Each saves a playable policy at the first completed
iteration crossing every two-hour threshold and at the first iteration
crossing 15 million nodes. Full continuation state is saved and durably
uploaded at 24 and 36 hours. Snapshot and exact diagnostic time is excluded
from the active-training clock. The 800-iteration cap and 54-hour Batch limit
are safety ceilings, not stopping targets.

The primary outcomes are mean exact exploitability over 24--36 hours and exact
exploitability at 36 hours, with the Simplified UCV trajectory as the principal
reference. Secondary outcomes include checkpoint RMSSD, late slope and worst
rebound, nodes touched, the 15-million-node endpoint, final exact same-seed
head-to-head value, exact tabular-average exploitability, and the neural
distillation gap. Deep CFR and Original UCV provide contextual references.
Training seed is the inferential unit; exact sign-flip tests and paired
intervals are reported. With five seeds, the smallest attainable two-sided
sign-flip p-value is `0.0625`. Head-to-head p-values are Holm-adjusted across
the three references.

## Mandatory local smoke test

From the repository root:

```bash
export SMOKE_OUTPUT="/tmp/exp29-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_promoted_ucv_cross_entropy_36h.sh smoke-local
```

Set `DEEP_CFR_REPO` only if the sibling Deep CFR repository is not in its usual
location. The smoke constructs tiny Experiment 21 and 24 references, trains
the revised candidate, verifies every snapshot, checks the promoted
configuration and cross-entropy objective, round-trips full continuation
state through an additional iteration, runs exact exploitability and
head-to-head evaluation, and renders all four principal charts. Smoke values
have no scientific meaning.

## Fully remote GCP run

Commit and push this implementation before setting `REPO_REF`. Reuse the
existing project, region, bucket and service account:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export EXP21_RUN_ID="exp21-36h-20260830-141641"
export EXP24_RUN_ID="exp24-selected-20260905-222025"
export RUN_ID="exp29-ce-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_promoted_ucv_cross_entropy_36h.sh run
```

The command submits a remote controller and returns. The laptop may then be
disconnected. The controller runs a clean cloud smoke, launches production
only after smoke succeeds, imports the frozen comparator archives, performs
aggregation and uploads the analysis. Full parallelism requires 40 regional
N2 vCPUs. Budget approximately 190--205 N2 VM-hours and allow about 38--42
elapsed hours plus provisioning. Each training task has no automatic retry;
use the same `RUN_ID` with `resume` to reuse its latest valid state.

Operational commands are:

```bash
./gcp/run_promoted_ucv_cross_entropy_36h.sh status
./gcp/run_promoted_ucv_cross_entropy_36h.sh resume
./gcp/run_promoted_ucv_cross_entropy_36h.sh dry-run
```

## Download and reproduce analysis

Download only the compact analysis when training completes:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp -r "$BUCKET/$RUN_ID/analysis/*" \
  "cloud_outputs/$RUN_ID/analysis/"
```

For a full local reaggregation, also download Experiment 29 `workers/` and
retain the local Experiment 21 and 24 archives:

```bash
python -m experiments.leduc_poker.promoted_ucv_cross_entropy_36h.run aggregate \
  --output-root "cloud_outputs/$RUN_ID" \
  --experiment-21-root \
    "cloud_outputs/leduc-escher-arch-exp21-36h-20260830-141641" \
  --experiment-24-root \
    "cloud_outputs/leduc-escher-arch-exp24-selected-20260905-222025"
```

Principal outputs are `exploitability_by_training_time.png`,
`exploitability_by_nodes_touched.png`, `average_policy_diagnostics.png`,
`late_window_performance_stability.png`,
`combined_checkpoint_policy_metrics.csv`,
`paired_candidate_comparisons.csv`, and
`final_same_seed_head_to_head.csv`.
