# Experiment 27: isolated average-policy redistillation

## Research question

Experiment 25 found that the exact reach- and iteration-weighted tabular
average had late-window exploitability of approximately `0.027--0.033`, while
the fitted neural average policy remained near `0.070--0.078`. Experiment 27
asks whether that gap is caused by the replay sample, the probability-regression
objective, or resetting the average-policy network at every fit.

This is an offline development experiment. It does not rerun game traversal,
the UCV estimator, critics or regret networks. It reads the frozen 24- and
36-hour continuation states from Experiment 25's averaged-critic-target arm and
redistils alternative policy heads from their identical average-policy
reservoirs.

## Frozen design

The three Experiment 25 development seeds are the independent units. Each
source trajectory is fitted with three optimiser seeds, which are averaged
within source seed and are not treated as additional experimental units.

At 36 hours the scalable `2x2` design is:

| Arm | Objective | Initialisation |
|---|---|---|
| `reset_mse` | existing iteration-weighted MSE | reset |
| `reset_cross_entropy` | weighted soft-target cross-entropy | reset |
| `warm_mse` | weighted MSE | continue the corresponding 24-hour fit |
| `warm_cross_entropy` | weighted cross-entropy | continue the corresponding 24-hour fit |

All four arms retain the source `3x64` MLP, one-million-observation reservoir,
batch size `2048`, learning rate `0.001`, iteration exponent and 5,000 update
steps. A fifth diagnostic arm fits the same network directly to the exact
tabular average. It tests representational and optimisation capacity only and
is not a deployable, scalable or model-free training procedure.

The analysis separately evaluates the exact tabular average, an empirical
tabular policy reconstructed from the reservoir, and the archived neural
policy. This decomposes data/replay error from neural fitting error.

The primary outcome is neural exploitability minus exact-tabular
exploitability at 36 hours. Secondary outcomes are neural exploitability,
information-set `L1` and KL error, worst information-set error, fitting
variation, runtime, and the loss/warm-start factorial effects. Negative paired
effects are favourable. With three source seeds this remains development
evidence; optimiser repetitions do not increase the inferential sample size.

## Mandatory smoke test

From the repository root:

```bash
export SMOKE_OUTPUT="/tmp/exp27-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_average_policy_redistillation.sh smoke-local
```

The smoke first creates tiny Experiment 25-compatible continuation states,
then loads them through the production state reader, reconstructs both tabular
comparators, fits every arm, reload-validates every saved model by checksum,
aggregates the seed-level unit and renders both charts. Smoke values have no
scientific meaning.

## Fully remote GCP run

Reuse the existing project, region, bucket and service account. Experiment 25
must remain at `exp25-fact-20260906-003836` in that bucket unless
`EXP25_RUN_ID` is explicitly changed.

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export EXP25_RUN_ID="exp25-fact-20260906-003836"
export RUN_ID="exp27-distill-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3

./gcp/run_average_policy_redistillation.sh run
```

The local command returns after the remote controller is accepted. The laptop
may then be disconnected. The controller runs a clean cloud smoke, starts
three independent workers, and aggregates their outputs. Each production
worker downloads only its own two source states, approximately 4.6--5.2 GB,
rather than the complete Experiment 25 archive. Workers use on-demand
`n2-standard-8` instances, a 12-hour hard limit and no automatic retry.

Operational commands are:

```bash
./gcp/run_average_policy_redistillation.sh status
./gcp/run_average_policy_redistillation.sh resume
./gcp/run_average_policy_redistillation.sh dry-run
```

## Outputs

Download the small worker policies and analysis products with:

```bash
mkdir -p "cloud_outputs/$RUN_ID"
gcloud storage cp -r "$BUCKET/$RUN_ID/*" "cloud_outputs/$RUN_ID/"
```

Principal analysis outputs are:

- `distillation_gap_decomposition.png`;
- `distillation_arm_comparison.png`;
- `source_decomposition.csv`;
- `seed_level_metrics.csv`;
- `metric_summary.csv`;
- `paired_vs_reset_mse.csv`;
- `factorial_effect_summary.csv`; and
- `information_set_errors.csv`.
