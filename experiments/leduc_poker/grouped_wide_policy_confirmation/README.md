# Experiment 35: fresh grouped-wide policy confirmation

Experiment 35 is the confirmatory follow-up to Experiment 34. It integrates
the selected average-policy fitting process into UCV-ESCHER and trains five
new trajectories for 36 active hours each on seed labels that were not used by
Experiments 19--34.

The frozen candidate retains the Experiment 29 regret and critic system and
changes only average-policy fitting:

- one iteration-weighted sufficient-statistic target per observed information
  set;
- a shared three-layer policy network with 136 hidden units per layer;
- soft-target cross-entropy, learning rate `0.003`, and 20,000 updates per fit;
- reset-from-scratch fitting at every checkpoint.

Every outer checkpoint also fits the former row-wise `3 x 64` policy head for
5,000 updates from the **same frozen reservoir**. This is a paired diagnostic,
not another training trajectory. Its wall-clock cost and all exact Leduc
diagnostics are excluded from the 36-hour active-training clock, and checkpoint
RNG restoration prevents them from changing subsequent UCV samples.

## Design and outputs

The five fixed production seeds are `470892`, `385626`, `145871`, `902492`,
and `318362`. Playable candidate checkpoints are saved at the first completed
iteration crossing every two-hour threshold, at 15 million nodes, and at 36
hours. Full continuation states are saved at 24 and 36 hours.

The aggregate analysis reports candidate, paired legacy fit, empirical
reservoir policy, and exact tabular average exploitability. It also imports the
frozen Experiment 29 trajectories for Deep CFR, original UCV-ESCHER,
simplified UCV, and the revised cross-entropy UCV candidate. Those four use the
Experiment 29 seed cohort and are therefore independent-sample historical
comparators, not paired confirmation observations. It creates:

- `checkpoint_policy_metrics.csv` and `checkpoint_policy_summary.csv`;
- `paired_policy_effects.csv` and `paired_policy_summary.csv`;
- `late_window_metrics_by_seed.csv`;
- `all_algorithm_checkpoint_policy_metrics.csv` and its summary;
- `unpaired_historical_comparisons.csv` using Welch independent-sample
  comparisons;
- all-algorithm exploitability-by-time and exploitability-by-nodes charts;
- separate average-policy diagnostic charts;
- a paired policy-gap chart and an explicit confirmation decision.

The primary outcomes are the 36-hour candidate exploitability, its
neural-minus-exact gap, and the paired candidate-minus-legacy effect. Because
there are five fresh seeds, the pre-specified directional sign test reaches
`p=0.03125` only if all five effects favour the candidate; the smallest
two-sided exact sign-flip value is `0.0625`.

## Run

From the repository root, run the complete local smoke first:

```bash
./gcp/run_grouped_wide_policy_confirmation.sh smoke-local
```

After pushing the tested commit, reuse the existing GCP configuration:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp35-confirm-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_grouped_wide_policy_confirmation.sh run
```

The cloud controller performs cloud smoke, the five-worker training array, and
aggregation. The laptop may be disconnected immediately after submission.
Status and resume commands are:

```bash
./gcp/run_grouped_wide_policy_confirmation.sh status
./gcp/run_grouped_wide_policy_confirmation.sh resume
```

The production array uses five on-demand `n2-standard-8` VMs. The nominal
training budget is 180 N2 VM-hours; allow roughly 38--45 elapsed hours for
iteration granularity, setup, checkpoint persistence, and final aggregation.
Every training attempt has a hard 54-hour Batch runtime ceiling. Task metadata
is filtered through an explicit marker and checked against the expected task
name before training, preventing incidental import output from entering a GCS
object name. The 24- and 36-hour continuation uploads make five attempts with
bounded backoff, and Batch allows one automatic task retry. A retry normally
restores the durable 24-hour state and completes only the remaining active
training. In the theoretical worst case where no state reached GCS, one retry
could repeat a task from the beginning; no task can receive more than that one
automatic retry.

## Download analysis

After aggregation succeeds:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp --recursive \
  "$BUCKET/$RUN_ID/analysis/*" \
  "cloud_outputs/$RUN_ID/analysis/"
```

Download worker checkpoints or full continuation states only if a later audit
or resume requires them; they are not needed to interpret the aggregate
results.
