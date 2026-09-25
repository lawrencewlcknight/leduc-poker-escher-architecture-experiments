# Experiment 44: Single UCV-ESCHER historical policy

Experiment 44 tests the SD-CFR-inspired hypothesis that part of UCV-ESCHER's
remaining error is caused by compressing its historical strategies into one
average-policy network. The regret and critic learner is held fixed at the
Experiment 43 candidate. The intervention changes only the representation of
the output policy.

## Frozen design

Three fresh development seeds train for 12 effective hours on separate
`n2-standard-8` workers. Immediately before every outer-iteration update, each
worker saves the two current cumulative-regret networks. This is the same
strategy that the exact reach- and iteration-weighted average observes for that
iteration. Evaluation, persistence and policy fitting are excluded from the
active-training clock.

At 2, 4, 6, 8, 10 and 12 hours the worker compares, on the identical learner
trajectory:

- the complete iteration-weighted historical-network mixture (the primary
  `Single UCV-ESCHER` arm);
- bounded weighted historical reservoirs with 16, 32, 64 and 128 slots;
- the ordinary contemporaneous neural average-policy fit;
- the incumbent 1,700-update grouped soft-target cross-entropy refinement;
- the grouped empirical policy-reservoir policy; and
- the exact reach- and iteration-weighted tabular average.

For the full historical mixture, a component is sampled with mass proportional
to `iteration ** gamma`; the corresponding regret policy is then used for the
whole hand. For exact Leduc evaluation this normal-form mixture is converted to
its reach-weighted behavioural equivalent. The worker asserts to numerical
tolerance that it reproduces the solver's independently accumulated exact
average. The bounded arms use independent streaming weighted-reservoir slots;
selection uses only the historical stream and never exploitability.

This is paired post-selection development evidence, not a final confirmatory
benchmark. If the full mixture materially improves on the incumbent neural
policy, the next step is a fresh five-seed 36-hour confirmation. If a bounded
capacity retains the gain, that capacity becomes the scalable deployment
candidate.

## Outputs

Each worker writes the per-iteration regret-network archive, its checksummed
manifest, policy metrics, bounded-reservoir selections, resumable states at 6
and 12 hours, and playable incumbent neural checkpoints. Aggregation produces:

- `combined_policy_metrics.csv`;
- `policy_summary.csv`;
- paired Single UCV-ESCHER versus incumbent effects and summaries;
- `single_ucv_escher_vs_policy_outputs.png`; and
- `bounded_historical_mixture_capacity.png`.

## Smoke test and cloud run

From the repository root:

```bash
./gcp/run_single_ucv_escher_12h.sh smoke-local

export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp44-single-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=3
./gcp/run_single_ucv_escher_12h.sh run
```

The cloud-owned controller runs a real smoke test before submitting the three
production workers and then aggregates the results. The laptop may be
disconnected after submission. Use the same launcher with `status` or `resume`
to inspect or recover the run.
