# Experiment 34: Rao--Blackwellised, structure-aware policy distillation

## Question

Can the Experiment 29 average-policy approximation be improved by replacing
repeated, varying replay targets with their exact information-set conditional
mean, by increasing capacity, or by preventing interference between player and
betting-round subproblems?

This is an offline fitting experiment. It reuses the frozen 24-hour and 36-hour
training states from Experiment 29 and does not repeat UCV regret learning.

## Design

The six deployable arms form a `2 x 3` factorial:

| Architecture | Individual replay rows | Grouped information-set targets |
|---|---|---|
| Current shared `3 x 64` MLP | current-contract baseline | aggregation only |
| Shared `3 x 136` MLP | capacity control | aggregation plus capacity |
| Four routed `3 x 64` experts | routing only | aggregation plus routing |

The dense `3 x 136` network has 41,891 parameters. The four experts together
have 41,996 parameters, so their comparison separates conditional routing from
total capacity. The routing variable is observed player crossed with observed
betting round; no learned gate or game-tree model is used.

For replay rows `(I, q_j, w_j)`, grouped targets are

```text
q_bar(I) = sum_j w_j q_j / sum_j w_j
W(I)     = sum_j w_j
```

and therefore preserve the complete empirical soft-target cross-entropy
objective exactly. The worker verifies this identity before fitting.

### Development and validation separation

- The five Experiment 29 source trajectories are the inferential units.
- The 24-hour states are used to choose learning rate and training budget.
- Candidate learning rates are `0.0003`, `0.001`, and `0.003`.
- Candidate budgets are 5,000 and 20,000 optimiser updates. A single run to
  20,000 updates supplies both checkpoints.
- There are three independent fitting replicates per source trajectory.
- The baseline is not tuned: it remains the current shared row-wise network at
  learning rate `0.001` and 5,000 updates.
- Configurations are frozen before fitting the 36-hour states.
- The 36-hour stage also fits each architecture to the exact tabular-average
  teacher. Those arms diagnose capacity but are not deployable candidates.

The primary outcome is exact Leduc exploitability (`NashConv / 2`). The
analysis also reports the neural-minus-exact gap, per-information-set error,
player-by-round losses, group-gradient cosine similarities, fitting variance,
three-model ensembles, parameter count, runtime, and a predeclared promotion
assessment.

## Outputs

The final `analysis/` directory contains:

- `aggregate_summary.json` and `aggregate_manifest.json`;
- `factorial_exploitability_36h.png`;
- `exact_teacher_capacity_36h.png`;
- development configuration and selection tables;
- seed-level and replicate-level validation metrics;
- paired effects and factorial contrasts;
- gradient, group, and information-set diagnostics;
- ensemble results and the promotion assessment; and
- a checksum-verified policy inventory.

## Local smoke test

From the repository root:

```bash
./gcp/run_rao_blackwellised_policy_distillation.sh smoke-local
```

The smoke test creates tiny Experiment 29 states and exercises development,
selection, locked validation, exact-teacher fitting, checkpoint reloads, plots,
and aggregation.

## Google Cloud run

Use the same project, region, bucket, and Batch service account as Experiments
29--33. `EXP29_RUN_ID` must identify the completed Experiment 29 run containing
both 24-hour and 36-hour training states.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp34-rb-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5

./gcp/run_rao_blackwellised_policy_distillation.sh smoke-cloud
```

After the cloud smoke job succeeds, submit the cloud-owned controller:

```bash
./gcp/run_rao_blackwellised_policy_distillation.sh run
```

The controller runs the five development workers, freezes the selection, runs
the five validation workers, and aggregates the results. The laptop may be
disconnected after the controller is submitted.

Allow approximately 3--5 elapsed hours with `PARALLELISM=5`. The expected
production consumption is approximately 12--20 `n2-standard-8` VM-hours,
plus short smoke, controller, selection, and aggregation jobs on smaller VMs.
The Batch specifications cap each development worker at six hours and each
validation worker at four hours.

Check status with:

```bash
./gcp/run_rao_blackwellised_policy_distillation.sh status
```

Resume a failed stage without repeating completed stages with:

```bash
./gcp/run_rao_blackwellised_policy_distillation.sh resume
```

## Download the analysis

```bash
mkdir -p "cloud_outputs/leduc-escher-arch-$RUN_ID/analysis"
gcloud storage cp -r \
  "$BUCKET/$RUN_ID/analysis/*" \
  "cloud_outputs/leduc-escher-arch-$RUN_ID/analysis/"
```

The complete validation policies remain in Cloud Storage under
`$BUCKET/$RUN_ID/validation_workers/`; they need not be downloaded for the
thesis analysis.
