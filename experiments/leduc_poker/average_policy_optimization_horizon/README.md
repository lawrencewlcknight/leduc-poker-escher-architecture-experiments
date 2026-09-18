# Experiment 41: average-policy optimisation horizon

Experiment 41 isolates the simplest successful intervention observed in
Experiment 40: continue fitting the frozen Experiment 29 average-policy network
to its existing replay reservoir without generating new poker experience.

## Research question

How far does the neural average policy continue to improve when the same
iteration-weighted empirical reservoir target receives additional full-batch
soft-target cross-entropy optimisation, and at what point does the improvement
plateau or reverse?

## Frozen design

- Sources: Experiment 29 `time_36h` training states for seeds
  `104729, 130363, 155921, 181081, 205759`.
- Initial network: the archived Experiment 29 average-policy network.
- Optimizer: one fresh Adam optimizer, continued without restart.
- Learning rate: `3e-4`, matching Experiment 40 uniform repair.
- Objective: one equally weighted row per unique replay information state. Its
  target is the iteration-weighted mean action distribution for that state.
- Training checkpoints: 400, 800, 1,200, 1,600 and 2,000 additional full-batch
  updates; all are saved as playable policies.
- Diagnostics: every 100 updates, including update zero.
- No checkpoint controls training or stopping. The complete 2,000-update path
  is run for every seed.

The training path consumes only replay tensors, action targets, legal-action
masks and iteration labels. It does not enumerate the game tree. Exact Leduc
exploitability, the exact tabular average and information-set errors are
post-hoc diagnostics only.

## Outputs

The aggregate analysis reports exploitability, full-batch cross-entropy,
neural-minus-empirical and neural-minus-exact gaps, policy error, paired change
from update zero, and marginal change over each 400-update interval. It creates
three thesis-ready charts:

- `exploitability_by_optimizer_update.png`;
- `distillation_gap_by_optimizer_update.png`;
- `training_objective_by_optimizer_update.png`.

The automatically reported best observed update is descriptive. It is not a
selected confirmatory checkpoint.

## Local smoke test

```bash
./gcp/run_average_policy_optimization_horizon.sh smoke-local
```

The smoke test creates a miniature Experiment 29 source, performs a real
continuous fit, reload-validates both required policy snapshots and completes
aggregation.

## GCP run

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export RUN_ID="exp41-fit-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_average_policy_optimization_horizon.sh run
```

One `n2-standard-4` worker is used per source seed. The remote controller owns
smoke, fitting and aggregation, so the laptop may be disconnected after the
controller has been submitted. Use `status` to inspect the run and `resume` to
reuse completed workers after an infrastructure or aggregation failure.
