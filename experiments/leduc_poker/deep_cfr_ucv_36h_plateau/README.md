# Experiment 21: Deep CFR and UCV-ESCHER 36-hour convergence

## Research question and frozen contract

This follow-on asks whether Deep CFR and UCV-ESCHER have reached a practical
exploitability floor by 36 hours, and whether their conclusions differ when
training effort is expressed as active time or nodes touched. It is a
convergence study, not a new algorithm-selection exercise.

The production contract is frozen in `config.py`:

- algorithms: Deep CFR and UCV-ESCHER only;
- seeds: `104729`, `130363`, `155921`, `181081`, and `205759`;
- active training horizon: 36 hours per algorithm/seed trajectory;
- policy checkpoints: first completed iteration crossing 2, 4, ..., 36 hours;
- machines: one standard `n2-standard-8` VM per trajectory;
- task map: two algorithms by five seeds, giving ten stable Batch tasks.

These are the first five seeds in Experiment 19's immutable ordering. They were
selected by position, not by inspecting which Experiment 19 runs were
favourable. Both algorithms reuse their frozen Experiment 19 configurations.
Only the iteration safety caps are enlarged so that time, rather than the cap,
remains the stopping rule.

The seed is the inferential unit. Five seeds provide useful trajectory and
effect-size evidence, but the smallest possible two-sided exact sign-flip
p-value is 0.0625. Conclusions should therefore emphasize effect sizes,
confidence intervals, consistency across seeds and curve shape rather than a
binary significance threshold.

## Timing and checkpoint semantics

Training is never interrupted midway through an algorithm iteration. For each
two-hour target, the worker saves the policy available after the first
completed iteration whose cumulative active training time crosses that target.
The manifest records the target time, observed time, overshoot, completed
iteration and nodes touched.

Snapshot serialization and validation time are excluded from active training
time. Exact exploitability is evaluated later from the archived playable
policies. This design ensures that:

- every trajectory receives 36 hours of actual training;
- evaluation cost cannot alter either learner's training budget;
- all plotted values can be independently reproduced from retained policies;
- nodes-touched curves use the node count observed at the same policies as the
  time curves.

Deep CFR checkpoints preserve the selected average-policy network and do not
force an additional policy-network fit. UCV-ESCHER checkpoints preserve its
playable average policy. Every worker reloads and exhaustively tabularises all
of its snapshots before writing `SUCCESS.json`; aggregation verifies every
SHA-256 digest again.

Only playable policy checkpoints are stored. They do not include sufficient
optimizer and replay-buffer state to resume a partially completed training
trajectory. A failed or preempted task therefore restarts, whereas a fully
completed task is reused after commit, contract and checksum validation.

## Exact analysis and interpretation

OpenSpiel enumerates Leduc exactly; no poker hands are sampled for policy
evaluation. The primary longitudinal outputs are:

| File | Contents |
| --- | --- |
| `checkpoint_policy_metrics.csv` | Exact exploitability (`NashConv / 2`), self-play value, observed time, overshoot, nodes and iteration for every policy. |
| `checkpoint_summary.csv` | Per-algorithm means, standard deviations, standard errors and 95% t intervals at each target time. |
| `exploitability_by_training_time.png` | Five faint seed trajectories plus the mean and 95% interval against active hours. |
| `exploitability_by_nodes_touched.png` | The same evaluated policies plotted against observed nodes touched. |
| `late_window_changes_by_seed.csv` | Per-seed exploitability improvements over 24--30, 30--36 and 24--36 hours. |
| `late_window_change_summary.csv` | Effect summaries and exact sign-flip tests for those late windows. |

The nodes plot is observational: at each two-hour checkpoint, different seeds
and algorithms can have different node counts. It is valuable for diagnosing
throughput and sample efficiency, but it is not a matched-node causal
comparison at identical budgets.

The final 36-hour policies are also compared head to head in both seats. The
five same-seed effects are the inferential observations;
`final_cross_seed_head_to_head.csv` contains all 25 cross-seed matchups only as
a descriptive league. These final comparisons are secondary to the requested
convergence curves.

A genuine practical floor would require small and inconsistent improvements in
the late windows, not merely a visually flat aggregate curve. Conversely, a
downward 30--36-hour trend, especially when repeated across seeds, is evidence
that the chosen horizon has not yet located the floor. The experiment cannot
prove asymptotic convergence.

## Mandatory local smoke test

Use the repository's Python 3.9 environment and keep the Deep CFR repository at
the normal sibling path:

```text
deep_cfr_v3/
  leduc_poker_escher_architecture/leduc-poker-escher-architecture-experiments/
  leduc_poker_deep_cfr/leduc-poker-deep-cfr-experiments/
```

Then run:

```bash
./gcp/run_deep_cfr_ucv_36h_plateau.sh smoke-local
```

The smoke uses development seed `0`, not a production seed. It executes both
training implementations with tiny budgets, creates three smoke checkpoints
per algorithm, reloads all six policies, computes exact exploitability and
head-to-head values, and renders both charts. Its numerical values have no
scientific meaning.

## One-command GCP run

Experiment 21 reuses the Experiment 19 remote-controller service-account
permissions. Commit and push the architecture repository first, then set:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west2"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export ARCH_REPO_REF="$(git rev-parse HEAD)"
export DEEP_CFR_REPO_REF="a7459be458650a1fe02db72f8456c97c9eefdc25"
export RUN_ID="exp21-36h-$(date -u '+%Y%m%d-%H%M%S')"

./gcp/run_deep_cfr_ucv_36h_plateau.sh run
```

The launcher returns after the remote controller is accepted. The laptop can
then be disconnected or switched off. The controller submits, waits for and
checks the clean cloud smoke, ten-task production run and exact aggregate job.
`status`, `resume` and `dry-run` are also supported.

At ten-way parallelism the experiment needs 80 concurrent N2 vCPUs and should
finish in roughly 37--40 elapsed hours after provisioning. The successful
training allocation is exactly 360 VM-hours before iteration overshoot and
bootstrap. Budget approximately 375--390 VM-hours for the full workflow. A
50-hour hard limit applies independently to every production task.

Artifacts are written to:

```text
gs://.../RUN_ID/
  smoke/
  workers/
    task_000_deep_cfr_seed_104729/
      worker_result.json
      SUCCESS.json
      snapshots/                 18 playable policies
    ... nine further task directories ...
  analysis/
    aggregate_summary.json
    checkpoint_policy_metrics.csv
    checkpoint_summary.csv
    late_window_changes_by_seed.csv
    late_window_change_summary.csv
    final_same_seed_head_to_head.csv
    final_cross_seed_head_to_head.csv
    exploitability_by_training_time.png
    exploitability_by_nodes_touched.png
```

To repeat exact aggregation after downloading the `workers/` tree:

```bash
python -m experiments.leduc_poker.deep_cfr_ucv_36h_plateau.run aggregate \
  --output-root /path/to/downloaded/RUN_ID
```
