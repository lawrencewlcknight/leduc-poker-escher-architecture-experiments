# Experiment 22: three-arm UCV simplification at 15 million nodes

## Question and frozen design

This development experiment tests whether either of the two most actionable
Experiment 8 simplifications survives a long horizon:

1. complete UCV-ESCHER;
2. complete UCV-ESCHER with the control-variate coefficient fixed at
   `beta=1`;
3. complete UCV-ESCHER with two rather than three strictly cross-fitted critic
   folds.

All other configuration fields are inherited from the UCV arm selected before
Experiment 19. The simplifications are imported directly from Experiment 8,
so their causal definitions cannot drift. Every run stops after the first
complete outer iteration crossing 15,000,000 training nodes; the 200-iteration
limit is only a failure guard.

The six paired development seeds are:

```text
452106, 864014, 716235, 928759, 809334, 945659
```

They were generated before running Experiment 22 from the first 32 bits of
`sha256("ucv-three-arm-15m-simplification-development-{index}")`, reduced to a
six-digit label. They do not overlap the Experiment 19/21 held-out seeds. They
remain development evidence: selecting a simplified architecture would require
a subsequent benchmark on new labels.

The training seed is the inferential unit. Six paired seeds permit a minimum
two-sided exact sign-flip p-value of `0.03125`. The two simplified-vs-full
tests receive Holm adjustment. Endpoint, trajectory, runtime, memory and
mechanism evidence must be considered together.

## Predeclared decision rule

Exploitability harm is candidate minus complete UCV-ESCHER. A simplified arm
is eligible for selection only if the upper bound of its paired 95% t interval
is below the predeclared non-inferiority margin of `0.01` exploitability. If
eligible, preference depends on:

- mean and seed-level endpoint exploitability;
- complete node- and wall-clock-indexed trajectories;
- runtime and peak-memory effects;
- whether its mechanism diagnostics supply a coherent explanation.

This rule prevents choosing a cheaper arm whose apparent saving masks a
material quality regression. The experiment is a development selection study,
not a new confirmatory benchmark.

## Diagnostics

The standard per-iteration curves retain exact exploitability, wall time,
critic losses, beta summaries, prediction gates, correction magnitudes,
critic disagreement, calibration loss and replay sizes.

An observation-only solver hook additionally records, without consuming
random numbers or changing the estimator:

- observed and predicted residual moments by player, information set and
  action;
- mean/variance calibration error by information-set/action;
- beta moments and a fixed-bin beta histogram;
- absolute importance-correction magnitude;
- realised advantage-target variance;
- the counterfactual target variance obtained by applying `beta=1` to the same
  sampled transition;
- sampled critic-target RMSE in one iteration and the mean absolute sampled
  local-regret target at the same information-set/action in the following
  iteration. This is an empirical prediction-residual diagnostic, not an exact
  decomposition of critic approximation error from return noise.

Workers also record wall-clock time and process peak resident memory. Dense
diagnostics are reduced online to sufficient statistics; individual sampled
returns are not retained. The hook is a no-op for every solver outside this
experiment.

The counterfactual target-variance comparison is descriptive. It holds the
realised trajectory and action sample fixed and therefore isolates estimator
arithmetic on those observations, but it is not a separately trained fixed-beta
trajectory.

## Mandatory smoke test

From the repository root in the Python 3.9 environment:

```bash
export SMOKE_OUTPUT="/tmp/exp22-smoke-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_ucv_three_arm_15m_simplification.sh smoke-local
```

The smoke runs all three real implementations on development seed `0`, checks
the variant invariants, creates and reloads playable policies, exercises every
diagnostic export, performs paired aggregation and renders every chart. Its
numerical results are not scientific.

## Fully remote GCP run

Experiment 22 reuses the remote-controller permissions established for
Experiment 19. Commit and push the implementation, then set:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp22-simpl-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=18

./gcp/run_ucv_three_arm_15m_simplification.sh run
```

The command returns once the controller is accepted. The controller performs
a clean cloud smoke, submits the 18 production workers only after the smoke
succeeds, and finally runs aggregation. The laptop may be closed after
submission.

At full parallelism the run needs 144 regional N2 vCPUs. The measured
pre-diagnostic estimate is approximately 143 training VM-hours; allow
155--170 N2 VM-hours and 10--13 elapsed hours for diagnostic overhead,
bootstrap, completed-iteration overshoot, smoke and aggregation. Each worker
has an independent 20-hour hard limit. Standard VMs are mandatory because the
playable endpoint policies do not contain optimizer/replay state for resuming a
preempted trajectory.

Use `status`, `resume`, or `dry-run` with the same launcher. A recovery run
reuses only workers whose result, commit and snapshot checksum all validate.

## Outputs

```text
gs://.../RUN_ID/
  smoke/
  workers/
    task_000_full_experiment_6_seed_452106/
      worker_result.json
      SUCCESS.json
      checkpoint_curves.csv
      snapshots/...node_15m.pkl
      diagnostics/
        information_action_diagnostics.csv
        beta_histogram.csv
        critic_error_subsequent_local_regret.csv
    ... 17 further workers ...
  analysis/
    aggregate_summary.json
    worker_manifest.csv
    seed_summary.csv
    paired_differences_vs_full.csv
    paired_inference.csv
    checkpoint_curves.csv
    information_action_diagnostics.csv
    diagnostic_summary_by_worker.csv
    beta_histogram.csv
    critic_error_subsequent_local_regret.csv
    exploitability_by_nodes.png
    exploitability_by_wall_clock.png
    final_exploitability.png
    final_runtime.png
    peak_memory.png
    performance_cost_frontier.png
    beta_by_nodes.png
    prediction_gate_by_nodes.png
    correction_magnitude_by_nodes.png
    target_variance_ratio.png
    calibration_reliability.png
    critic_error_vs_subsequent_local_regret.png
    beta_distribution.png
```

After downloading the complete `workers/` tree, repeat aggregation with:

```bash
python -m experiments.leduc_poker.ucv_three_arm_15m_simplification.run \
  aggregate --output-root /path/to/downloaded/RUN_ID
```
