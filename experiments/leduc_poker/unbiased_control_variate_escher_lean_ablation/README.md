# Experiment 8: lean Experiment 6 ablation

Experiment 8 asks whether the performance of Experiment 6 requires all of its
learned auxiliary machinery. It runs paired seeds `0`, `1`, and `2` to the same
per-seed node budgets used by Experiment 6 (approximately 4.7 million nodes).
The full control is trained again so exploitability and wall-clock comparisons
are measured under the same code and execution conditions.

## Arms and causal questions

| Arm | Change from full Experiment 6 | Question |
|---|---|---|
| Full Experiment 6 | None | Fresh control |
| Fixed beta = 1 | Fix the control-variate coefficient; calibration still drives sampling | Is learned beta useful? |
| Prediction gate = 0 | Predictor is trained and diagnosed but never used | Is optimism useful, independent of its training cost? |
| Beta = 1 + no predictor | Remove the instantaneous predictor and use DCFR+ accumulation | Do fixed correction and conservative accumulation interact favourably? |
| Two cross-fitted critics | Reduce three critics to two | Does ensemble redundancy justify its cost? |
| Single frozen-target critic | Use one persistent frozen-target critic | Is strict cross-fitting necessary? |
| Uniform full-support sampling | Remove residual-adaptive sampling | Does learned sampling improve node efficiency? |
| Lean unbiased DCFR+ candidate | Beta 1, no predictor, two critics, uniform sampling, no calibration | Does the actual proposed lean architecture match the full system? |

The first seven rows include every requested arm. The final combined arm is
essential: one-factor ablations cannot establish whether individually harmless
removals remain harmless when composed.

## What is genuinely removed

`Prediction gate = 0` retains and trains the instantaneous predictor. It
isolates the effect of using predictions but should not materially reduce
training cost. In contrast, both `no predictor` arms instantiate the
non-predictive `VRDCFRPlusRegretTrainer`; there is no instantaneous network,
optimizer, forward pass or training pass.

The lean candidate also does not instantiate the residual calibration network
or replay buffer. Fixing beta to one makes calibration unnecessary for the
coefficient, while uniform full-support sampling makes it unnecessary for the
sampling distribution. Two critics retain strict cross-fitting: each
trajectory is written to one fold and evaluated by the other fold. The
single-critic arm is persistent and frozen-target, but deliberately not
cross-fitted.

All arms retain the estimator

\[
\widetilde Q(a)=\beta(a)\widehat Q(a)+
\frac{\mathbf 1\{A=a\}}{\xi(a)}
\left(G-\beta(a)\widehat Q(a)\right).
\]

For every predictable beta and every full-support sampling policy,
`E[Q_tilde(a)] = Q(a)`. The lean candidate therefore removes learned machinery
without introducing the shrinkage bias that complicated Experiment 3. With a
no-regret DCFR+ accumulator, exact tabular function representation and the
usual sampling assumptions, the standard average-strategy-to-Nash route is
still available.

## Experimental contract

- Seeds are paired across all arms.
- Seed `0` targets `4,700,205` nodes, seed `1` targets `4,701,540`, and seed `2`
  targets `4,684,695`, exactly as in Experiment 6.
- Each run evaluates the untrained policy, at approximately 10,000 nodes, and
  after every outer iteration.
- Exact evaluation nodes are excluded from `nodes_touched`.
- Every arm stops after the first complete outer iteration crossing its target.
- Apart from the declared mechanism override, Experiment 6 configuration is
  unchanged.
- Runtime is a first-class outcome and fresh full-control runs are used for its
  paired comparison.

Three seeds make this a screening experiment, not a definitive equivalence
test. Before inspecting results, a useful decision rule is that a lean arm is a
candidate for confirmation if its mean final exploitability is within `0.02`
of full Experiment 6, its node-normalised AUC does not regress materially, and
its paired wall-clock ratio is meaningfully below one. Any winner should then
be confirmed with more seeds and multiple poker games.

## Local full run

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run
```

## Local smoke test

This executes all eight arms for one seed with tiny buffers and training counts.
It verifies solver construction, true component removal, mechanism invariants,
worker isolation, summaries and plots. Its exploitability values are not
scientifically meaningful.

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
  --seeds 0 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --calibration-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --early-evaluation-nodes 10 \
  --output-root outputs/smoke_tests
```

## Runtime and Batch timeouts

Experiment 6 measured approximately `3.53` hours per seed on the existing
eight-vCPU Batch machine. If none of the simplifications saved time, 24 runs
would take `84.7` hours sequentially. Accounting for removed networks gives a
working sequential estimate of approximately **72 hours**; use the documented
**5,760-minute timeout** (96 hours, `345600` seconds) for one complete job.

The requested and recommended execution is the single job below, which runs all
eight variants and three seeds and writes the combined outputs before exiting.
Variant-specific jobs remain available only as a recovery path. These are
conservative estimates, and the experiment itself measures the actual cost of
every arm.

## GCP Batch environment

Push the current commit so the Batch VM can clone it, then define:

```bash
export PROJECT_ID="your-project-id"
export REGION="europe-west1"
export BUCKET="gs://your-escher-results-bucket"
export SA_EMAIL="batch-runner@your-project-id.iam.gserviceaccount.com"
export REPO_URL="https://github.com/lawrencewlcknight/leduc-poker-escher-architecture-experiments.git"
```

## Full single GCP Batch job

```bash
JOB_NAME="leduc-escher-arch-exp8-lean-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

## Optional split-job recovery

```bash
for VARIANT in \
  full_experiment_6 \
  fixed_beta_one \
  prediction_gate_zero \
  fixed_beta_one_no_predictor \
  two_cross_fitted_critics \
  single_frozen_target_critic \
  uniform_full_support_sampling \
  lean_candidate
do
  JOB_NAME="leduc-escher-arch-exp8-${VARIANT//_/-}-$(date -u +%Y%m%d-%H%M%S)"
  ./gcp/submit_batch_experiment.sh \
    "$JOB_NAME" \
    "python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
      --variants $VARIANT \
      --output-root outputs/cloud/$JOB_NAME" \
    n2-standard-8 86400 8000 32000 100
done
```

After the jobs are downloaded, regenerate combined outputs by repeating
`--aggregate-run-dir` for each directory:

```bash
python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
  --aggregate-run-dir cloud_outputs/EXP8_FULL_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_FIXED_BETA_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_GATE_ZERO_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_BETA_NO_PREDICTOR_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_TWO_CRITICS_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_SINGLE_CRITIC_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_UNIFORM_JOB \
  --aggregate-run-dir cloud_outputs/EXP8_LEAN_JOB \
  --output-root outputs/experiment_8_aggregated
```

## GCP Batch smoke test

The smoke test is one Batch job covering all eight arms for seed `0`. Its
timeout is **360 minutes** (`21600` seconds).

```bash
JOB_NAME="leduc-escher-arch-exp8-lean-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.unbiased_control_variate_escher_lean_ablation.run \
    --seeds 0 \
    --target-nodes 50 \
    --traversals 4 \
    --max-iterations 2 \
    --advantage-train-steps 1 \
    --policy-train-steps 1 \
    --q-train-steps 1 \
    --calibration-train-steps 1 \
    --batch-size 2 \
    --buffer-size 128 \
    --early-evaluation-nodes 10 \
    --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

Monitor and retrieve a job with:

```bash
gcloud batch jobs describe "$JOB_NAME" --location "$REGION"
./gcp/read_batch_task_logs.sh "$JOB_NAME"
gcloud storage cp --recursive \
  "$BUCKET/$JOB_NAME/outputs" \
  "cloud_outputs/$JOB_NAME/"
```

## Outputs

- `seed_summary.csv` and `aggregate_summary.json`;
- `checkpoint_curves.csv`;
- `paired_differences_vs_full.csv`, including paired wall-clock ratios;
- `exploitability_by_nodes.png` and `exploitability_by_wall_clock.png`;
- `final_exploitability.png` and `final_wall_clock.png`;
- `performance_cost_frontier.png`;
- metadata, partial recovery files, worker inputs, results and logs.
