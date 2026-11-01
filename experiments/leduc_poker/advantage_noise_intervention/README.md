# Experiment 52: intervene on advantage-label noise

This is an **oracle diagnostic**, not a model-free algorithm or another ranking
benchmark. It asks whether reducing conditional sampling noise improves the
finite neural regret-fitting operation, with the mean target, input histories,
previous regret prediction and optimization budget held fixed.

Experiment 50 asks where critic averaging helps. Experiment 51 measures critic
update budgets in the native learner. This experiment isolates a downstream
link: advantage noise -> regret-fitting error -> regret-matched policy fidelity.
Neither a successful fit experiment nor an exact variance calculation proves
that variance explains every difference between algorithms.

## Fixed protocol

* Sources: Experiment 35 full training states at **24h and 36h**, from seeds
  **145871, 318362, 385626**, the first three numerically sorted production
  seeds. No selection on exploitability or intervention outcomes. Historical
  array indices are looked up from the original seed order, not assumed to
  equal this new order.
* Six checkpoint workers, **eight independent paired repetitions** per worker.
* At each repetition, collect **10,000 trajectories per player** under the
  original frozen policy profile, native adaptive sampler and calibration,
  with the two held-out critic folds equally represented.
* Construct three datasets with noise amplitudes **1, 0.5 and 0**. Both players'
  panels are collected before any fitting. No player sees a policy altered by
  another arm's fit.
* Each regret-network fit starts from identical archived weights, target-network
  weights and Adam state, then performs **750 native optimizer updates**, batch
  size **2,048**, using identical minibatch draws within each triplet. Preserve
  the original network, learning rate, masking and regret-matching fallback.
* Total: **288 short network fits / 216,000 optimizer steps**. No new training
  from scratch, critic fits, calibration fits or average-policy fits.

At a source iteration t, collect and construct the isolated next update at
T=t+1, including calibration's iteration-dependent features and the native
discount factor. The saved current strategy is held fixed for the whole
operation. This deliberately differs from a full sequential CFR iteration.

Collection and exact enumeration share one float64-normalized frozen policy
table. Native DFS is checked against that same table. This avoids conflating
float32 probability-normalization roundoff with an estimator bias at the 1e-9
audit threshold; it is common to all intervention arms. The previous-regret
offset follows the native float32 operation order, and the advantage reference
is computed in float64. Fitting retains native float32 tensors.

## Intervention and exact reference

For full history h and sampled continuation Z, the stored advantage vector is

`a_gamma(h,Z) = mu(h) + gamma * (a_sampled(h,Z) - mu(h))`.

`mu(h)` is the exactly enumerated advantage of the frozen policy, with the
learner's utility normalization. Transform the **complete legal-action vector**
after collection. Recursive sampled returns remain unmodified. Gamma 1 is an
explicit bitwise identity operation on recorded labels; gamma 0 uses mu exactly.
Conditional covariance is multiplied by gamma squared. Higher moments also
change: this is a noise-amplitude intervention, not an isolated manipulation of
variance as a scalar. Hidden-history and visitation variation remain.

The native fitting target is

`((T-1)^alpha / ((T-1)^alpha + 1.5)) * max(R_previous(I,a), 0) + a_gamma(h,a)`.

Only the advantage component is changed. Do not clip the new labels or the
complete regression target. Full history is used by the oracle only; the fitted
network still receives its original information-state tensor.

The exact information-set reference averages mu over hidden histories using
their **sampling-reach probabilities under the frozen collection process**,
including the equal fold mixture. It does not use uniform hidden-history weights,
counterfactual reach, a sampled training-label mean or a fitted oracle network.
Only positive collection-reach information sets contribute to primary fitting
metrics. Their weights sum to one separately for each player. Predictions at
zero-reach information sets still enter whole-game exploitability.

The full action covariance is enumerated, including covariance introduced by
centering. Maximum conditional mean/centering error must be <=1e-9 and minimum
covariance eigenvalue >=-1e-9. Exact references and audits use a separate game
tree enumeration; they do not depend on experimental training labels. A mean
over history-conditioned covariance matrices excludes the between-history
component. The saved gamma covariance matrices are analytically expected
values, not empirical estimates from the training panels.

## Outcomes and statistical unit

Primary: **gamma 0 minus gamma 1 in exact-mean regret MSE**. MSE uses sampling
reach and the native fixed action-output dimension normalization (illegal action
outputs and targets are zero), matching masked `nn.MSELoss`.

Secondary outcomes:

* Gamma 0.5 minus gamma 1, to inspect a graded response.
* Sampling-reach-weighted total variation from the regret-matched exact-mean
  reference policy. That policy is an update reference, not an equilibrium.
* Policy TV within a prespecified sensitive subset and its complement.
  The paired change in their gap (`sensitivity_tv_gap`) is also reported:
  a negative effect means a larger absolute TV reduction in the sensitive group.
* Exact exploitability of each hybrid profile (one player refitted, opponent
  unchanged), and of the joint profile assembled from both isolated fits.
  The duplicated joint value in the two player rows is averaged, not counted
  as two independent observations. These are **current policies**, not fitted
  average-policy heads or a new averaged CFR strategy.

Before any fit, perturb each legal coordinate of the exact reference regret
vector by +/- eta, where eta is 1% of the reach-weighted RMS information-set mean
advantage across legal actions. Average the resulting policy TV over coordinates
and signs. Mark the top ceil(N/4) supported information sets as sensitive, using
lexicographic information-state order to break ties. Save eta, all sensitivities,
and the number of strictly positive sensitivities. If all sensitivities vanish,
the tie-broken subset has no meaningful sensitivity interpretation.

For each paired repetition, average the two players' differences. Average eight
repetitions within each checkpoint, then average the two checkpoints within each
source seed with equal weight. The cross-seed interval therefore has **n=3**, not
48 repetitions, 288 fits, or thousands of states. Conditional repetition intervals
describe Monte Carlo variation for a saved checkpoint, not independent training
seeds. Report all per-seed/checkpoint effects. The sole primary interval is a
95% paired t interval; secondary intervals are descriptive and unadjusted.
With three source seeds, generalization and normality assumptions remain weak.

A convincing result combines lower exact regret error and greater policy fidelity,
especially at sensitive states. Immediate exploitability need not decrease after
one update. Improved fitting alone must not be described as proven long-run
strategic improvement. The study cannot identify averaging as the source of a
noise benefit; that connection needs the separate averaging experiment.

## Local execution

From the repository root in an environment with the project dependencies:

```bash
python -m unittest discover -s tests -p test_advantage_noise_intervention.py
python -m experiments.leduc_poker.advantage_noise_intervention.run smoke \
  --output-root /tmp/exp52-smoke-UNIQUE
python -m experiments.leduc_poker.advantage_noise_intervention.run contract
```

Smoke uses a synthetic trained checkpoint, 16 trajectories/player, two paired
repetitions, three-step fits and eight-row minibatches. It exercises all gamma
arms, exact references, saved models, receipts, resume and analysis. **Smoke
results are not scientific evidence.** Production and pilot refuse synthetic
sources or modified historical learning configurations.

Download one trusted source worker/state and run one checkpoint:

```bash
python -m experiments.leduc_poker.advantage_noise_intervention.run fetch-source \
  --bucket gs://YOUR_SOURCE_BUCKET --source-run-id exp35-confirm-20260916-011231 \
  --task-index 0 --directory /tmp/exp52-source-0
python -m experiments.leduc_poker.advantage_noise_intervention.run worker \
  --task-index 0 --source-worker /tmp/exp52-source-0 --output-root /tmp/exp52-study
```

Repeat for indices 0–5, using the same output root, then run `aggregate
--output-root /tmp/exp52-study`. Task order is seed first, then 24h/36h. Full
source files are verified against historical SUCCESS receipts before loading;
only trusted repository archives should be supplied to the PyTorch loader.

## Efficiency, restart and retained artifacts

Historical replay is loaded with the archive but is not allocated a second time
or retained by the reconstructed solver. Only inference networks and the regret
optimizer states are restored. Exact policy/control/reference caches are built
once per checkpoint and reused across repetitions. The native estimator and
native regret trainer are reused. Collection omits passive replay writes because
no critic, calibration or average-policy training is performed.

Each repetition stores the shared panels (full histories, features, sampled
advantages, means, masks, policy, trajectory and fold IDs), fitted regret weights,
information-set metrics, fit timings and minibatch hashes. Complete repetitions
receive checksummed receipts; an interrupted partial repetition is rerun from
its deterministic seeds. A completed repetition is verified and skipped.
Remote result files are uploaded before repetition receipts, and the final
worker receipt is uploaded last.
Source, code, configuration or runtime changes reject restart. Final aggregation
requires all six workers, all eight triplets for both players, matching identities,
valid audits and every required artifact. It rejects incomplete or corrupt work.

Outputs include `reference.csv`, `oracle_audit.json`, per-repetition panels and
models, `metrics.csv`, manifests and checksums. Analysis provides fit-level data,
paired repetition differences, conditional effects, seed-level effects, timings,
`summary.json` and `report.md`. Store the oracle label artifacts separately from
model-free learning results. There is no effect-based stopping or seed replacement.

## Cloud execution

Worker metadata is emitted by `experiments.cloud_worker_identity`. Configuration
imports can print OpenSpiel optional-module warnings; these are routed to stderr,
never included in worker names or GCS paths. Destinations are validated before
source download or computation, and cloud smoke executes the generated metadata
commands in fresh interpreters. The pilot and production worker use the same path.
This fixes the upload failure in `exp52-noise-20261101-220804-pilot-initial`
without changing the scientific protocol or weakening completion checks.

When upgrading to this launcher fix, use a new run ID and the corrected commit,
then rerun `pilot-cloud` before `run`. Pilot/code identity checks intentionally
do not accept pilot results from another implementation revision. The same
worker-metadata fix applies to Experiments 50 and 51.

Commit and push before using cloud jobs. Pin the exact 40-character pushed SHA;
the source bucket can differ from the destination bucket:

```bash
export PROJECT_ID=YOUR_PROJECT REGION=YOUR_REGION BUCKET=gs://YOUR_OUTPUT_BUCKET
export SOURCE_BUCKET=gs://YOUR_EXPERIMENT_35_BUCKET
export SOURCE_RUN_ID=exp35-confirm-20260916-011231
export SA_EMAIL=YOUR_BATCH_SERVICE_ACCOUNT
export REPO_REF=YOUR_PUSHED_40_CHARACTER_COMMIT_SHA
export RUN_ID=exp52-noise-YYYYMMDD PARALLELISM=3
bash gcp/run_advantage_noise_intervention.sh dry-run
bash gcp/run_advantage_noise_intervention.sh pilot-cloud
```

Dry-run writes specifications only. The pilot downloads task 0's original 24h
state and executes one full-size paired repetition: both 10,000-trajectory
collections, six 750-step fits, exact references, audits and evaluation. It uses
the first prescribed repetition seeds and is explicitly excluded from study
aggregation. It is an engineering check, never an effect-based go/no-go test.
Inspect its memory and timing at `$BUCKET/$RUN_ID/pilot/PILOT_SUCCESS.json` and
the pilot worker's repetition timing before launching the full study. Multiplying
the whole pilot time by 48 is conservative because each checkpoint worker reuses
the same oracle/reference caches across its eight repetitions; archived states
and learned proposals can nevertheless differ in cost.

```bash
bash gcp/run_advantage_noise_intervention.sh run
bash gcp/run_advantage_noise_intervention.sh status
```

The remote controller requires the pilot, runs synthetic smoke and the regression
tests on the production Linux environment, verifies matching code/package
identities, then submits six checkpoint workers followed by strict aggregation.
Each worker uses one `n2-standard-8`, 150 GB balanced disk, 30,000 MiB requested
memory and a 12-hour cap. Pilot cap: six hours; smoke/analysis caps: three hours.
Default concurrency three uses 24 vCPUs; `PARALLELISM` accepts 1–6. There are no
automatic retries. Resource caps are engineering limits, not runtime forecasts.
The laptop can disconnect after submission.

To continue after an infrastructure failure, keep the run ID and commit:

```bash
export ATTEMPT=retry1
bash gcp/run_advantage_noise_intervention.sh resume
```

Workers verify and reuse completed repetitions. Invalid checksums stop processing
for inspection; no failed seed/checkpoint is silently dropped. Paid jobs are
launched only by explicit cloud commands, never by implementation or local smoke.
