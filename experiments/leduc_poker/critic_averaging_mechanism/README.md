# Experiment 50: where critic averaging helps

This experiment separates two uses of the critic that earlier ablations changed
together: the frozen network that supplies bootstrapped training labels, and the
frozen network used as a control variate during collection. It is a mechanism
study on Leduc, not another algorithm ranking or a confirmation on FHP.

## Frozen design

| Arm | TD target snapshot | Collection control snapshot |
|---|---|---|
| A | Latest completed fit | Latest completed fit |
| B | Mean of four completed fits | Latest completed fit |
| C | Latest completed fit | Mean of four completed fits |
| D | Mean of four completed fits | Mean of four completed fits |

Averaging is of **parameters**, not predictions. During startup, use all available
completed fits, up to four; before the first fit, use the common initial network.
Both roles are frozen throughout each collection/fit cycle. Each of two cross-fit
critics has its own online optimizer, four-fit history, TD snapshot and collection
snapshot. The inherited cross-fitting rules remain in effect.

The scientific starting configuration is Experiment 35's grouped-wide UCV
candidate. Experiment 25 supplies the historical averaging comparison.
Experiments 47/48 supply the per-fit frozen-target cache. The original 10,000
critic updates and batch size 2,048 are retained; caching changes computation,
not labels or update counts. No production configuration is edited in place.

One intentional intervention applies to **every** arm: the traverser's proposal
is `0.2 uniform + 0.8 current strategy`, independent of the critic. Opponents
remain on-policy. Residual calibration is disabled; beta stays at one and the
instantaneous predictor stays disabled. This prevents critic averaging from
also changing the proposal directly. Conclusions must explicitly identify this
controlled proposal; a later production-sampler confirmation remains necessary.

* Three fresh, deterministic seed blocks, four arms each: **12 runs**. Seeds are
  defined by the versioned SHA256 construction in `config.py`, never selected
  from observed outcomes. These are the first three blocks from the original
  eight-seed plan, reduced before production to control compute cost. Smoke/pilot
  use a separate seed. This is an exploratory mechanism study.
* **600 completed iterations** per run, with 10,000 traversals per player per
  iteration: **12 million trajectories per run**. Node count is reported and
  can differ between arms. This is iteration/trajectory matching, not node or
  wall-time matching.
* Exact own-reach-weighted average-strategy exploitability every 50 iterations.
  The inherited average observes the strategy at the start of each iteration.
* Primary endpoint: the arithmetic mean at iterations **450, 500, 550, 600**.
  Primary comparison: paired **D minus A**, with a 95% paired t interval.
* Secondary contrasts: **B minus A**, **C minus A**, and **D minus B minus C plus
  A**. Report 98.333% intervals (Bonferroni across these three contrasts).
* The seed block is the inferential unit. Checkpoints and information sets do
  not increase the sample size. Exact sign-flip sensitivity is also reported;
  it assumes symmetric paired differences and is not labelled a randomized
  treatment-assignment test.
* A practically meaningful benefit is established only if the upper primary
  interval is below **-0.005** exploitability. This is a proposed threshold in
  game-native `NashConv / 2` units, not a demonstrated power calculation.
  Three seeds give imprecise intervals; an inconclusive result is valid. The
  two-sided exact sign-flip sensitivity cannot attain p < 0.25 with three
  nonzero paired differences. Treat any mechanism conclusion as preliminary.
* No effect-based early stopping, checkpoint selection, seed replacement or
  dropping failed workers. Analysis requires every scheduled worker and all
  required artifacts. Infrastructure failures resume the same seed/arm.

The long horizon matters: the historical averaging advantage emerged late.
A 15-million-node run would not test the late-training hypothesis adequately.
600 iterations is the fixed horizon; historical node/time estimates do not
silently shorten it.

## Mechanism measurements

At iterations **100, 300, 600**, exact game-tree recursion compares latest and
four-fit controls under the **same current policy and proposal**, separately for
both folds. It reports advantage variance, conditional mean error, critic error,
and information-action detail. The recursion includes covariance induced by
centering action estimates. It fails if maximum absolute estimator mean error
exceeds `1e-9`.

Variance mixes hidden histories using sampling reach within information sets;
the weighted summary is not claimed to be a counterfactual-reach-weighted global
regret variance. The within-checkpoint comparison holds its weighting measure
fixed. Critic RMSE is against conditional information-set action values and
should not be interpreted as a full-history prediction-error decomposition.

Every critic fit also records TD-label movement on up to 4,096 deterministically
spaced occupied replay rows. The same rows and next-state strategy are used
before and after installing the new TD snapshot. This isolates label movement
caused by changing that snapshot. The panel changes as replay changes; it is a
local drift diagnostic, not a fixed longitudinal dataset or a gradient-noise
measurement. Final minibatch loss, parameter movement and cache costs are saved.

Interpret the evidence together:

* C improving on A, accompanied by reduced frozen-policy estimator variance,
  supports a collection-control mechanism.
* B improving on A, accompanied by reduced TD-label movement, supports a
  training-target stability mechanism. Drift alone is insufficient: slowly
  moving but inaccurate targets can also have low drift.
* The interaction tests whether the effects combine non-additively. D beating A
  alone cannot establish which mechanism is responsible.
* If D does not outperform A under this proposal, do not claim that the
  production averaging effect has been explained. Proposal adaptation or
  other interactions may be necessary.

These are falsifiable explanations, not guaranteed outcomes. Exact variance
reduction does not by itself prove improved nonlinear learning or convergence.

The deployed grouped policy head is fitted only at **300 and 600**, with the
inherited 20,000-step budget. Its exploitability and playable snapshots are
secondary outcomes, kept distinct from the exact-average primary endpoint.
Policy fitting, collection, regret training and each critic use isolated RNG
streams; evaluation does not consume subsequent learning randomness.

## Local verification

Use an environment with the repository's dependencies installed:

```bash
python -m unittest discover -s tests -p test_critic_averaging_mechanism.py
PYTHON=python bash gcp/run_critic_averaging_mechanism.sh smoke-local
```

Smoke performs real six-iteration training for all four arms, policy extraction,
exact diagnostics, full-state persistence and complete-block aggregation. Its
single development seed and reduced networks/budgets are **not scientific
results**. By default output goes to a fresh `/tmp/exp50-smoke.*` directory;
set `SMOKE_OUTPUT` to retain it elsewhere.

The tests cover shared initialization, snapshot-role separation, compatibility
with native one/four-fit averaging, cached/uncached update equivalence, evaluation
isolation, exact learning-state continuation for all arms, disk resume with
snapshot extraction, corruption rejection, incomplete-analysis rejection and
all five generated cloud scripts. Cloud smoke reruns these tests using the Linux
production dependencies before a training array can start.

## Engineering pilot and cloud execution

All stages use isolated per-invocation working directories and bounded package
installation retries, including when Batch reuses a VM for the next task.
For infrastructure-only recovery, retain `RUN_ID` and `REPO_REF`, set
`LAUNCHER_REF` to the pushed launcher-fix commit, and choose a new `ATTEMPT`.
See [shared recovery instructions](../advantage_noise_intervention/README.md#startup-reliability-and-infrastructure-only-recovery).
The scientific source/runtime/checkpoint checks remain mandatory.

Worker metadata uses the shared import-safe helper described in
[Experiment 52's cloud notes](../advantage_noise_intervention/README.md#cloud-execution).
Cloud smoke tests the generated metadata commands, and invalid destinations fail
before training. Pilots are optional timing measurements, not prerequisites for
`run` or `resume`; they may still be running when full training is submitted.
An older pilot does not need to be repeated merely to match the launcher commit.
The experiment configuration and training budget are unchanged.

First commit and push the implementation. Cloud jobs require an exact pushed
40-character SHA. Set the existing project, region, bucket and service account:

```bash
export PROJECT_ID=YOUR_PROJECT REGION=YOUR_REGION BUCKET=gs://YOUR_BUCKET
export SA_EMAIL=YOUR_BATCH_SERVICE_ACCOUNT
export REPO_REF=YOUR_PUSHED_40_CHARACTER_COMMIT_SHA
export RUN_ID=exp50-roles-YYYYMMDD PARALLELISM=8
bash gcp/run_critic_averaging_mechanism.sh dry-run
```

`dry-run` writes job specifications only. It does not submit a job.

For an optional timing pilot, use a trusted archived **Experiment 35 full training
state**, not a playable policy snapshot, from iteration 300 or later. Provide
its independently verified SHA256:

```bash
export SOURCE_URI=gs://YOUR_BUCKET/PATH/TO/FULL_STATE.pt
export SOURCE_SHA256=YOUR_VERIFIED_64_CHARACTER_SHA256
bash gcp/run_critic_averaging_mechanism.sh pilot-cloud
```

The pilot runs three mature continuation iterations per arm, from independent
reloads of the same archived state, applying the controlled proposal and role
split. This measures engineering cost; it is not pooled with the confirmation.
Its state schema and critical model/training settings are checked. Inspect
`$BUCKET/$RUN_ID/pilot/pilot.json`, including memory feasibility and iteration
cost, when available. The reported 600-iteration extrapolation excludes
setup, checkpoint upload and scheduled diagnostics. A fresh reduced pilot is
also available through the Python CLI but is not a production runtime forecast.

The following full-run command does not read or wait for a pilot receipt and
does not need `SOURCE_URI` or `SOURCE_SHA256`. Existing pilots can continue in
their separate `pilot/` output prefix. Their results are not pooled with training.

```bash
bash gcp/run_critic_averaging_mechanism.sh run
bash gcp/run_critic_averaging_mechanism.sh status
```

A remote controller waits for cloud smoke, checks its completion receipt and
source fingerprint against the pinned scientific checkout, then submits 12
training tasks and the aggregator. One `n2-standard-8` VM per task; eight simultaneous tasks by default
(64 vCPUs). Each task has a 72-hour cap, 150 GB disk and no automatic retries.
The controller is bounded as well. Starting before the pilot completes accepts
uncertainty about mature runtime and memory. Use the pilot to refine estimates;
a provisional 24–36 hours per run implies 288–432 worker VM-hours, or
roughly two to three days at concurrency eight, before overhead. This is not a
measured forecast for the new experiment. `PARALLELISM` accepts 3–12; reducing
concurrency lowers the simultaneous quota requirement, not the training work.

### Rough cost estimate (23 October 2026)

For planning, allow **US$150–250** for one successful 12-run experiment,
before tax, without retries or long-term artifact retention. This is an estimate,
not a billing cap. Runtime remains provisional until the mature-state pilot.

Google's public [general-purpose VM price table](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en)
shows a reference on-demand `n2-standard-8` price of $0.388472/hour; the actual
rate depends on the selected region. The reference
[balanced disk price](https://cloud.google.com/compute/disks-image-pricing)
is $0.000136986 per GiB-hour, or about $0.02055/hour for 150 GiB.
Use a rounded **$0.45 per worker-hour including disk** as a planning assumption,
allowing some regional price variation; verify the chosen region before launch.

| Runtime per run | Total worker-hours | Training plus attached disk at $0.45/hour |
|---|---:|---:|
| 24 hours | 288 | $130 |
| 30 hours | 360 | $162 |
| 36 hours | 432 | $194 |

Allow a further **$20–40** for the pilot, setup, evaluation/checkpoint overhead,
cloud smoke, controller, aggregation and short-term artifact storage. Rounded
up, **$250 is a sensible provisional budget**. The 72-hour worker timeout is not
a $250 spending limit: if every worker used the full allowance, training and
disks alone would be about $389 at this assumed rate, before other charges.
Eight simultaneous workers imply roughly two waves, or two to three days at
the provisional runtimes; the total work is unchanged by concurrency.

Continuation uses the same run namespace and source, with a new Batch attempt:

```bash
ATTEMPT=retry1 bash gcp/run_critic_averaging_mechanism.sh resume
```

States include networks, optimizers, replay contents/cursors, temporal history,
exact-average accumulators, all component RNG streams and diagnostic progress.
Two alternating state files are retained, with an atomic checksum pointer
advanced only after the state and its supporting artifacts have uploaded.
Checkpoints occur every 100 iterations; an interruption can require repeating
up to 99 completed iterations. Completed workers verify artifact receipts and
return without retraining. Configuration/source/runtime mismatch or corrupted
state fails closed. Do not run two controllers for the same namespace at once.

Results live beneath `$BUCKET/$RUN_ID/`: `workers/`, `pilot/`, `smoke/`, and
`analysis/`. Per-worker files include `metrics.csv`, `critic_fits.csv`, exact
`diagnostic_*.csv`, policy snapshots, full training states, and hashed receipts.
Analysis produces `summary.json`, `seed_blocks.csv`, `curves.csv`, `report.md`,
`mechanism_diagnostics.csv`, and descriptive `td_drift_by_seed.csv`.

## What has and has not been validated

Local real-Leduc tests and the complete reduced four-arm smoke validate the
implementation paths. Linux cloud smoke is still required before production;
the mature archived-state pilot is optional. Neither a scientific result nor a
production runtime estimate follows from local smoke. The experiment does not launch
cloud resources merely by importing the module or generating specifications.
