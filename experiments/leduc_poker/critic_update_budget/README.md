# Experiment 51: how much critic training does native UCV-ESCHER need?

This Leduc study measures the effect of reducing critic optimizer updates in
the established Experiment 35 learner, using Experiment 48's frozen-target
cache. It retains native adaptive sampling, residual calibration and average
policy fitting. It does **not** inherit Experiment 50's controlled proposal,
disabled calibration or reduced policy-fitting schedule.

## Frozen design

| Arm | Critic updates per member per iteration | Interpretation |
|---|---:|---|
| A | 10,000 | Established learning configuration |
| B | 2,500 | Quarter budget |
| C | 500 | Twentieth budget |
| D | 0 | Identically zero Q control |

There are **three paired seed blocks and four arms: 12 runs**, each lasting
**600 completed iterations**. Every iteration collects 10,000 trajectories per
player: 12 million trajectories per run. This matches iterations and
trajectories, not nodes or wall time. Node counts and elapsed times are reported.
The fresh seed labels are `36707599`, `41452265`, and `602067218`, generated
deterministically by the versioned SHA256 rule in `config.py`; development uses
a separate seed. No production observations were used to select these seeds.

The positive-budget arms differ only in critic-update count. All retain:

* Two cross-fit critics; a four-fit **parameter** average for the frozen critic.
  Before four completed fits, use the available history. Each critic receives
  the stated update budget independently, with minibatch size 2,048.
* Native residual-standard-deviation sampling with a 20% uniform floor, learned
  residual calibration (2,000 updates per iteration), fixed beta one and no
  instantaneous regret predictor.
* The native regret learner, replay capacities, representations, learning rates
  and traversal budgets, including 750 regret updates per player per iteration.
* The grouped 3x136 average-policy network, reset and fitted for **20,000 updates
  every iteration**, with its native cross-entropy objective and learning rate.
* Frozen-target caching, rebuilt for each positive-budget critic fit. It changes
  computation, not the labels, averaging rule or optimizer update count.

Arm D explicitly returns zero critic predictions from initialization onward;
stored online and target network parameters are also zero. It does not train
a randomly initialized critic, build a TD-target cache or refresh averaging
history. **Residual calibration and its adaptive sampler remain active.** Critic
replay is still populated to preserve the surrounding data path, so this is a
zero-Q component ablation, not a separately optimized critic-free architecture.
The observed gain in runtime is correspondingly conservative about removable
critic bookkeeping.

Native adaptive sampling may change as each arm learns different critics and
residuals. That is part of the intended effect on the complete algorithm. This
study does not isolate optimizer count under a shared, frozen sampling policy.

Common initialization and separate random-number streams for collection, regret
training, each critic, calibration and policy fitting prevent a smaller critic
budget from merely shifting every subsequent random draw. These streams are
the same across arms within each seed; they do not force trajectories to remain
identical after policies diverge. The 10,000-update arm preserves the historical
learning configuration, not the exact historical random trajectory.

Only passive historical diagnostics and the paired legacy comparison-policy
fit are omitted. The candidate policy fit remains per iteration. No existing
experiment implementation or configuration is changed.

## Outcomes and interpretation

Exact-average and deployed neural-policy exploitability are evaluated every
50 iterations. The exact average is the inherited own-reach-weighted analytical
accumulator of strategies at the start of each iteration; it is used for
measurement and does not supply training targets.

The primary outcome is mean exact-average exploitability at **450, 500, 550 and
600 iterations**, in game-native `NashConv / 2` units. Report paired **B minus A,
C minus A and D minus A** differences, with 98.333% paired t intervals
(Bonferroni over these three comparisons). Negative differences favour the
smaller budget. Deployed-policy late means are secondary; retain all learning
curves and seed-level results rather than selecting the best checkpoint.

A prespecified exploratory loss margin of **0.005** is reported: does the upper
interval lie below that margin? This threshold is a practical proposal, not a
power calculation. With three independent seed blocks, intervals can be very
wide and assumptions difficult to check. A nonsignificant difference does not
show equivalence; the experiment cannot establish general noninferiority from
three seeds. Checkpoints, folds and information sets do not increase the
inferential sample size. A promising reduced budget should receive fresh-seed
confirmation before being promoted as a replacement baseline.

There is no effect-based stopping, arm elimination, seed replacement or pooling
of incomplete runs. Analysis requires all 12 valid workers. Infrastructure
failures resume the same seed and arm. The fixed horizon must not be shortened
because one arm runs more slowly.

## Exact diagnostics and compute accounting

At iterations **100, 300 and 600**, exact game-tree recursion audits both critic
folds under the arm's current policy and native adaptive proposal. It compares
the deployed Q control against a zero Q control while holding that policy,
calibration and proposal fixed. This diagnostic counterfactual does not retrain
the sampler. For Arm D the two audits must agree.

Outputs include conditional advantage variance, its unweighted rowwise 95th
percentile and maximum, conditional estimator mean error, and information-set
critic RMSE. Exact recursion includes covariance from centering action
estimates. The worker fails if maximum absolute advantage or action-value mean
error exceeds `1e-9`. Detailed information-action rows are retained.

The variance summary weights information-action rows by sampling reach and
mixes hidden histories within information sets using that reach. It is not a
global CFR regret-variance estimate. Critic RMSE is against conditional
information-set values, not a full-history error decomposition. Across arms,
policies and reach measures can differ; the clean local variance comparison is
between controls **within the same checkpoint and fold**. These exact audits
describe the current estimator and are not held-out generalization tests.

Every critic fit records configured and actual optimizer steps, cumulative
steps, replay occupancy, history length, target version, parameter movement,
final minibatch loss, cache size and fit time. Fits may legitimately take zero
steps before a full minibatch is available. Aggregation verifies the complete
ledger against the configured budget and rejects critic work in the zero arm.

Learning time includes native policy fitting and exact-average accumulation.
Scheduled evaluation, exact audits and checkpoint persistence are recorded
separately; component timers overlap and must not all be added together.
Compare accuracy against both iterations and recorded learning time. Savings
at the fixed workload do not establish what a longer equal-time run would
achieve. In particular, retaining the 20,000-step policy fit limits the total
speedup obtainable from critic reductions alone.

Interpretation should distinguish these possibilities:

* B or C preserves performance and retains the deployed-control variance
  advantage: fewer critic steps may be sufficient at this learning horizon.
* D degrades while B or C preserves performance: a trained critic remains
  useful, but the original fitting budget may be excessive.
* D performs similarly: test whether the remaining sampler/calibration is doing
  most of the useful work; do not conclude that all variance reduction is
  unnecessary, since residual calibration remains enabled.
* A smaller budget improves performance: investigate critic overfitting or
  feedback through adaptive sampling. This experiment alone does not prove
  either mechanism.

## Local verification

From the repository root, use Python with the repository dependencies installed:

```bash
python -m unittest discover -s tests -p test_critic_update_budget.py
PYTHON=python bash gcp/run_critic_update_budget.sh smoke-local
```

Smoke runs all four arms for six real Leduc iterations with small networks and
budgets. It exercises exact diagnostics, playable policy snapshots, full-state
checkpoints, checksummed receipts and complete-block aggregation. These are
engineering checks, not scientific results. Output defaults to a new
`/tmp/exp51-smoke.*` directory; set `SMOKE_OUTPUT` to choose another empty folder.

Tests cover native sampler inheritance, common initialization, actual optimizer
steps, zero Q values, native cached-averaging equivalence, per-iteration policy
fitting, evaluation isolation, exact continuation for every arm including
calibration and replay, disk resume and corrupt-checkpoint rejection, mature
pilot configuration checks, incomplete analysis, and generated shell syntax.
The cloud smoke reruns tests and the complete pipeline on the production Linux
dependencies before production training can begin.

## Mature pilot and cloud execution

Worker metadata uses the shared import-safe helper described in
[Experiment 52's cloud notes](../advantage_noise_intervention/README.md#cloud-execution).
Cloud smoke tests the generated metadata commands, and invalid destinations fail
before training. Pilots are optional timing measurements, not prerequisites for
`run` or `resume`; they may still be running when full training is submitted.
An older pilot does not need to be repeated merely to match the launcher commit.
The experiment configuration and training budget are unchanged.

Commit and push before cloud execution; all jobs require an exact pushed
40-character commit SHA. Set the existing cloud project, region and Batch
service account, and choose a persistent run ID:

```bash
export PROJECT_ID=YOUR_PROJECT REGION=YOUR_REGION BUCKET=gs://YOUR_BUCKET
export SA_EMAIL=YOUR_BATCH_SERVICE_ACCOUNT
export REPO_REF=YOUR_PUSHED_40_CHARACTER_COMMIT_SHA
export RUN_ID=exp51-budget-YYYYMMDD PARALLELISM=8
bash gcp/run_critic_update_budget.sh dry-run
```

`dry-run` writes job specifications only. It submits no cloud work.

For an optional engineering pilot, use a trusted archived
**Experiment 35 full training state at iteration 300 or later**, with its
independently verified SHA256. A playable-policy file is insufficient, and
Experiment 48 intentionally did not retain full states.

```bash
export SOURCE_URI=gs://YOUR_BUCKET/PATH/TO/FULL_STATE.pt
export SOURCE_SHA256=YOUR_VERIFIED_64_CHARACTER_SHA256
bash gcp/run_critic_update_budget.sh pilot-cloud
```

The pilot independently reloads this mature state for each arm and runs three
iterations, retaining native calibration and policy fitting. For D it explicitly
zeros the restored critic. It checks the historical learning configuration and
records iteration times, replay sizes, process peak memory and component costs.
It is an engineering continuation, not fresh scientific evidence. Peak memory
is a process high-water mark across sequential arms and is conservative for
later arms. Its reported 600-iteration extrapolation excludes setup, scheduled
diagnostics and checkpoint persistence.

Inspect `$BUCKET/$RUN_ID/pilot/pilot.json` when available. Use the
sum of each arm's projected hours multiplied by three seeds to estimate worker
VM-hours, then allow overhead and apply current regional VM, disk and storage
prices. **Do not reuse Experiment 50's cost forecast**: that study omits native
per-iteration policy fitting and calibration. A small smoke is not a production
runtime or memory forecast. For a local mature pilot, set `SOURCE_STATE`,
`SOURCE_SHA256`, optionally `PILOT_OUTPUT`, and use `pilot-local`.

The following full-run command does not read or wait for a pilot receipt and
does not need `SOURCE_URI` or `SOURCE_SHA256`. Existing pilots can continue in
their separate `pilot/` output prefix. Their results are not pooled with training.

```bash
bash gcp/run_critic_update_budget.sh run
bash gcp/run_critic_update_budget.sh status
```

The controller runs cloud smoke, checks its completion receipt and source
fingerprint against the controller's pinned checkout, and then submits the
12-worker array and analysis. No pilot completion or pilot-code match is required.
Each worker uses one `n2-standard-8` CPU VM with 150 GB disk and a 72-hour cap;
automatic retries are disabled. Default concurrency eight uses 64 vCPUs.
`PARALLELISM` accepts 3–12; lower concurrency reduces simultaneous quota needs,
not the total training work. Starting before the pilot completes accepts uncertainty
about mature runtime and memory; use its results to refine the resource forecast.
The controller can continue after the laptop disconnects.

Full training states are saved every 100 iterations to two alternating files,
with a checksummed pointer uploaded last. They include networks, optimizers,
replay, calibration, averaging history, exact-average state, component random
streams, update counts and accumulated metrics. Playable grouped policies are
saved at 300 and 600. Receipts hash every required result artifact; source,
runtime or configuration changes reject continuation.

To resume after an infrastructure failure, retain `RUN_ID`, `REPO_REF` and all
settings, choose a fresh attempt label, and explicitly resubmit:

```bash
export ATTEMPT=retry1
bash gcp/run_critic_update_budget.sh resume
```

Completed workers validate their artifacts and exit; incomplete workers resume
from their last valid full state. This does not guarantee survival of a
corrupted remote checkpoint: invalid checksums stop the worker for inspection.
Repeated jobs incur additional costs, and the 72-hour cap is a per-attempt limit.

## Outputs

Each worker saves `manifest.json`, `metrics.csv`, `critic_fits.csv`, exact
diagnostic summaries and detail, two full-state slots, playable policies,
`result.json`, and `SUCCESS.json`. Cloud aggregation downloads analytical
artifacts and policies, excluding large training states. The analysis directory
contains `summary.json`, `report.md`, `seed_blocks.csv`, `curves.csv`,
`mechanism_diagnostics.csv` and `costs_by_seed.csv`.

A subsequent FHP experiment should test the most informative reduced budget
against its own established baseline on fresh seeds. This Leduc study supplies
a budget choice and mechanistic evidence; it does not establish transfer to FHP.
