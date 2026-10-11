# Experiment 49 — RNaD re-solving of the confirmed UCV-ESCHER policy

## Question and frozen source

Can neural test-time RNaD improve the exact equilibrium quality of the best
confirmed UCV-ESCHER deployment policy within a practical decision budget?
Does it improve on a tabular re-solver under the **same** budget and opt-out
construction? This is an evaluation/fine-tuning experiment, not another
36-hour UCV training run.

Use Experiment 35, `exp35-confirm-20260916-011231`, grouped shared
`136 × 136 × 136` policy networks, seeds
`470892, 385626, 145871, 902492, 318362`. The **36-hour** snapshots are primary;
the corresponding **12-hour** snapshots are a secondary weaker-blueprint
comparison. The source URI defaults to the existing Leduc results bucket
(which happens to contain `dream` in its name); it is separate from the
destination `BUCKET`. Only ten small playable policies and three analysis
files are fetched, not replay reservoirs or resumable training states.

The loader checks inventory SHA-256 **before** unpickling trusted snapshots,
frozen Experiment 35 configuration, seed/checkpoint provenance, batched actor
parity and source exact exploitability (tolerance `1e-5`).

## Arms and budgets

| Arm | Local optimizer | Opponent protection |
|---|---|---|
| Blueprint | None | Original policy |
| `cfr_plus_gadget` | Alternating full-tree CFR+, linear average | Counterfactual opt-out gadget |
| `rnad_gadget` | Neural RNaD / NeuRD, fixed blueprint regularizer | Same opt-out gadget |
| `rnad_bayesian` | Same neural optimizer | None; ablation only |

Both main solvers start at the blueprint strategy. CFR+ is initialized with
one unit of positive pseudo-regret per information set, distributed according
to its blueprint probabilities; its gadget starts uniform. This is an explicit
warm-start convention, not a persistent regularizer or an oracle-selected
prior strength. Subsequent alternating regret updates and linear averaging
are ordinary CFR+. The neural actor starts from the exact saved weights.

For each saved blueprint, the two main interventions receive **0.1, 0.5,
2 and 5 seconds per public root and resolving seat**, with **eight** nested
resolver repetitions. The prespecified primary cell is **36h, 0.5s,
`rnad_gadget`**. The no-gadget ablation is restricted to 36h and 0.5s.
One generous **4,096-update** diagnostic per 36h blueprint is included for
both main solvers. Update counts are **not** equivalent work across the two
algorithms; this diagnostic tests optimization headroom, not speedup.
There are 690 full-run intervention cases, each representing a complete
stitched policy, plus the unchanged blueprint references.

The CPU engineering pilot uses the first 36h blueprint, two resolver
repetitions, 0.1s/0.5s, the no-gadget ablation at 0.5s, and 64-update diagnostics:
12 intervention cases. It covers **all** public roots, not just favourable
positions. A pilot alone cannot substantiate a policy-quality claim.

## Public-state solve and information boundary

At the start of the second betting round, after the public card, build a
public continuation problem across **all 20 compatible private deals**.
There are 30 legal public roots; both resolving seats are evaluated, giving
60 resolves per complete policy. First-round play stays identical to the
blueprint. A resolved continuation is retained for the rest of that betting
round; it is not re-optimized after each action.

The solve does not receive the acting player's actual deal, nor the
opponent's actual card. Actor and value-network features are exclusively
OpenSpiel information-state tensors. Privileged simulator outcomes supply
training returns in the usual way; they are not inference features. One
network parameterization shares learning across all compatible information
sets in the public problem. Separate local copies are reset from the frozen
blueprint and value model for every root, resolving seat and repetition.
There are no persistent cross-hand updates.

For hero `i`, gadget root weights are proportional to
`chance_reach × blueprint_reach_i`, **excluding opponent action reach**.
An opponent can deviate into a root that has zero blueprint self-play reach;
such roots are not discarded. A genuinely zero hero counterfactual-reach
root falls back to the blueprint and is reported.

At the root, the opponent observes only their own root information set and
chooses termination or continuation. The termination payoff is their exact
conditional best-response value against the frozen hero blueprint. These
values are computed by aggregating indistinguishable histories **before**
maximization, not by choosing an action separately for each hidden deal.
The other player's payoff is its negative. CFR+ explicitly learns the
gadget choice. Neural RNaD uses a two-action tabular-logit gadget, integrates
termination exactly, samples continuation, and weights actor/value updates
by the frozen pre-batch continuation probability. The no-gadget ablation
instead uses the Bayesian posterior including both players' blueprint reach.

Only the **resolving hero's** updated probabilities are stitched into the
deployed policy. The local opponent's optimized strategy is never assumed
to describe the real opponent.

## Neural optimizer

This is a PyTorch adaptation of the OpenSpiel RNaD/NeuRD equations, not
PPO plus a KL penalty. It includes zero-sum regularized rewards, the
alternating-player importance-corrected V-trace recurrence, and centered,
thresholded NeuRD action-value forces. See `NOTICE.md` for the pinned
reference implementation, attribution, licence and methodological sources.
It is **not** claimed to reproduce the 2026 authors' entire system.

| Setting | Value |
|---|---:|
| Fixed blueprint regularization coefficient | 0.2 |
| Actor and value learning rates | 0.0003 |
| Gadget learning rate | 0.001 |
| Adam betas | (0.99, 0.999) |
| Continuation batch | 64 trajectories |
| NeuRD centered-logit threshold | 2 |
| Advantage clip | 5 |
| V-trace lambda / discount | 0.95 / 1 |
| Target value-network Polyak rate | 0.001 |

All utilities are in **native OpenSpiel Leduc chips**, without silent
normalization. No deployment probability pruning or thresholding is used.
The fixed regularizer's legal log probabilities are obtained directly from
the frozen actor's log-softmax, avoiding probability-underflow logarithms.

The value estimator is a **separate neural model**, initialized from the
blueprint's hidden layers with a new scalar head. Before local solving, it
is fitted on 4,096 independently sampled blueprint self-play hands for 1,000
minibatch updates; 1,024 further whole hands assess MC value error. These
labels are sampled returns, **not** exact values or best-response labels.
All local solves start from the same prepared value model. Preparation cost,
held-out MSE, zero-predictor MSE and observed information-set coverage are
reported separately. The privileged UCV training critic is not reused.

The exact public ranges and exact best-response termination bounds are
deliberate Leduc-only diagnostic advantages. They isolate optimization from
belief/value-bound estimation. Extending this method to FHP would require
additional work; a positive result here is not evidence of scalable solving.

## Timing, correctness and safety qualifications

Each decision timer starts **before** public-subtree construction, bound
calculation, model copying and optimizer allocation. It includes fitting and
probability extraction. Blueprint loading, global game indexing and value
pretraining are separately recorded/offline preparation; exact post-hoc
evaluation is not charged as gameplay computation. Framework startup is
warmed once per process, but optimizer state is new for every resolve.

CPU updates are not interruptible: these are **soft deadlines**. If an
update crosses its deadline, its result is discarded; the last fully
completed candidate is deployed, or the blueprint if none completed.
The elapsed overrun is still reported. This does not guarantee hard
100ms/500ms response times. Report median/p95/max latency, setup/fitting/
extraction time, attempts/completions, fallback fraction and overruns.

Zero compute gives bit-identical blueprint probabilities. Fixed-update
repetitions with the same RNG seed are deterministic on the same runtime;
wall-time repetitions can complete different numbers of updates. The
eight repetitions describe this variation and sampled optimization, not
eight additional independently trained blueprints.

Exact evaluation is conditional on each saved realization of the stochastic
resolver. Averaging these exact exploitabilities does **not** give the
exploitability of a player that secretly mixes resolver realizations during
play. The latter is a different estimand; this audit makes no such claim.

Opt-out bounds are **not a blanket safety guarantee** for finite-budget,
regularized neural optimization. Exact exploitability and per-root bound
excess are calculated **after** solving and are never used for filtering,
blending, early stopping or choosing a checkpoint. All degradation is retained.

## Evaluation and interpretation

Primary: exact **full-game exploitability = NashConv / 2**, after stitching
every public-root continuation. Also report:

- Exact two-seat EV against the unchanged source blueprint.
- Exact EV against a fixed panel of all five saved 36h Experiment 35 policies;
  per-opponent values identify the own-seed member. A separate mean excludes
  that member, retaining the four other training seeds.
- Changes from the unchanged blueprint against the same panel.
- RNaD-minus-CFR+ differences at each matched time budget.
- Nested resolver variation, worst repeat/seed, bound excess and latency.

Average repetitions **within training seed first**. Across-seed summaries
use five inferential units, mean/SD/SE and pointwise 95% t intervals; the
RNaD-versus-CFR+ comparison also reports exact two-sided seed-level sign-flip
tests. These small-sample intervals/tests are not multiplicity-adjusted
confirmation of every cell. Lower exploitability is needed to support
equilibrium improvement; beating the blueprint alone does not establish it.
Do not pick the best repetition or retrospectively report the most favourable
budget as the prespecified primary result.

The fixed-update diagnostic and value-fit metrics help interpret a negative
timed result, but do not by themselves prove that RNaD cannot work. A poor
value fit, too few accepted updates, or large fallback rate calls for an
engineering/optimization review before making an algorithmic claim.

## Local checks and cloud execution

From the architecture repository root:

```bash
python3 -m pytest -q tests/test_rnad_ucv_resolving.py tests/test_exp49_rnad_pipeline.py
bash gcp/run_exp49_rnad_resolving.sh smoke-local
```

Cloud submission requires committing and pushing the implementation first.
It uses **one `n2-standard-16` CPU VM**, with up to four independent,
single-threaded case processes. No GPU, new training VMs or controller job
is used. Both solvers receive the same process/thread allocation. The
cloud script runs the synthetic smoke test before reading real snapshots.

```bash
export PROJECT_ID="clever-overview-399515"
export REGION="europe-west1"
export BUCKET="gs://clever-overview-399515-leduc-poker-dream-results"
# Reuse the existing working Leduc Batch SA_EMAIL; do not use a placeholder.
export REPO_REF="$(git rev-parse HEAD)"  # must already be pushed
export RUN_ID="exp49-rnad-$(date -u '+%Y%m%d-%H%M%S')"
export MODE=pilot
export WORKERS=4
bash gcp/run_exp49_rnad_resolving.sh dry-run
bash gcp/run_exp49_rnad_resolving.sh run
```

The pilot produces a measured full-run runtime extrapolation in
`analysis/summary.json`. `MAX_HOURS=24` is a wall-clock safety ceiling, **not**
a runtime estimate or training duration. Review the CPU pilot's correctness,
accepted-update counts, latency and critic diagnostics before proceeding:

```bash
export PILOT_RUN_ID="$RUN_ID"
export RUN_ID="exp49-rnad-$(date -u '+%Y%m%d-%H%M%S')"
export MODE=full
bash gcp/run_exp49_rnad_resolving.sh run
```

Full mode refuses to run without a completed pilot with identical code,
dependencies, source hashes and protocol. No job is launched automatically
after the pilot. Source and destination buckets can differ via `SOURCE_URI`.

For manual local execution, the equivalent commands are:

```bash
python3 -m experiments.leduc_poker.rnad_ucv_resolving.run fetch --source-root outputs/exp49_source
python3 -m experiments.leduc_poker.rnad_ucv_resolving.run run \
  --source-root outputs/exp49_source --output-root outputs/exp49_pilot --mode pilot --workers 4
python3 -m experiments.leduc_poker.rnad_ucv_resolving.run run \
  --source-root outputs/exp49_source --output-root outputs/exp49_full --mode full --workers 4 \
  --pilot-output outputs/exp49_pilot
```

`--resume` explicitly reuses completed, checksum-verified cases under the
identical manifest. Without it, nonempty output directories are rejected.
Aggregation requires every prespecified case; missing or failed cases cannot
silently disappear. `SUCCESS.json` is published to GCS only after other
outputs have uploaded successfully.

## Artefacts and deviations from training-output conventions

- `experiment_metadata.json`, `manifest.json`: configuration, sources, code/
  runtime provenance and complete expected task list.
- `preparation/`: separate MC critic weights/diagnostics and unchanged playable
  blueprint tables; these are small neural models, not training reservoirs.
- `policies/*.npz`: complete, reload-tested playable information-set policies.
- `tasks/*.json`: all raw per-case metrics and public-root diagnostics.
- `analysis/`: seed-level and paired summaries, per-root timing, runtime
  extrapolation, figure and completion summary.

`evaluate.load_policy(path, blueprint)` returns an OpenSpiel-compatible
tabular policy without pickle. Preserve these policies for later checks.
No UCV training states or replay buffers are saved. Compact tables are a
Leduc evaluation artefact, not a claim that full-game tabulation scales.

The source cohort deliberately differs from the historical Experiment 28
default: this question concerns the confirmed **Experiment 35** policy.
There are no new training-node trajectories, training AUCs or neural
36-hour fits. Test-time budgets replace those coordinates. The figure shows
pointwise 95% training-seed intervals rather than training-curve SE bands.
These deviations are recorded in experiment metadata.
