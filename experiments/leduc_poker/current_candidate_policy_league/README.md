# Experiment 46: current-candidate Leduc policy leagues

Evaluation only. No CFR training, replay fitting, new seeds or best-checkpoint
selection. This replaces the outdated UCV entry in a **new retrospective
league**, without modifying the old Experiment 17 or frozen Experiment 19.
Update the thesis after this evaluation succeeds, not before its results exist.

## Frozen sources and endpoints

| Algorithm | Near-15M league | 36-active-hour league |
|---|---|---|
| Deep CFR | Architecture 17 archived selected Deep CFR policies | Architecture 21 Deep CFR |
| SD-CFR | Deep CFR repository 29, uniform historical mixture | Same run, final endpoint |
| DREAM | DREAM repository 46, grouped CE | Same run, final endpoint |
| ESCHER | Standard ESCHER repository 49, grouped CE | Same run, final endpoint |
| VR-DeepDCFR+ | Architecture 17 author-configuration policies | Not available |
| VR-DeepPDCFR+ | Architecture 17 author-configuration policies | Not available |
| UCV-ESCHER | Architecture 35, grouped-wide neural candidate | Same run, 36h endpoint |

The UCV candidate has fixed beta=1, two critics, no predictor, four-fit critic
averaging, and a reset 136×136×136 average-policy network fitted with grouped,
replay-mass-weighted soft-target CE (20,000 updates, learning rate 0.003).
Its policy is the saved neural network, **not** the empirical replay or exact
weighted tabular average. SD-CFR alone uses its algorithm-defined exact
own-reach-weighted uniform historical mixture, reconstructed from saved models.

All arms contribute five training seeds. The source identifiers and seed sets
are fixed in `sources.py`; changing shell `RUN_ID` or `BUCKET` cannot select a
different source experiment. `BUCKET` controls the destination only.

The primary league uses 35 policies: 21 algorithm pairs × 25 cross-seed pairs
= 525 matchups. The secondary 36h league uses 25 policies and 250 matchups.
Total: **775 matchups, 1,550 exact seat-specific expected values**. Self-play
and sign checks add diagnostic evaluations. Chance and action branches are
enumerated exactly; there is no Monte Carlo hand-count error.

### Comparability limits that must stay visible

- The historical selected Deep CFR policies stop at 14.88–14.96M nodes.
  Other near-15M policies are first completed checkpoints crossing 15M.
  Actual counts and overshoots are reported; no playable policy is interpolated.
- The 36h league uses each experiment's active-training clock. DREAM/ESCHER
  output fitting was deferred and costs additional time. It is not an equal
  total-VM-cost league. It has no VR arms and must not be pooled with the 15M league.
- These are retrospectively selected configurations, not a newly randomized
  seven-arm confirmation. Different seed cohorts are not forcibly paired.
- Each algorithm receives equal weight in league strength, and each opposing
  seed equal weight. The 25 matches per algorithm pair are **not 25 independent
  training replicates**. Intervals use 10,000 training-seed cluster-bootstrap
  draws; shared historical seed labels are resampled together, independent
  cohorts separately. Intervals are pointwise exploratory, not multiplicity
  adjusted. No significance-selected checkpoint or ranking is used.

## Validation and resources

The launcher validates all 60 endpoint records before submitting. The VM first
runs a real one-seed-per-arm smoke (12 policies, 31 matchups) covering both
endpoints and every loader. No incomplete seed set is silently dropped.
Source GCS object generations, file SHA-256 hashes, original metadata and
evaluator commits are preserved. Existing source checksums are verified before
unpickling trusted artifacts. Every reloaded policy must reproduce original
exploitability within 1e-5; legal-action probabilities, zero two-seat self-play
and algorithm/seat reversal are checked. SD-CFR inference code is pinned to
`1669e5af4cbc88c648626148fd9c395c2e5d4583` in its original repository.

One on-demand **n2-standard-8**, 80 GB boot disk, 8-hour hard task timeout,
**zero automatic retries**. The larger RAM allocation accommodates SD-CFR
archives; one archive/seed is reconstructed at a time. There is no training
controller and no child-job permission requirement. Source archives are several
GB in total, but the downloadable analysis is small. No replay reservoirs are
downloaded. Runtime includes dependency setup, downloads and SD-CFR mixture
reconstruction, not just the 775 fast tabular matches.

## Run on GCP

Run from the **ESCHER architecture repository**, after committing and pushing
the implementation. Reuse `PROJECT_ID`, `REGION`, `BUCKET` and `SA_EMAIL`.
The service account must read the existing DREAM, ESCHER and Deep CFR result
buckets, and write the destination bucket. Existing cross-repository evaluation
permissions may already provide this. The launcher checks the account exists;
VM smoke checks its actual source access before full evaluation.

```bash
export REPO_REF="$(git rev-parse HEAD)"
export RUN_ID="exp46-league-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_current_candidate_policy_league.sh run
```

The smoke is built into the same remote job. Laptop may disconnect after
submission. Do not use the DREAM Experiment 46 launcher: numbering is repo-local.

```bash
./gcp/run_current_candidate_policy_league.sh status
```

Optional metadata-only preflight (no paid VM, no policy downloads):

```bash
./gcp/run_current_candidate_policy_league.sh preflight
```

Optional local full real-checkpoint smoke needs Torch, OpenSpiel, NumPy, SciPy,
Matplotlib and h5py, plus the pinned Deep CFR checkout. It downloads about 1.4 GB
for the first SD-CFR archive, so cloud smoke is the default recommendation:

```bash
./gcp/run_current_candidate_policy_league.sh smoke-local
```

## Download and thesis update

After the evaluation job reports `SUCCEEDED`:

```bash
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp -r "${BUCKET%/}/$RUN_ID/analysis/*" "cloud_outputs/$RUN_ID/analysis/"
gcloud storage cp "${BUCKET%/}/$RUN_ID/SUCCESS.json" "cloud_outputs/$RUN_ID/"
```

These commands assume `BUCKET` includes `gs://`. No archive download is needed.
Principal outputs:

- `policy_metrics.csv`: actual nodes/time, source policy, reload error, exploitability.
- `pairwise_exact_values.csv`: both seat values for every cross-seed matchup.
- `head_to_head_summary.csv`: mean effects, cluster SEs and exploratory intervals.
- `algorithm_strength.csv` and `endpoint_summary.csv`.
- `node_15m/` and `time_36h/`: payoff heatmap, league-strength and exploitability charts.
- `aggregate_summary.json`: scientific limitations, provenance and completion counts.

The run also saves compact, exact tabulations of the **loaded output policies**
under `tables/` for future evaluation; these are inference artifacts, not an
alternative training algorithm. The full source manifests and package versions
are uploaded separately. Raw downloaded SD-CFR archives are not re-uploaded.

After the results arrive, update the summary's head-to-head figures/tables and
regenerate its learning-curve figures from original trajectories with consistent
time accounting. This league does not fabricate denser trajectory observations,
replace 15M figures with 36h numbers, or rewrite the thesis automatically.
