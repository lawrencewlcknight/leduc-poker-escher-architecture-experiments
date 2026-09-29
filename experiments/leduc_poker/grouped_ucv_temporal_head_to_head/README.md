# Experiment 45: grouped UCV-ESCHER temporal head-to-head

This is **architecture-repository Experiment 45**, not standard-ESCHER
Experiment 45 in the separate repository. It is evaluation-only: no training,
policy refitting, reservoir downloads, or new seeds.

## Question and fixed design

Does later training of the best-supported grouped-policy UCV configuration
improve direct play against its earlier policies, and does that improvement
continue during the late 24–36-hour window?

The source is Experiment 35, run `exp35-confirm-20260916-011231):

- Five source seeds: 470892, 385626, 145871, 902492, 318362.
- Fixed beta 1, two critics, prediction disabled, four-fit critic averaging.
- Grouped replay-mass-weighted soft-target cross-entropy, shared 136×136×136
  average-policy network, reset fitting, 20,000 updates at learning rate 0.003.
- All 18 saved time checkpoints: 2, 4, ..., 36 active hours.
- Each later checkpoint plays every earlier checkpoint from the **same seed**.
  This gives 153 pairs per seed, 765 pairs total, evaluated in both seat
  assignments (1,530 exact seat-specific expectations).

OpenSpiel enumerates Leduc exactly. The effect is
0.5 × [later-policy value as player 0 + later-policy value as player 1].
Positive values favour the later policy. There is no sampled hand count or
match-sampling uncertainty. Policies are tabulated once per checkpoint; only
one seed's tables are held in memory at a time.

This is deliberately time-indexed, not a reproduction of the old 3/6/9/12/15M
node schedule. Actual checkpoint times, completed iterations and training nodes
are preserved. The node plot uses mean checkpoint coordinates, **not** newly
matched-node policies. No checkpoint is chosen based on observed head-to-head
performance.

## Inference

The independent unit remains the five training seeds, not 765 matches.
The primary estimand is each seed's mean later-versus-earlier EV, then averaged
over seeds. Secondary summaries cover adjacent policies and final-versus-first.
The shared Experiment 16 statistics report mean, SD, SE, 95% t intervals,
positive-seed fractions and exact one-sided sign-flip p-values.
Checkpoint-pair tests receive Holm correction across all 153 pairs.

The 24–36-hour sensitivity tables use the seven late checkpoints and are
secondary/exploratory; their aggregate p-values do not form a new confirmatory
claim. With five seeds, the minimum one-sided exact p-value is 1/32.
This is retrospective evaluation of existing confirmation policies, **not**
a new independently trained confirmation. Direct-play gains and lower
exploitability are different outcomes.

## Safety and provenance

All 90 required policies must be present: no incomplete seed is silently
dropped. Source hashes are checked before loading trusted pickles; candidate,
architecture, fitting settings, seed, iteration and checkpoint times are
verified. Frozen configurations must match across snapshots. Every reloaded
policy's exact exploitability must reproduce its original diagnostic within
1e-5. Two-seat self-play must have zero EV. Tests independently check seat
antisymmetry. Hashes, original commit IDs, evaluator commit, versions and
resolved design are recorded in experiment_metadata.json.

Never load untrusted pickle files. This runner is intended for your own saved
Experiment 35 outputs.

## Local real-checkpoint smoke test

Use the repository's Python environment with Torch/OpenSpiel installed and
the usual gcloud login. From the repository root:

```bash
./gcp/run_grouped_ucv_temporal_head_to_head.sh smoke-local
```

This downloads only three real policies (seed 470892 at 2, 4 and 36 hours),
plus small analysis manifests, and performs all three two-seat matchups and
chart/table generation. No model is trained. Output is a timestamped directory
under outputs/exp45_smoke_*. A single-seed smoke result is not thesis evidence.

## Cloud run

First commit and push this experiment. Reuse PROJECT_ID, REGION, BUCKET and
SA_EMAIL from previous runs. Explicitly replace RUN_ID and REPO_REF to avoid
inheriting a previous experiment's values:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP35_RUN_ID="exp35-confirm-20260916-011231"
export RUN_ID="exp45-h2h-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_grouped_ucv_temporal_head_to_head.sh run
```

One on-demand n2-standard-4 VM downloads approximately 16 MB of neural
checkpoints plus manifests, performs the real-checkpoint smoke gate, runs all
five seed evaluations sequentially, aggregates, and uploads the outputs.
There is no controller or child-job IAM requirement. The existing runner
service account needs source-object read access, destination write access and
the normal Batch VM permissions. The launcher checks that it exists.

The laptop can disconnect after submission. The VM task has a four-hour
timeout (including setup) and zero automatic retries; this is a safety ceiling,
not the expected duration. Allow roughly 10–30 minutes including provisioning
and dependency installation, subject to cloud throughput.

```bash
./gcp/run_grouped_ucv_temporal_head_to_head.sh status
mkdir -p "cloud_outputs/$RUN_ID/analysis"
gcloud storage cp -r "${BUCKET%/}/$RUN_ID/analysis/*" "cloud_outputs/$RUN_ID/analysis/"
gcloud storage cp "${BUCKET%/}/$RUN_ID/experiment_metadata.json" \
  "${BUCKET%/}/$RUN_ID/SUCCESS.json" "cloud_outputs/$RUN_ID/"
```

The download commands assume BUCKET includes gs://. Source artefacts are never
modified. If the task fails, partial per-seed results and the log are uploaded
when the process can run its exit trap; abrupt VM termination can prevent this.
Use a new RUN_ID to rerun the cloud evaluation. This still never retrains.
For local interrupted evaluations, reuse the output root: completed seed
results are reused only with identical input hashes and evaluator code.
Completed output roots are protected against accidental replacement.

## Local full evaluation / exact reaggregation

```bash
python3 -m experiments.leduc_poker.grouped_ucv_temporal_head_to_head.run fetch \
  --source-root outputs/exp45_source --source-uri "$BUCKET/$EXP35_RUN_ID"
python3 -m experiments.leduc_poker.grouped_ucv_temporal_head_to_head.run run \
  --source-root outputs/exp45_source --output-root outputs/exp45_full
```

The fetch step downloads no reservoirs or training states. A source root
containing the original analysis/ and workers/ layout is also accepted.

## Principal outputs

Under analysis/:

- head_to_head_later_vs_earlier.png — mean two-seat EV heatmap.
- head_to_head_adjacent.png and head_to_head_final_vs_earlier.png.
- exploitability_by_training_time.png and exploitability_by_nodes.png.
- checkpoint_metrics.csv and checkpoint_summary.csv — full source coordinates,
  recomputed exploitability and reload checks; means, SDs and SEs.
- head_to_head_pairwise.csv — both seat-specific values and their average.
- seed_summary.csv, head_to_head_inference_summary.csv,
  head_to_head_pairwise_inference.csv, and late_window_* tables.
- checkpoint_strength.csv and best_checkpoint_summary.csv — round-robin
  strength versus exploitability, including whether their best checkpoints differ.
- aggregate_summary.json — design, primary statistics and completion counts.

The figures use means ± one SE across seeds; individual exploitability
trajectories are also shown. Exact evaluator tree visits do not alter the saved
training-node counts.
