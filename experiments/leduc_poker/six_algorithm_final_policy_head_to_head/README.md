# Experiment 17: Six-Algorithm Final-Policy Head-to-Head

## Research question

How do the best approximately 15-million-node configurations of Deep CFR,
DREAM, ESCHER, VR-DeepDCFR+, VR-DeepPDCFR+ and UCV-ESCHER perform in direct
play?

## Training and snapshot contract

The common training seeds are `1234`, `2025`, `31415`, `27182`, and `16180`.
The experiment archives the already-trained final policies from Deep CFR
Experiment 27, DREAM Experiment 43, ESCHER Experiment 43, and UCV-ESCHER
Experiment 16. It trains only the two missing VR arms to the first complete
outer iteration crossing 15,000,000 nodes. Their configuration is imported
unchanged from Experiment 7, including the authors' reported Leduc
parameterisation and the documented implementation corrections.

Every selected input snapshot is copied into the new run directory and recorded
with its SHA-256 checksum. The output is therefore self-contained and can be
analysed again without the original experiment directories.

## Exact match protocol

Leduc is small enough to evaluate policies exactly. Each match evaluates
algorithm A as player 0 and as player 1 and reports the mean of those two exact
expected values. No games are sampled, so there is no match-level Monte Carlo
error and no arbitrary game-count or stopping rule.

The primary analysis contains all 15 unordered algorithm pairs for each of the
five paired training seeds. A secondary descriptive league evaluates all 25
cross-seed policy combinations for every algorithm pair. These 25 combinations
are not treated as independent observations: the independent training seed is
the inferential unit.

For each pair, the runner reports the mean exact EV, a 95% t interval, an exact
two-sided paired sign-flip test, an exploratory directional sign-flip test, and
a Holm correction across all 15 pairwise tests. Five seeds impose an important
limit: the smallest attainable two-sided exact sign-flip p-value is `2/32 =
0.0625`. Therefore this experiment can report effect sizes, intervals and
cross-seed consistency, but cannot establish conventional two-sided `p < 0.05`
significance from the paired seed test. More training seeds would be required
for that claim.

## Runtime estimate

The completed Experiment 7 runs on `n2-standard-8` measured:

| Algorithm | Hours per 15M-node seed | Five sequential seeds |
| --- | ---: | ---: |
| VR-DeepDCFR+ | 6.03 | 30.15 |
| VR-DeepPDCFR+ | 7.03 | 35.15 |
| **VR training total** | **13.06** | **65.3 hours** |

Snapshot staging and exact evaluation should be small compared with training.
Plan for approximately **66 hours measured** and **70--80 hours conservatively**
on one `n2-standard-8` VM. The standard 96-hour Batch limit (`345600` seconds)
provides appropriate headroom.

## Local full run

When this repository is inside the existing `deep_cfr_v3` workspace, the
runner automatically finds the four audited source experiment directories:

```bash
python -m experiments.leduc_poker.six_algorithm_final_policy_head_to_head.run
```

For a portable snapshot bundle, create these subdirectories and pass its root:

```text
SNAPSHOT_ROOT/
  deep_cfr/
  dream/
  escher/
  unbiased_control_variate_escher/
```

```bash
python -m experiments.leduc_poker.six_algorithm_final_policy_head_to_head.run \
  --snapshot-root /path/to/SNAPSHOT_ROOT
```

Analysis can be repeated without retraining:

```bash
python -m experiments.leduc_poker.six_algorithm_final_policy_head_to_head.run \
  analyse \
  --smoke \
  --seeds 1234 \
  --run-dir outputs/six_algorithm_final_policy_head_to_head/RUN_DIRECTORY
```

Omit `--smoke --seeds 1234` for the five-seed production run.

## Local smoke test

This uses one existing seed and trains both VR arms with deliberately tiny
budgets. Its numerical results are not scientific.

```bash
python -m experiments.leduc_poker.six_algorithm_final_policy_head_to_head.run \
  --smoke \
  --seeds 1234 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --early-evaluation-nodes 10 \
  --output-root outputs/smoke_tests
```

## Full GCP Batch job

The four existing snapshot jobs are distributed across the Deep CFR, DREAM and
ESCHER repository result buckets. The Batch wrapper stages each audited source
from its repository-specific bucket and then starts the runner with the staged
snapshot root. Use the project-wide standard Batch configuration:

```bash
JOB_NAME="leduc-escher-arch-exp17-six-algorithm-h2h-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "bash gcp/run_experiment_17.sh \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

The runner is resumable when invoked again with `--run-dir`: completed VR
worker results and matching snapshots are reused by checksum.

## GCP smoke test

```bash
JOB_NAME="leduc-escher-arch-exp17-six-algorithm-h2h-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "bash gcp/run_experiment_17.sh \
     --smoke \
     --seeds 1234 \
     --target-nodes 50 \
     --traversals 4 \
     --max-iterations 2 \
     --advantage-train-steps 1 \
     --policy-train-steps 1 \
     --q-train-steps 1 \
     --batch-size 2 \
     --buffer-size 128 \
     --early-evaluation-nodes 10 \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 345600 8000 32000 100
```

## Principal outputs

| File | Contents |
| --- | --- |
| `snapshot_inventory.csv` | All 30 selected policies, nodes/checkpoints and checksums. |
| `vr_training_curves.csv` | Exploitability, nodes and wall-clock trajectory for the ten new VR runs. |
| `vr_training_summary.csv` | Final VR training metrics and snapshot provenance. |
| `final_policy_metrics.csv` | Recomputed exact exploitability and self-play value for every final policy. |
| `head_to_head_same_seed_pairwise.csv` | Primary 15 pairs × 5 seeds, evaluated in both seats. |
| `head_to_head_pairwise_inference.csv` | Seed-level intervals, exact p-values and Holm correction. |
| `head_to_head_cross_seed_league.csv` | Secondary 15 pairs × 25 seed combinations. |
| `algorithm_strength_by_seed.csv` | Each policy's mean EV against the other five algorithms. |
| `algorithm_strength_summary.csv` | Five-seed league ranking and uncertainty. |
| `head_to_head_mean_ev_heatmap.png` | Antisymmetric mean exact-EV matrix. |
| `algorithm_strength.png` | Mean league strength with 95% intervals. |
| `final_exploitability.png` | Recomputed final exploitability comparison. |
| `snapshots/` | Archived existing policies and newly trained VR policies. |
