# Experiment 42: row-wise average-policy continuation

## Research question

Experiment 42 asks whether the Experiment 41 improvement can be recovered by
ordinary row-sampled continuation without sorting, deduplicating or grouping
the replay reservoir by information set. It distinguishes additional neural
optimisation from Experiment 41's uniform information-set weighting.

## Frozen design

The five Experiment 29 `time_36h` policies and reservoirs are reused. Every fit
warm-starts the archived network and uses a fresh Adam optimiser, learning rate
`3e-4`, gradient clipping at `10.0`, iteration-weighted soft-target
cross-entropy and uniform replay-row sampling with replacement.

Two new arms are evaluated:

- `rowwise_equal_examples` processes exactly 1,700 times the frozen
  Experiment 41 information-set count for that source seed (approximately
  1.58 million examples);
- `rowwise_equal_updates` performs 1,700 ordinary 2,048-row optimiser updates
  (3,481,600 examples).

Experiment 41 is imported as the immutable grouped full-batch reference. No
grouping operation is permitted in the new training worker. Its per-seed
information-set counts and source checksums are frozen from Experiment 41's
validated manifest.

The primary comparison is row-wise equal-example minus grouped exploitability
at the fixed 1,700-equivalent endpoint. A `0.0025` upper practical
non-inferiority margin is declared before running. The five saved source
reservoirs remain the inferential units.

Exact exploitability and the exact tabular average are Leduc-only post-hoc
diagnostics. Training consumes only sampled replay rows, legal-action masks,
iteration labels and the archived neural policy.

## Local smoke test

```bash
./gcp/run_average_policy_rowwise_continuation.sh smoke-local
```

The smoke test creates a miniature Experiment 29 archive, builds an Experiment
41 grouped reference, runs both row-wise arms, verifies exact resource
accounting and playable policy reloads, and completes joined aggregation.

## GCP run

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export EXP41_RUN_ID="exp41-fit-20260918-180815"
export RUN_ID="exp42-row-$(date -u '+%Y%m%d-%H%M%S')"
export PARALLELISM=5
./gcp/run_average_policy_rowwise_continuation.sh run
```

Five `n2-standard-4` workers run one source reservoir each. The cloud-owned
controller performs smoke, fitting and aggregation, so the laptop can be
disconnected after submission. `status` inspects the run and `resume` reuses
completed workers after an infrastructure or aggregation failure.

## Principal outputs

- `rowwise_vs_grouped_exploitability.png`;
- `paired_final_effects.csv` and `paired_final_summary.csv`;
- `combined_exploitability_metrics.csv`;
- `final_resource_tradeoff.csv`;
- `worker_manifest.csv` and `snapshot_inventory.csv`.
