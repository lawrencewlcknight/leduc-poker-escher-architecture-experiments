# Output Conventions

All architecture experiments must preserve the measurement and presentation
contract below. This makes new results directly comparable with Experiment 28
and with the existing MPhil thesis figures.

## Run layout and provenance

Write each run to a timestamped directory under `outputs/`:

```text
outputs/<experiment_name>_<YYYYMMDD_HHMMSS>/
```

Every completed run should include `experiment_metadata.json` containing the
experiment name, full resolved configuration, variant definitions, seeds,
timestamp, command/module, package versions where available, and relevant Git
commit. Failed or partial work should remain distinguishable from completed
summaries.

Use stable, filesystem-safe `variant_id` values and separate human-readable
`variant_label` values. Never use plot labels as join keys.

## Canonical Leduc metrics

- Use OpenSpiel's `leduc_poker` implementation.
- Compute exact NashConv with OpenSpiel for reported checkpoints.
- Report heads-up exploitability as `NashConv / 2`, including that definition in
  column names, axis labels, and metadata where ambiguity is possible.
- Use player 0's Leduc Nash-equilibrium value,
  `-0.085606424078`, as the average-policy-value target.
- Preserve cumulative `nodes_touched` from the shared solver and cumulative
  `wall_clock_seconds`; do not substitute iterations for computational work.
- Record final, best, final-window, and area-under-curve metrics using the shared
  utilities rather than reimplementing them in an experiment package.

## Seeds and comparisons

The full baseline seed set is:

```text
1234, 2025, 31415, 27182, 16180
```

Architecture comparisons should use matched seeds and the same training budget
as Experiment 28 unless the research question explicitly concerns budget. For
each scalar aggregate, report count, mean, standard deviation, and standard
error. For treatment-versus-baseline claims, also export per-seed paired deltas
and their aggregate summary. Define deltas as `treatment - baseline`, so negative
exploitability deltas are improvements.

## Common machine-readable outputs

Where applicable, retain these names and schemas:

- `experiment_metadata.json` — full provenance and resolved configuration;
- `seed_summary.csv` — one row per variant and seed;
- `checkpoint_curves.csv` — one row per variant, seed, and evaluation checkpoint;
- `aggregate_summary.json` — aggregate scalar metrics;
- `variant_aggregate_summary.csv` — long-form variant/metric aggregates;
- `paired_differences_vs_baseline.csv` — per-seed matched deltas;
- `paired_difference_summary.csv` and `.json` — aggregate paired deltas;
- `summary.json` — completion status and links to principal summaries.

Long-form rows should include `experiment_name`, `variant_id`, `variant_label`,
`seed`, and checkpoint coordinates where relevant. Preserve raw precision in CSV
and JSON; rounding belongs in presentation layers.

## Plots

Use the shared plotting and chart-title utilities. Every multi-seed curve should
show the mean and standard error and, when readable, faint individual seed
curves. Use consistent colours for the baseline across figures.

At minimum, a full multi-seed architecture comparison should provide:

- exploitability by nodes touched;
- exploitability by iteration;
- average-policy value by nodes touched and iteration, with the Nash target;
- policy-value error;
- final exploitability by variant or seed;
- runtime or node-budget comparison when architectures have different cost.

Label exploitability axes `Exploitability (NashConv / 2)`. Plot the Nash target
as a black dashed line. Figures intended for the thesis should be PNG at 200 dpi
or greater, use the shared typography, and remain legible at thesis column width.

## Thesis promotion

Raw run directories, checkpoints, replay buffers, traces, and logs remain
untracked. After reviewing a run, promote only lightweight figures, tables,
summary JSON, and provenance metadata with
`scripts/promote_thesis_artifacts.py`. See `THESIS_ARTIFACTS.md` for selection
rules.

Any intentional deviation from this document must be recorded in the new
experiment's README and metadata so it cannot be mistaken for a like-for-like
comparison.
