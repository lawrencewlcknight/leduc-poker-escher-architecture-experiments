# Thesis Artifact Promotion

Full experiment output directories under `outputs/` are scratch or working data.
They can contain large checkpoints, policy snapshots, logs, traces, replay files,
NumPy arrays, and failed-run tracebacks. Those files should generally stay out of
git.

Curated thesis-facing outputs live under:

```text
thesis_artifacts/<experiment_name>/<run_directory_name>/
```

Only lightweight artifacts should be promoted into this tracked tree: graph
images, CSV tables, aggregate summary JSON files, and experiment
metadata/provenance.

## Promote Artifacts

After downloading cloud outputs locally, run the promotion script from the repo
root:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

Preview what would be copied without changing files:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
```

Replace already-promoted files for a run:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --overwrite
```

You can also pass a specific run directory or a local parent directory under
`outputs/`:

```bash
python scripts/promote_thesis_artifacts.py outputs/smoke_tests
python scripts/promote_thesis_artifacts.py outputs/my_experiment_run_20260517_120000
```

The script searches recursively for `experiment_metadata.json`, treats each
matching directory as a completed run, and copies selected files into
`thesis_artifacts/`. Each promoted run receives a `promotion_manifest.json` that
records source paths, destination paths, selected files, skipped files, the
promotion timestamp, whether the command was a dry run, and whether overwrite was
enabled.

## Default Selection Rules

Included by default:

- `*.png`
- `*.csv`
- `aggregate_summary.json`
- `paired_difference_summary.json`
- `paired_aggregate_summary.json`
- `best_checkpoint_summary.json`
- `experiment_metadata.json`

Excluded by default:

- `*.pt`
- `*.pth`
- `*.npz`
- `*.log`
- `failed_seeds.json`
- `failed_runs.json`
- `checkpoints/*`
- `snapshots/*`
- `traces/*`

Extra comma-separated include or exclude globs can be supplied when needed:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME \
  --include "summary/*.json" \
  --exclude "debug/*"
```

## Why Promotion Is Local

Cloud Batch jobs should not push directly to git. A local promotion step keeps
the workflow conservative:

- full outputs remain available for inspection before anything enters the repo;
- heavyweight or sensitive run byproducts are filtered out;
- thesis artifacts can be reviewed with `git diff`;
- commits can group related results intentionally;
- failed or partial cloud jobs cannot accidentally mutate the repository.

The cloud job’s responsibility is to run experiments and upload complete working
outputs. The local machine’s responsibility is to curate the lightweight thesis
artifacts worth tracking.
