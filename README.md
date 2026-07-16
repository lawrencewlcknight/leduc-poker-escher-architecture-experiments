# Leduc Poker ESCHER Architecture Experiments

This repository contains model-free ESCHER architecture experiments for Leduc
poker. It was created as a clean successor to the original ESCHER experiment
repository so that new architectural changes can be evaluated with the same
metrics, plots, seeds, and artifact conventions used in the MPhil thesis.

The only historical experiment retained is **Experiment 28**, the strongest
validated ESCHER configuration at the point this repository was created. It is
the control against which all new architecture experiments should be assessed.

## Baseline contract

The canonical baseline is defined in
`experiments/leduc_poker/escher_candidate_architecture_multiseed/config.py`.
Its important fixed properties are:

- OpenSpiel game: `leduc_poker`;
- seeds: `1234`, `2025`, `31415`, `27182`, and `16180`;
- 80 iterations, 500 regret traversals, and 500 value traversals per iteration;
- `(256, 256, 128)` policy, regret, and value trunks;
- a 64-unit per-action regret head;
- standardised legal-action regret targets;
- exact exploitability reported as `NashConv / 2`;
- node-touch and wall-clock accounting retained alongside exploitability.

Do not edit this baseline in place for a new hypothesis. Create a new experiment
from `experiments/leduc_poker/escher_architecture_base.py` and record only the
architectural difference. This keeps comparisons auditable and prevents
baseline drift.

## Repository layout

```text
escher_poker/                         Shared solver, networks, metrics, and plots
experiments/leduc_poker/
  escher_candidate_architecture_multiseed/  Experiment 28 baseline
  escher_vs_vr_deep_cfr_matched_nodes/      Three-seed matched-node comparison
  escher_architecture_base.py               Baseline-copy helper
  escher_variant_config_utils.py            Derived-config validation
  escher_variant_ablation_runner.py         Multi-variant experiment runner
  escher_single_seed_variant_runner.py      Single-seed diagnostic runner
tests/                                 Unit and baseline-contract tests
docs/                                  Output, cloud, and thesis conventions
scripts/promote_thesis_artifacts.py     Curates lightweight thesis artifacts
outputs/                               Untracked working output
thesis_artifacts/                      Tracked, curated result artifacts
```

## Setup

The code targets Python 3.9.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Run the Experiment 28 baseline

Full five-seed run:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run
```

Fast wiring smoke test:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run \
  --seeds 1234 \
  --iterations 2 \
  --traversals 2 \
  --value-traversals 2 \
  --policy-network-train-steps 1 \
  --regret-network-train-steps 1 \
  --value-network-train-steps 1 \
  --evaluation-interval 1 \
  --batch-size-regret 2 \
  --batch-size-value 2 \
  --batch-size-average-policy 2 \
  --memory-capacity 128 \
  --output-root outputs/smoke_tests
```

The smoke test verifies the entry point and export pipeline; it is not a useful
performance estimate.

## Run the matched-node algorithm comparison

The comparison with VR-DeepDCFR+ and VR-DeepPDCFR+ uses the paper's Leduc
training settings, evaluates each VR outer iteration, and stops each VR seed at
the first iteration crossing the paired Experiment 28 node count:

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.run
```

See
`experiments/leduc_poker/escher_vs_vr_deep_cfr_matched_nodes/README.md` for the
comparison contract, upstream provenance, expected memory requirements, and a
fast wiring test.

## Add an architecture experiment

Start every new experiment by calling:

```python
from experiments.leduc_poker.escher_architecture_base import make_default_config

config = make_default_config("leduc_poker_escher_my_architecture")
```

Then change only the fields required by the hypothesis, give each variant a
stable ID and human-readable label, and reuse the shared runner and plotting
utilities. New architectural mechanisms belong in `escher_poker/`; experiment
packages should contain configuration and orchestration rather than forked
solver implementations.

See `docs/OUTPUT_CONVENTIONS.md` before adding metrics or figures. See
`TESTING.md` for verification and `docs/GCP_BATCH_EXPERIMENTS.md` for cloud runs.

## Thesis artifacts

Raw outputs and cloud downloads remain outside Git. Promote reviewed plots,
tables, aggregate summaries, and provenance metadata with:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

The selected files are copied under
`thesis_artifacts/<experiment_name>/<run_directory_name>/` with a promotion
manifest.
