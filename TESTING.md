# Testing and Smoke Tests

Run checks from the repository root in the Python 3.9 environment described in
`README.md`.

## Static checks

```bash
python -m compileall -q escher_poker experiments scripts tests
python -m ruff check escher_poker experiments scripts tests
```

## Unit tests

```bash
python -m pytest
```

The baseline configuration test is a contract test: it deliberately asserts the
important Experiment 28 values so accidental baseline drift fails visibly.

## Experiment smoke test

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

Confirm that the run directory contains metadata, per-seed histories, aggregate
CSV/JSON summaries, and the standard exploitability plots. Do not compare the
smoke-test exploitability with the full baseline.

## Matched-node comparison smoke test

The ESCHER/VR-DeepCFR+ experiment has a separate one-seed wiring check because
the paper defaults are deliberately expensive:

```bash
python -m experiments.leduc_poker.escher_vs_vr_deep_cfr_matched_nodes.run \
  --seeds 0 \
  --escher-iterations 2 \
  --escher-traversals 2 \
  --escher-value-traversals 2 \
  --escher-evaluation-interval 1 \
  --escher-policy-train-steps 1 \
  --escher-regret-train-steps 1 \
  --escher-value-train-steps 1 \
  --escher-batch-size 2 \
  --escher-memory-capacity 128 \
  --vr-traversals 2 \
  --vr-max-iterations 3 \
  --vr-advantage-train-steps 1 \
  --vr-policy-train-steps 1 \
  --vr-baseline-train-steps 1 \
  --vr-batch-size 2 \
  --vr-buffer-size 128 \
  --output-root outputs/smoke_tests
```

Confirm that each VR summary reports the paired ESCHER target, a final node
count at or just above it, and at least one exact exploitability checkpoint.

## Full baseline

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run
```

A new architecture should first pass unit and smoke tests, then be compared with
Experiment 28 using the same five seeds and computational budget. Any intentional
departure must be recorded in experiment metadata and the experiment README.
