# Experiment 28: retained architecture baseline

This is the unchanged control configuration for all experiments in this new
architecture repository. It trains the best validated ESCHER architecture from
the original experiment series over the same five fixed development seeds.

The candidate combines:

- deep plain `(256, 256, 128)` policy, regret, and value trunks;
- no LayerNorm and no residual trunk connections;
- a 64-unit per-action regret head;
- standardised legal-action regret targets, without clipping;
- the validated full-training protocol from Experiment 28.

The configuration is defined locally rather than importing historical
experiment packages. Its values remain the exact Experiment 28 defaults.

Default full run:

```bash
python -m experiments.leduc_poker.escher_candidate_architecture_multiseed.run
```

Useful smoke-test settings:

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

The smoke-test output checks only that the experiment entry point, candidate
configuration, plotting, and summary exports are operational. It is not an
estimate of ESCHER performance.
