# Algorithm provenance and scope

The alternating-player V-trace recurrence and NeuRD force in `rnad.py` are
adapted from OpenSpiel's RNaD implementation (Copyright 2019 DeepMind
Technologies Limited), distributed under Apache-2.0, reproduced in
`LICENSE.openspiel`:

https://github.com/google-deepmind/open_spiel/blob/931e39a/open_spiel/python/algorithms/rnad/rnad.py

Changes: independent CPU PyTorch implementation, fixed blueprint regularizer,
Leduc public-range continuation sampler, separate MC-pretrained value network,
opponent opt-out gadget, soft-deadline policy snapshots and exact evaluation.
The original JAX trainer, rolling regularizer schedule, and discretization/
thresholding of the deployed strategy are not used. This is not a full
reproduction of DeepNash or the 2026 paper's implementation.

Methodological references:

- Perolat et al., *Mastering the game of Stratego with model-free multiagent
  reinforcement learning* (RNaD / DeepNash): https://arxiv.org/abs/2206.15378
- Hennes et al., *Neural Replicator Dynamics*:
  https://arxiv.org/abs/1906.00190
- Kubicek, Lisy and Sandholm, *Test-time Reinforcement Learning in Imperfect
  Information Games*: https://arxiv.org/abs/2608.30635
- Burch, Johanson and Bowling, *Solving Imperfect Information Games Using
  Decomposition*: https://arxiv.org/abs/1303.4441

Exact public ranges and opponent counterfactual best-response values are a
deliberate small-game diagnostic advantage. Finite-budget neural optimization,
regularization, and imperfect value fitting do not carry a blanket safety
guarantee. All degradation is retained and measured; the evaluation oracle is
never used to select or repair the output policy.
