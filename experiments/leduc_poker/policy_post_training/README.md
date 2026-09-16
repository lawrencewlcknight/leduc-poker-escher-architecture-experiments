# Experiments 36--39: post-training policy improvement

These four development experiments start from the frozen 36-hour Experiment
29 policy and do not retrain UCV-ESCHER. Each uses all five Experiment 29 seed
trajectories, three pre-specified hyperparameter arms and exact Leduc
exploitability at a fixed update schedule.

- Experiment 36 performs supervised cross-entropy repair towards the empirical
  reservoir policy, with weights derived from visitation under exact opponent
  best responses. The three arms use best-response multipliers 1, 10 and 100.
- Experiment 37 performs policy-gradient ascent against each player's current
  exact best response, constrained by KL divergence from the frozen blueprint.
  The KL coefficients are 0.01, 0.1 and 1.0.
- Experiment 38 implements the NeuRD/Hedge update by applying detached
  counterfactual advantages directly to policy logits. It compares learning
  rates 0.0001, 0.0003 and 0.001.
- Experiment 39 is the deliberately conventional control: clipped PPO from
  sampled self-play returns, with a learned value baseline and the same three
  learning rates as Experiment 38.

Experiments 36--38 use exact Leduc best responses, reach probabilities or
counterfactual values to isolate whether the proposed objective has merit.
Those are development oracles, not a claim that the post-training procedure is
already scalable. Experiment 39 is model-free. A successful oracle method
would require a subsequent experiment replacing exact quantities with sampled
UCV or learned estimates.

At every recorded update the worker reports exact exploitability, KL distance
from the Experiment 29 blueprint and seat-averaged value against that
blueprint. It saves the final network and the minimum-exploitability development
checkpoint. Selecting a checkpoint or arm with exact exploitability makes all
four studies exploratory; a winner requires fresh-seed confirmation.

## Run

Each experiment has an independent launcher. Run its local smoke first and
then submit with the same GCP variables used by Experiment 35:

```bash
./gcp/run_best_response_guided_repair.sh smoke-local
./gcp/run_kl_exploitability_descent.sh smoke-local
./gcp/run_neurd_fine_tuning.sh smoke-local
./gcp/run_ppo_self_play.sh smoke-local
```

For any one experiment, set a fresh `RUN_ID`, retain the completed Experiment
29 source ID, and use the corresponding launcher:

```bash
export REPO_REF="$(git rev-parse HEAD)"
export EXP29_RUN_ID="exp29-ce-20260912-171556"
export PARALLELISM=5

export RUN_ID="exp36-brrepair-$(date -u '+%Y%m%d-%H%M%S')"
./gcp/run_best_response_guided_repair.sh run
```

Use `exp37-kled`, `exp38-neurd` or `exp39-ppo` as the run prefix for the other
launchers. The remote controller performs cloud smoke, five-worker training and
aggregation, so the laptop can be disconnected after submission. Pass `status`
or `resume` to the same experiment-specific launcher.

Each worker uses an on-demand `n2-standard-4`, has a 12-hour hard ceiling and
one automatic retry. Retry workspaces are isolated. Aggregation excludes saved
policy weights from its download and writes the compact charts and tables under
`analysis/`.

