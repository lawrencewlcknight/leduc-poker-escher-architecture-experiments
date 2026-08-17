# Experiment 18: parallel UCV-ESCHER equivalence

This experiment tests whether the strongest validated UCV-ESCHER configuration
from Experiment 7 can be parallelised without materially changing solution
quality. The sequential and parallel arms use the same three seeds, 15-million-
node stopping rule, network architecture, optimiser settings, replay capacities,
train-step budgets and evaluation method.

## Parallel architecture

The implementation is synchronous and retains one authoritative learner:

1. At the start of each traverser's collection phase, the driver freezes and
   publishes the cumulative-regret, instantaneous-regret, three Q-target and
   residual-calibration inference networks.
2. Three persistent, CPU-only Ray actors receive that one object-store snapshot.
   The fixed traversal budget is partitioned exactly; it is never multiplied by
   worker count.
3. Every trajectory is assigned a global ID before dispatch. Its Q replay is
   returned to `trajectory_id % 3`, preserving the cross-fitted fold contract.
4. Workers return only newly generated regret, average-policy, Q-fold and
   calibration samples. All persistent replay and reservoir decisions remain in
   the driver, so aggregate replay capacity is identical to the sequential arm.
5. The driver merges the batches and performs the same ordered regret updates.
   Once both traversers are complete, the three disjoint Q critics and the
   residual-calibration learner train concurrently. They read frozen regret
   networks and write disjoint model, optimiser and replay state, so this does
   not introduce asynchronous gradients.

Worker batch arrays are preallocated from OpenSpiel's maximum Leduc game length
and the largest exact traversal partition. This avoids million-row replay copies
on every actor. PyTorch intra-op threads are divided over concurrent control
learners to avoid CPU oversubscription. Ray model snapshots are placed in the
object store once per traverser rather than serialised separately per actor.

The parallel run is not expected to reproduce floating-point-identical policies:
worker RNG streams and minibatch interleaving differ. The appropriate claim is
statistical practical equivalence, assessed on paired training seeds.

## Equivalence protocol

The primary metrics are final exact exploitability (`NashConv / 2`) and final
average-policy value. The parallel-minus-sequential paired differences use
pre-declared absolute margins:

- exploitability: `0.02`;
- average-policy value: `0.01`.

The report includes every paired delta and a 90% confidence interval/two one-
sided-tests summary. With only three seeds this is a validation experiment, not
a high-powered proof of equivalence; the per-seed effects and confidence interval
must be reported alongside the Boolean TOST result.

Systems outputs include initialisation, training and end-to-end speedups, worker
collection time, driver collection latency, snapshot-sync time, replay-merge
time, concurrent-control-learner time and peak worker-result payload size.

## Production run

```bash
python -m experiments.leduc_poker.ucv_escher_parallel_equivalence.run
```

Experiment 7 measured approximately 11.22 hours per sequential UCV-ESCHER seed.
Before measuring the new implementation, the parallel arm is conservatively
estimated at 8 hours per seed. Six isolated runs, plotting and process startup
are expected to require approximately **64 hours** on `n2-standard-8`. Configure
an **84-hour (5,040-minute)** Batch timeout.

```bash
JOB_NAME="leduc-escher-arch-exp18-ucv-parallel-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_escher_parallel_equivalence.run \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-8 302400 8000 32000 100
```

## Local smoke test

```bash
python -m experiments.leduc_poker.ucv_escher_parallel_equivalence.run \
  --seeds 0 \
  --target-nodes 50 \
  --traversals 4 \
  --max-iterations 2 \
  --advantage-train-steps 1 \
  --policy-train-steps 1 \
  --q-train-steps 1 \
  --calibration-train-steps 1 \
  --batch-size 2 \
  --buffer-size 128 \
  --evaluation-frequency 1 \
  --early-evaluation-nodes 10 \
  --parallel-num-workers 2 \
  --parallel-ray-object-store-memory 268435456 \
  --output-root outputs/smoke_tests/ucv_escher_parallel_equivalence
```

## GCP Batch smoke test

```bash
JOB_NAME="leduc-escher-arch-exp18-ucv-par-smoke-$(date -u +%Y%m%d-%H%M%S)"

./gcp/submit_batch_experiment.sh \
  "$JOB_NAME" \
  "python -m experiments.leduc_poker.ucv_escher_parallel_equivalence.run \
     --seeds 0 \
     --target-nodes 50 \
     --traversals 4 \
     --max-iterations 2 \
     --advantage-train-steps 1 \
     --policy-train-steps 1 \
     --q-train-steps 1 \
     --calibration-train-steps 1 \
     --batch-size 2 \
     --buffer-size 128 \
     --evaluation-frequency 1 \
     --early-evaluation-nodes 10 \
     --parallel-num-workers 2 \
     --parallel-ray-object-store-memory 268435456 \
     --output-root outputs/cloud/$JOB_NAME" \
  n2-standard-4 21600 4000 16000 100
```

## Outputs

- `seed_variant_summary.csv`: final quality, replay and timing metrics;
- `checkpoint_curves.csv`: node-aligned intermediate exact evaluations;
- `paired_differences_and_speedups.csv`: paired quality deltas and speedups;
- `paired_equivalence_summary.json`: pre-declared CI/TOST assessment;
- `aggregate_summary.json`: mean, standard deviation, standard error and range;
- `exploitability_by_nodes.png`: primary learning-quality comparison;
- `exploitability_by_wall_clock.png`: learning efficiency comparison;
- `final_exploitability.png`: paired-arm final quality summary;
- `end_to_end_runtime.png`: primary systems comparison;
- `worker_logs/`: isolated logs for every variant/seed run.
