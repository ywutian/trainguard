# Experiment protocol

## Correctness

1. Run a fixed-seed, uninterrupted CPU reference with the same configuration and worker count.
2. Run a fault-injected attempt, recover from a committed checkpoint, and finish at the same step.
3. Compare final model, optimizer, scheduler, step, and effective sample sequence. Enable dropout for the RNG negative test.
4. Deliberately omit RNG, optimizer, and data cursor restoration in separate tests to prove the validator detects each error.

The recovery test suite and validator are planned, not yet implemented. Numeric tolerance must be set before comparing runs and recorded with the results.

## Performance

Compare runs without checkpoints, with synchronous DCP, and with native asynchronous DCP using identical workload, save interval, and validation rules. Repeat each configuration at least three times and report raw measurements, median, and range. Keep CPU, GPU, and simulated storage-delay results separate. Include time spent staging, writing, checksumming, committing, restarting, and recomputing rolled-back steps.

No performance results are claimed in this repository until those experiments are run.
