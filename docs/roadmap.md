# Roadmap

## 1. Reference workload

- [x] Launch two CPU DDP workers with Gloo.
- [x] Generate stable, disjoint sample IDs without a data download.
- [x] Record per-rank progress and a final model digest.
- [x] Repeat the reference run and verify identical final model state.

## 2. Complete checkpoint transaction

- [x] Save model and optimizer with DCP at an update boundary.
- [x] Save each rank's scheduler, RNG, step, and next data position.
- [x] Verify expected files, sizes, hashes, and configuration fingerprint.
- [x] Publish a manifest and commit marker only after validation.
- [x] Reject incomplete and corrupted candidates and fall back to an older version.

## 3. Recovery controller

- [x] Add SQLite run, attempt, checkpoint, and recovery records.
- [x] Detect worker exit and stalled step progress with bounded timeouts.
- [x] Restart the full group with a new attempt ID and limited retries.
- [x] Support explicit resume after controller exit with process ownership checks.
- [x] Reject stale messages from earlier attempts.

## 4. Validation and experiments

- [x] Add deterministic fault hooks for worker exit, save interruption, corruption, and hang.
- [x] Compare recovered final state and effective sample sequence with a reference.
- [x] Add negative tests for omitted RNG, optimizer, and data cursor state.
- [x] Benchmark sync and native async DCP with repeated runs and raw results.
- [x] Produce a reproducible report with environment and limitations.

GPU and FSDP experiments are independent extensions after the CPU recovery path passes.
