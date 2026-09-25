# Roadmap

## 1. Reference workload

- [x] Launch two CPU DDP workers with Gloo.
- [x] Generate stable, disjoint sample IDs without a data download.
- [x] Record per-rank progress and a final model digest.
- [x] Repeat the reference run and verify identical final model state.

## 2. Complete checkpoint transaction

- [ ] Save model and optimizer with DCP at an update boundary.
- [ ] Save each rank's scheduler, RNG, step, and next data position.
- [ ] Verify expected files, sizes, hashes, and configuration fingerprint.
- [ ] Publish a manifest and commit marker only after validation.
- [ ] Reject incomplete and corrupted candidates and fall back to an older version.

## 3. Recovery controller

- [ ] Add SQLite run, attempt, checkpoint, and recovery records.
- [ ] Detect worker exit and stalled step progress with bounded timeouts.
- [ ] Restart the full group with a new attempt ID and limited retries.
- [ ] Support explicit resume after controller exit with process ownership checks.
- [ ] Reject stale messages from earlier attempts.

## 4. Validation and experiments

- [ ] Add deterministic fault hooks for worker exit, save interruption, corruption, and hang.
- [ ] Compare recovered final state and effective sample sequence with a reference.
- [ ] Add negative tests for omitted RNG, optimizer, and data cursor state.
- [ ] Benchmark sync and native async DCP with repeated runs and raw results.
- [ ] Produce a reproducible report with environment and limitations.

GPU and FSDP experiments are independent extensions after the CPU recovery path passes.
