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
- [x] Detect a live launcher even if its PID has not yet been recorded.
- [x] Cover file-commit/index-update and worker-completion/status-publication exit windows.
- [x] Refuse resume while orphaned owned workers are alive.
- [x] Reject stale messages from earlier attempts.

## 4. Validation and experiments

- [x] Add deterministic fault hooks for worker exit, save interruption, corruption, and hang.
- [x] Compare recovered final state and effective sample sequence with a reference.
- [x] Independently reject duplicate, missing, invalid, and out-of-order event evidence.
- [x] Add negative tests for omitted RNG, optimizer, and data cursor state.
- [x] Benchmark sync and native async DCP with repeated runs and raw results.
- [x] Separate worker training time from launch overhead in a longer CPU experiment.
- [x] Exclude separately recorded warm-up runs from performance statistics.
- [x] Record checkpoint payload bytes and host load for two CPU model sizes.
- [x] Produce a reproducible report with environment and limitations.

GPU and FSDP experiments are independent extensions after the CPU recovery path passes.

## 5. Version 0.2 upgrade

- [x] Share strict event/final evidence audit between controller and validator.
- [x] Recheck successful resume and clean every post-spawn controller exception.
- [x] Commit ready async snapshots at common update boundaries, including coordinated failure/deadline handling on final flush.
- [x] Measure preparation, staging, upload, waiting, commit lag, rank training/resource peaks and fault-to-first-update RTO.
- [x] Add default-disabled retention, retryable deletion intent, loading protection and descending recovery scan.
- [x] Persist benchmark slots/failures, resume missing runs, balance six orders and retain paired differences/source/runtime.
- [x] Add immutable JSONL data, deterministic shuffle/crop, worker prefetch, tails/epochs and gradient accumulation.
- [x] Verify CPU BF16 and scaler skipped-update boundary behavior.
- [x] Implement CUDA DDP/FSDP2 bindings, rank-local CUDA RNG and shard-only state digests.
- [x] Add persistent acceptance campaigns, storage microbenchmarks and interval budget estimates.
- [x] Evaluate PyTorch 2.14 in isolation, update the dependency and explicitly version/reject incompatible evidence.
- [ ] Run CUDA FP32/BF16/FP16 and FSDP2 campaigns on at least two actual GPUs.
- [ ] Implement and accept scheduler/node lifecycle, fencing epoch and fixed multi-node recovery in the selected infrastructure.
- [ ] Implement and accept remote object-generation uploads/conditional manifest commit in the selected storage.
- [ ] Accept Linux filesystem power-loss behavior with an independent crash/reboot rig.
- [ ] Establish stable performance differences on a dedicated host with two batches of at least 12 repetitions per mode.

The last five gates remain open; local CPU results are not substituted for them. See [upgrade acceptance](experiments/full-upgrade-2026-09-26.md).

## 6. Version 0.2.1 local hardening

- [x] Reject symbolic-link checkpoint roots before creation, selection, commit and retention.
- [x] Reject a retention byte budget when retention is disabled.
- [x] Hash payloads once during publication, detect changed files, and keep full validation before recovery load.
- [x] Recheck real CPU recovery, two-version retention, 99 tests, build and local 64/256 MiB stage measurements.
- [ ] Close the five infrastructure gates above when the required environments are available.

Evidence: [local hardening and measurement](experiments/local-hardening-2026-09-26.md).

## 7. Version 0.2.2 evidence and release identity

- [x] Reject success if a progressing recovery attempt lacks matching per-rank `state_loaded` or `training_started` evidence.
- [x] Cross-check recovery decision, selected path, update and consumed-batch boundary.
- [x] Reject exact cross-run comparison when format 2 source or runtime identities differ.
- [x] Make source identity stable between a source checkout and its installed wheel; distinguish unavailable Git state from a clean checkout.
- [x] Add a package import check and an automated CPU/static/build verification workflow.
- [x] Broadcast rank-0 commit failure to every rank before leaving the save collective; verify with two Gloo ranks.
- [ ] Run the workflow on its remote Linux runner and preserve its output as independent release evidence.
- [ ] Inject actual ENOSPC/EIO/readonly filesystem errors and verify recovery on the selected Linux storage.

The full supported-scope and infrastructure plan is in the [closure assessment](analysis/closure-assessment-2026-09-26.md).
The [0.2.2 local acceptance](experiments/closure-2026-09-26.md) archives the completed CPU campaign and package checks.
