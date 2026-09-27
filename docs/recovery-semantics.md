# Recovery semantics

## Completed update boundary

`global_step` and `optimizer_updates` count successful optimizer updates. The scheduler advances only after an update succeeds. `consumed_batches` counts microbatches already used, including batches from an AMP update skipped for nonfinite gradients. Checkpoints are taken only after a complete gradient accumulation window (`accumulation_phase=0`). `next_data_step` equals consumed batches, which can differ from optimizer updates. All ranks must agree on both counters.

Format 2 records model and optimizer DCP state, scheduler, optional FP16 scaler, Python/NumPy/Torch CPU RNG and the bound CUDA device RNG, counters, data/configuration fingerprints, source digest, package and PyTorch versions. CPU BF16 has no scaler. CUDA binds `LOCAL_RANK`; fixed logical rank/device topology is part of this contract.

## Eligible transaction

Every expected rank and DCP file must be regular, with the listed size and SHA-256. `COMMITTED` contains the manifest digest. Each candidate is fully verified before load; an uncommitted, corrupted, incompatible or mixed-boundary candidate is rejected. Selection validates candidates from newest to oldest and stops at the first valid version. `audit-checkpoints` explicitly checks full history. The file transaction remains authoritative if publication precedes SQLite indexing.

Version 0.2.1 refuses a symbolic link at the checkpoint root before candidate creation, selection, commit or retention. Commit hashes each payload once, checks file identity around the read and after publication, and verifies the published manifest and marker. Recovery independently rereads all payloads before loading. A retention byte budget requires `keep_last_k`; otherwise the configuration is rejected before launch.

Async save permits one request in flight. Default CPU staging is synchronous; a separate staging response is awaited before optimizer mutation. A dedicated Gloo save group and independent Gloo control group separate checkpoint traffic from training. At identical update boundaries, all ranks coordinate Future completion, error and deadline status. Ready uploads commit before the next scheduled save point. Forced waits and final flush use the same collective failure rule. Commit publication synchronizes files and parent directories; actual host power-loss behavior remains untested.

If rank 0 encounters an exception while publishing a checkpoint, it broadcasts the failure before leaving the control collective. Every rank then exits that save with an error. The real filesystem's ENOSPC/EIO and power-loss behavior remain separate environment tests.

Retention is disabled by default. `keep_last_k >= 2` keeps verified fallback versions, protects a loading candidate, and excludes unfinished candidates. Optional `max_retained_bytes` is a soft budget: protected versions take precedence and unmet budgets are reported. Durable deletion intent is retried after interruption, but deletion pauses when fewer than two valid fallbacks remain. Lifecycle operations run under the controller's exclusive ownership.

## Data contract

Synthetic IDs remain deterministic and disjoint per batch. JSONL input consists of `{"tokens": [integer, ...]}` rows, at least `sequence_length+1` tokens each. The configuration fixes the content SHA-256. Epoch permutations and optional random crop depend on seed/epoch/row, independent of worker scheduling. Distributed sampling pads to equal rank lengths, so real-data padding may repeat a row; a final batch can be smaller. Prefetch is reconstructed from the consumed position; no global worker RNG or arbitrary third-party augmentation state is supported. Changing dataset contents is rejected before launch.

This implementation supports this explicit map-style data adapter with 0–16 workers. General iterable datasets, stateful third-party transforms, tokenizers and their worker queues require separate state providers and acceptance.

## Completion and recovery

A fresh full process group restores the selected boundary. Every attempt has its own identity and logs. The controller checks a strict final summary, rank state digests/counters, all rank completion records and complete effective update/batch evidence before success. Explicit resume rechecks completed runs under a lock, refuses live launchers/orphan workers, and rejects source/runtime/configuration changes. All exceptions after spawn enter group cleanup; controller errors remain terminal diagnostics rather than becoming an unrelated retry error.

Validation reconstructs effective update and consumed-batch sequences after each recorded rollback. Final rank state digests, counters and scaler must match the uninterrupted reference exactly (`atol=0`, `rtol=0`), in the same device/runtime/topology. Bad records and missing evidence fail with diagnostics. Only an incomplete trailing line from a failed attempt can be ignored. Omission controls deliberately use a wrong RNG, optimizer or sample position and must be detected.

Version 0.2.2 also checks that each recovery attempt has one matching recovery decision and that every rank which made progress records a single state load and training start at the selected update and consumed-batch boundary. The training-start path must match the selected checkpoint. Exact cross-run comparison rejects different format 2 source or runtime identities. The source digest now covers installed package Python files consistently across checkout and wheel; Python and resolved dependency versions remain separate resume constraints. This evidence is application-generated and is not a tamper-resistant signature.

## Compatibility and infrastructure gates

Format 1 and earlier run schemas are explicitly rejected by this release. Historical files remain unchanged; replay requires their recorded source and runtime in a separate checkout. DCP cross-version compatibility is not assumed and checkpoints are never silently relabeled or migrated.

This is a single-host controller. Multi-node retry ownership/fencing, scheduler restart, remote immutable object transactions, elastic world sizes, full disk loss and host power-loss durability are separate implementation and infrastructure gates. CUDA DDP/FSDP2 code requires successful acceptance on actual devices before a tested-support claim.
