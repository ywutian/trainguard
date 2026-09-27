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

The guarded profile additionally requires two distinct save boundaries, reserves local free space before each save, and rejects a complete checkpoint whose payload, manifest and commit marker exceed its declared size. Its final audit requires the current attempt's final-step candidate and at least two structurally verified retained candidates. It separately counts every file under the checkpoint tree, including unfinished or invalid candidates, and refuses to mark the run successful when actual tree bytes exceed the configured local budget or the free-space floor is lost. These are application-level checks, not a hard filesystem quota; unstructured launcher output and other processes can still consume space. A structurally verified DCP candidate can fail during actual load, so two candidates do not establish two proven recoveries without a restore rehearsal on the target runtime.

## Data contract

Synthetic IDs remain deterministic and disjoint per batch. JSONL input consists of `{"tokens": [integer, ...]}` rows, at least `sequence_length+1` tokens each. The configuration fixes the content SHA-256. Epoch permutations and optional random crop depend on seed/epoch/row, independent of worker scheduling. Distributed sampling pads to equal rank lengths, so real-data padding may repeat a row; a final batch can be smaller. Prefetch is reconstructed from the consumed position; no global worker RNG or arbitrary third-party augmentation state is supported. Changing dataset contents is rejected before launch.

This implementation supports this explicit map-style data adapter with 0–16 workers. General iterable datasets, stateful third-party transforms, tokenizers and their worker queues require separate state providers and acceptance.

## Completion and recovery

A fresh full process group restores the selected boundary. Every attempt has its own identity and logs. The controller checks a strict final summary, rank state digests/counters, all rank completion records and complete effective update/batch evidence before success. Explicit resume rechecks completed runs under a lock, refuses live launchers/orphan workers, and rejects source/runtime/configuration changes. All exceptions after spawn enter group cleanup; controller errors remain terminal diagnostics rather than becoming an unrelated retry error.

Validation reconstructs effective update and consumed-batch sequences after each recorded rollback. Final rank state digests, counters and scaler must match the uninterrupted reference exactly (`atol=0`, `rtol=0`), in the same device/runtime/topology. Bad records and missing evidence fail with diagnostics. Only an incomplete trailing line from a failed attempt can be ignored. Omission controls deliberately use a wrong RNG, optimizer or sample position and must be detected.

In guarded runs, the customer supplies a private HMAC key outside the run directory. Sample events carry ordered keyed commitments and event MACs instead of raw sample IDs; validation requires the same key and rejects missing or altered evidence. The key holder can re-sign an event, so this is privacy and accidental-tamper detection within the customer trust boundary, not independent authorship proof. Other run files and application output remain subject to customer data controls.

Version 0.2.2 also checks that each recovery attempt has one matching recovery decision and that every rank which made progress records a single state load and training start at the selected update and consumed-batch boundary. The training-start path must match the selected checkpoint. Exact cross-run comparison rejects different format 2 source or runtime identities. The source digest now covers installed package Python files consistently across checkout and wheel; Python and resolved dependency versions remain separate resume constraints. This evidence is application-generated and is not a tamper-resistant signature.

## Compatibility and infrastructure gates

Format 1 and earlier run schemas are explicitly rejected by this release. Historical files remain unchanged; replay requires their recorded source and runtime in a separate checkout. DCP cross-version compatibility is not assumed and checkpoints are never silently relabeled or migrated.

This is a single-host controller. Multi-node retry ownership/fencing, scheduler restart, remote immutable object transactions, elastic world sizes, full disk loss and host power-loss durability are separate implementation and infrastructure gates. CUDA DDP/FSDP2 code requires successful acceptance on actual devices before a tested-support claim.

## Version 0.3.0 selection, durability and simulation contract

The manifest and `COMMITTED` marker remain necessary but no longer sufficient for recovery selection. Selection and retention parse DCP metadata, validate every referenced shard and range, and decode referenced payload fragments with CPU-only, weights-only loading. A hash-consistent but unloadable newest candidate is skipped in favor of a valid older version. Ordinary save publication retains structural and digest checks without repeating this decode; the stronger read belongs to recovery selection and retention. The extra decode may increase recovery I/O and memory usage at scale.

Before rank 0 publishes a checkpoint, all ranks now synchronize their event log files and parent directories and collectively confirm success. This orders durable step/sample evidence before the checkpoint marker. Completion events are synchronized before a worker returns success. The guarantees depend on the selected filesystem honoring these calls; a local test does not establish behavior during host power loss.

An async Future that is cancelled or fails is converted into a collective failure decision. A completed Future whose callback timestamp is still absent after its deadline is treated conservatively as expired. Ten-case campaign success additionally requires one matching fault-injection event, exactly one recovery, and the expected failed/succeeded attempt sequence.

The remote object and epoch module is a standalone executable protocol model. Its store contract requires atomic conditional writes, strongly consistent reads and listing, and a revision token that cannot repeat after an object changes and later regains identical bytes. A real service adapter must establish this contract, including unknown-result retries, rather than assuming a provider's content-derived ETag has revision semantics. Its isolation oracle represents independent proof that earlier actors can no longer run. The current training controller does not consume that protocol; multi-host acceptance remains open.
