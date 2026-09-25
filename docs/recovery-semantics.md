# Recovery semantics

This document describes the intended recovery contract. The current baseline has no checkpoint or recovery implementation.

`global_step` means the number of completed optimizer updates. A checkpoint is taken after the optimizer and scheduler updates for that step. A rank's data cursor identifies the next sample it would read.

An eligible checkpoint must include model parameters and buffers, optimizer state, scheduler state, the step, rank-local RNG states, rank-local data cursor, configuration and data fingerprints, world size, and software version. All entries must correspond to the same logical step. The first implementation keeps `world_size` fixed and uses `num_workers=0` for data loading.

Recovery may recompute steps after the selected checkpoint. Validation must discard the failed attempt's rolled-back suffix before comparing the effective sample sequence with an uninterrupted reference run. Missing, uncommitted, incompatible, or checksum-invalid checkpoints must not be loaded. If no valid checkpoint remains, the run fails clearly rather than restarting silently from step zero.

The process-failure contract does not claim durability after entire-disk loss or host power failure.
