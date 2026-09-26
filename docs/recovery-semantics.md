# Recovery semantics

`global_step` is the count of completed optimizer updates. The scheduler update for that step has also completed. `next_data_step` identifies the next deterministic sample batch each rank will consume. A valid checkpoint has equal `global_step` and `next_data_step` on every rank.

An eligible checkpoint contains DCP model parameters, buffers, and optimizer state; each rank's scheduler and Python/NumPy/CPU Torch random state; the step and next data position; configuration and data fingerprints; fixed world size; and package and PyTorch versions. Every expected rank and DCP file must exist, be a regular file, and match the size and SHA-256 digest listed in `manifest.json`. `COMMITTED` must contain that manifest's digest. An incomplete, uncommitted, incompatible, or checksum-invalid candidate is rejected before DCP load. The newest valid older candidate is selected if necessary.

Recovery starts a new process group at the selected completed update. It can recompute updates that were executed after the checkpoint in a failed attempt. The validator discards that rolled-back suffix, then compares the effective per-rank sample IDs for every step against an uninterrupted reference. It also requires exact final model, optimizer, scheduler, and completed-step equality on this deterministic CPU workload (`atol=0`, `rtol=0`).

If the controller exits, explicit resume first checks for live launchers and workers owned by the run, including an unrecorded launcher and orphaned workers whose launcher has exited. A file lock blocks simultaneous controllers. The copied run configuration and committed manifests control recovery even if the original configuration file or SQLite checkpoint index has changed. A completed attempt is reconciled from its matching final attempt summary before another attempt or retry is allocated. Event evidence is audited separately by `validate`.

The process-failure contract does not claim durability after entire-disk loss or host power failure. The implementation fixes `world_size` and keeps data-loading workers at zero.
