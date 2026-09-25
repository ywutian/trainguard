# Architecture

## Runnable baseline

`trainguard run` validates the YAML configuration, records a run ID and configuration fingerprint, and launches one `torchrun` process group with zero framework restarts. Each rank trains the same model through DDP on disjoint, deterministic samples. Rank events are written to separate JSONL files. Rank 0 writes a final summary after all ranks complete.

The launcher bounds the total attempt time and terminates its process group on timeout. The baseline does not select checkpoints or restart workers.

## Recovery architecture to implement

The controller will be the only recovery decision maker. Workers will save model and optimizer state through PyTorch Distributed Checkpoint, plus rank-local scheduler, RNG, step, and data cursor state. A candidate checkpoint will be eligible for recovery only after all expected rank state is present, file hashes match a manifest, and the application-level commit marker is published.

The controller will monitor worker exit and step progress, stop the old process group, select the newest valid checkpoint, and launch a new attempt with a distinct ID. Events and completion messages must carry run, attempt, rank, and step identifiers so a late message from an older attempt cannot alter the current run.

SQLite will index runs, attempts, checkpoints, and recoveries. Committed checkpoint manifests remain the source of truth for checkpoint validity if the controller exits between a file commit and a database update.
