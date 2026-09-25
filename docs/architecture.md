# Architecture

## Training and control

`trainguard run` validates and snapshots configuration, creates a run directory and SQLite index, then launches one `torchrun` process group at a time with `--max-restarts=0`. The controller alone selects checkpoints and decides whether to restart. Every attempt has a distinct ID and its own event files and summary. Rank events include run, attempt, rank, and completed-step IDs; the controller reads only the current attempt and rejects mismatched IDs.

The workload uses fixed-size CPU DDP with Gloo, one deterministic token batch per rank and completed update, and `num_workers=0`. Each step performs an optimizer update, then a scheduler update, then records the next data position. `global_step` counts completed optimizer updates. Model, optimizer, scheduler, RNG, step, and cursor are captured at that boundary.

## Checkpoint transaction

`checkpoint_io.py` uses PyTorch DCP to save model and optimizer state. Synchronous save finishes before training continues. Native asynchronous save stages state, allows one save in flight, and waits for its future before starting another. DCP operations use a dedicated Gloo group, separate from DDP training collectives. Every rank writes a JSON file containing scheduler, Python/NumPy/CPU Torch RNG, completed step, next data step, and compatibility data.

The candidate directory is never eligible while writing. After all ranks finish, rank 0 verifies rank-local state and DCP files, hashes every payload file, writes `manifest.json`, and publishes `COMMITTED` containing the manifest hash. `checkpoint.py` validates the marker, manifest, expected file set, sizes, hashes, fingerprints, world size, software versions, rank states, and step before returning a candidate. The controller scans every candidate and chooses the highest valid step; a corrupt newest candidate does not hide an older valid one.

## Recovery

`controller.py` monitors worker exit, per-rank step progress, an attempt deadline, and a progress deadline. On failure it stops the process group, scans committed manifests, records the decision in SQLite, and starts a new full group from the selected step. Retry count is bounded. If no valid checkpoint remains, the run fails rather than starting from step zero.

`trainguard resume` acquires a per-run file lock. It checks the stored launcher PID and process start identity, then scans the launcher's process group for workers matching the run directory, run ID, and attempt ID. A live owned group causes an error, even if the launcher itself has exited. The controller also stops orphaned workers before an automatic retry. If the prior group has exited, it reconciles completed attempts and reserved attempt directories, scans checkpoint files into SQLite, and resumes. A per-attempt summary prevents a late prior attempt from replacing the final run summary.

SQLite indexes runs, attempts, checkpoint inspections, and recoveries. The on-disk committed manifest remains the authority for checkpoint validity if a controller exits between file publication and a database update.

## Scope

The first implementation is single-node, fixed-world-size CPU training. It does not claim GPU or FSDP support, multi-node recovery, elastic world-size changes, or durability after host power failure or complete disk loss.
