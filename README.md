# TrainGuard

TrainGuard tests whether fixed-size distributed PyTorch training resumes from a consistent update boundary. A successful recovery must restore the model, optimizer, scheduler, rank-local random state, completed update count, and next data position. The validator compares the final state and effective sample sequence with an uninterrupted run.

## Status

The CPU recovery path is implemented and tested with two Gloo workers. It supports synchronous and native asynchronous Distributed Checkpoint (DCP), application-level checkpoint commits, bounded full-group restarts, explicit resume, deterministic fault injection, correctness validation, and repeated local benchmarks. GPU, FSDP, multi-node training, and host-power-loss durability remain outside this implementation.

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run trainguard validate-config --config configs/cpu_demo.yaml
uv run trainguard run --config configs/cpu_demo.yaml
uv run trainguard run --config configs/recovery_demo.yaml
```

The first run is an uninterrupted reference. The second injects a rank-0 exit after step 2, restarts the group from a committed checkpoint, and completes. Each command prints its run directory. Compare the two directories:

```bash
uv run trainguard validate --reference runs/<reference-id> --recovered runs/<recovered-id>
```

Validation writes `validation.json` in the recovered run directory. It checks exact SHA-256 digests for model, optimizer, and scheduler state, the completed step, and each rank's effective sample sequence after removing rolled-back work. The fixed CPU comparison tolerance is `atol=0, rtol=0`. A mismatch exits with status 1.

If a controller exits after a committed checkpoint, resume its run directory with:

```bash
uv run trainguard resume runs/<run-id>
```

Resume refuses to start while the previous owned launcher or worker group is still running, including the interval before a launcher PID is recorded. A run with no valid committed checkpoint fails clearly. Configuration is copied into the run directory so later changes to the original YAML do not alter recovery.

## Checkpoints and records

Each candidate is stored under `runs/<run-id>/checkpoints/step-<step>-<attempt>/`. DCP writes model and optimizer state. Every rank writes its scheduler, Python/NumPy/CPU Torch RNG, completed step, next data step, and compatibility fingerprints. Rank 0 validates all expected files, records sizes and SHA-256 hashes in `manifest.json`, then publishes `COMMITTED`. Recovery scans these files and ignores incomplete, incompatible, or corrupted candidates, falling back to the newest valid older checkpoint.

The run directory also contains `run.json`, a SQLite index (`run.sqlite3`), `launcher.log`, per-attempt rank event logs and summaries, and a final `summary.json`. The committed manifest is the checkpoint validity source if the controller stops before updating SQLite.

## Experiments

Run five repetitions each without checkpoints, with synchronous DCP, and with native asynchronous DCP on the longer CPU workload:

```bash
uv run trainguard benchmark --config configs/cpu_benchmark.yaml --output-root runs --repetitions 5
```

This writes raw `results.json` and a Markdown report under `runs/benchmark-<id>/`. Both total elapsed time and the worker training window are measured, and every run is checked against the uninterrupted reference. The [extended CPU experiment](docs/experiments/cpu-extended-2026-09-25.md) includes raw measurements, environment, method, and limits; the earlier [four-step smoke test](docs/experiments/cpu-2026-09-25.md) remains available. Asynchronous saves use a separate communication group so checkpoint traffic can overlap training without mixing collective operations.

The fault configuration supports `worker_exit`, `save_interrupt`, `corrupt`, and `hang` on the first attempt. `recovery.omit_state` can deliberately omit `rng`, `optimizer`, or `cursor` restoration for negative validation experiments.

## Development

```bash
uv run ruff check .
uv run pytest
```

See the [roadmap](docs/roadmap.md), [architecture](docs/architecture.md), [recovery semantics](docs/recovery-semantics.md), and [experiment protocol](docs/experiment-protocol.md) for the acceptance contract and limitations.

## License

MIT. See [LICENSE](LICENSE).
