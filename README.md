# TrainGuard

TrainGuard tests whether fixed-size distributed PyTorch training resumes from a consistent update boundary. A successful recovery must restore the model, optimizer, scheduler, rank-local random state, completed update count, and next data position. The validator compares the final state and effective sample sequence with an uninterrupted run.

## Status

The CPU recovery path is implemented and tested with two Gloo workers. It supports synchronous and native asynchronous Distributed Checkpoint (DCP), application-level checkpoint commits, bounded full-group restarts, explicit resume, deterministic fault injection, correctness validation, and repeated local benchmarks. CUDA DDP and FSDP2 adapters are implemented but require real-device acceptance. Multi-node/object-storage recovery and host-power-loss durability remain open gates.

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

Validation writes `validation.json` in the recovered run directory. It checks exact SHA-256 digests for model, optimizer, and scheduler state, the completed step, and each rank's effective sample sequence after removing rolled-back work. It also rejects duplicate or out-of-order steps, invalid sample IDs, missing successful-rank logs, and invalid final summaries in either run. A failed attempt may end with a truncated final log line. The fixed CPU comparison tolerance is `atol=0, rtol=0`. A mismatch exits with status 1.

If a controller exits after a committed checkpoint, resume its run directory with:

```bash
uv run trainguard resume runs/<run-id>
```

Resume refuses to start while the previous owned launcher or worker group is still running, including the interval before a launcher PID is recorded. A run with no valid committed checkpoint fails clearly. Configuration is copied into the run directory so later changes to the original YAML do not alter recovery.

## Checkpoints and records

Each candidate is stored under `runs/<run-id>/checkpoints/step-<step>-<attempt>/`. DCP writes model and optimizer state. Every rank writes its scheduler, Python/NumPy/Torch CPU and optional CUDA RNG, scaler, update and consumed-batch counters, and compatibility fingerprints. Rank 0 validates all expected files, records sizes and SHA-256 hashes in `manifest.json`, then publishes `COMMITTED`. Recovery scans these files and ignores incomplete, incompatible, or corrupted candidates, falling back to the newest valid older checkpoint.

The run directory also contains `run.json`, a SQLite index (`run.sqlite3`), `launcher.log`, per-attempt rank event logs and summaries, and a final `summary.json`. The committed manifest is the checkpoint validity source if the controller stops before updating SQLite.

## Experiments

Run five repetitions each without checkpoints, with synchronous DCP, and with native asynchronous DCP on the longer CPU workload:

```bash
uv run trainguard benchmark --config configs/cpu_benchmark.yaml --output-root runs --repetitions 5 --warmups 1
uv run trainguard benchmark --config configs/cpu_benchmark_large.yaml --output-root runs --repetitions 5 --warmups 1
```

This writes raw `results.json` and a Markdown report under `runs/benchmark-<id>/`. Warm-up runs are validated and stored separately from measured runs; `--warmups 0` disables them. Both total elapsed time and the worker training window are measured, with committed checkpoint payload bytes and host load snapshots. Every run is checked against the uninterrupted reference. The two configurations vary model and checkpoint size while keeping steps and save frequency fixed; the [two-size CPU report](docs/experiments/cpu-sizes-2026-09-25.md) records 18 measured runs and 6 warm-ups. The [extended CPU experiment](docs/experiments/cpu-extended-2026-09-25.md) includes earlier raw measurements, environment, method, and limits; the [four-step smoke test](docs/experiments/cpu-2026-09-25.md) remains available. Asynchronous saves use a separate communication group so checkpoint traffic can overlap training without mixing collective operations.

The fault configuration supports `worker_exit`, `save_interrupt`, `corrupt`, and `hang` on the first attempt. `recovery.omit_state` can deliberately omit `rng`, `optimizer`, or `cursor` restoration for negative validation experiments.

## Development

```bash
uv run ruff check .
uv run pytest
```

See the [roadmap](docs/roadmap.md), [architecture](docs/architecture.md), [recovery semantics](docs/recovery-semantics.md), and [experiment protocol](docs/experiment-protocol.md) for the acceptance contract and limitations.

## License

MIT. See [LICENSE](LICENSE).

## Upgrade workflow

Version 0.2 adds strict completion evidence and cleanup, early async commit, upload deadlines and stage metrics, retryable retention, resumable benchmarks, JSONL data with deterministic shuffle/crop and prefetch, gradient accumulation, CPU BF16, optional CUDA FP16 scaler, and CUDA DDP/FSDP2 adapters. The default dependency is PyTorch 2.14. GPU adapters require actual-device acceptance; multi-node/object-storage control and power-loss durability remain open infrastructure gates. Format 2 rejects older schemas, source changes and runtime changes; historical evidence remains intact.

```bash
uv run trainguard acceptance --config configs/cpu_demo.yaml
uv run trainguard acceptance-resume runs/acceptance-<id>
uv run trainguard run --config configs/cpu_data_demo.yaml
uv run trainguard benchmark-resume runs/benchmark-<id>
uv run trainguard audit-checkpoints runs/<id>
uv run trainguard storage-benchmark --repetitions 3
```

On a host with two CUDA GPUs:

```bash
uv run trainguard acceptance --config configs/cuda_ddp.yaml
uv run trainguard acceptance --config configs/cuda_fsdp2.yaml
uv run pytest tests/test_gpu_acceptance.py
```

The JSONL example contains illustrative token rows, not a real-corpus performance claim. Supply an immutable token file and its SHA-256 for real training. See [current acceptance and remaining gates](docs/experiments/full-upgrade-2026-09-26.md).
