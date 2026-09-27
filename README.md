# TrainGuard

TrainGuard tests whether fixed-size distributed PyTorch training resumes from a consistent update boundary. A successful recovery must restore the model, optimizer, scheduler, rank-local random state, completed update count, and next data position. The validator compares the final state and effective sample sequence with an uninterrupted run.

## Status

The CPU recovery path is implemented and tested with two Gloo workers. It supports synchronous and native asynchronous Distributed Checkpoint (DCP), application-level checkpoint commits, bounded full-group restarts, explicit resume, deterministic fault injection, correctness validation, and repeated local benchmarks. CUDA DDP and FSDP2 adapters are implemented but require real-device acceptance. Multi-node/object-storage recovery and host-power-loss durability remain open gates.

For the proposed customer deployment, integration, operations, security and commercial acceptance scope, see the [product closure and release gates](docs/plans/product-closure-2026-09-26.md). The current release is an experiment package and has not passed those production gates.

The [customer pilot template](docs/commercial/customer-pilot-template.md), [operations runbook](docs/commercial/operations-runbook.md), and [machine-readable release gates](docs/commercial/release-gates.json) record the commercial scope and evidence required before customer production use. The existing code is MIT licensed; pilot fees cover agreed integration, validation and support services.
The [market evidence](docs/commercial/market-evidence-2026-09-26.md) separates current competitor facts from unvalidated positioning and pricing hypotheses.

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run trainguard validate-config --config configs/cpu_demo.yaml
uv run trainguard run --config configs/cpu_demo.yaml
uv run trainguard run --config configs/recovery_demo.yaml --allow-experiment
```

The first run is an uninterrupted reference. The second injects a rank-0 exit after step 2, restarts the group from a committed checkpoint, and completes. Each command prints its run directory. Compare the two directories:

An installed wheel can write its packaged CPU example outside the source checkout with `trainguard init-config --output cpu_demo.yaml`. A configuration with `run.profile: guarded` requires checkpointing and rejects fault injection or omitted recovery state. This guard does not imply production acceptance.

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

Raw run evidence can contain sample identifiers and paths. Keep it in a restricted customer-controlled directory. `trainguard support-bundle runs/<run-id> --output support.json` produces a read-only summary with allowlisted scalar fields; the customer should review and approve it before sharing.

For a paired trial cost calculation, fill in [the pilot ledger template](docs/commercial/pilot-ledger-template.json) with customer rates and measured results, then run `python scripts/calculate_pilot_value.py <ledger.json> --output <result.json>`. The result is conditional on the injected scenarios and is not a realized savings or billing record.

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

Version 0.2.1 rejects linked checkpoint roots and inactive retention budgets, and avoids a second full payload read at commit. See the [local hardening and measurement report](docs/experiments/local-hardening-2026-09-26.md). CUDA, multi-node, remote storage, power-loss and dedicated-host performance gates remain open.

Version 0.2.2 binds each recovery decision to every progressing rank's state-load and training-start evidence, rejects cross-run software identity mismatches, and broadcasts commit failures to all ranks. The source fingerprint is computed from package code so an installed wheel and its source checkout have the same identity; resolved runtime versions remain a separate resume check. Verify a built artifact with:

```bash
uv build --wheel --sdist --out-dir dist
uv run python scripts/verify_wheel.py dist/trainguard-0.3.1-py3-none-any.whl
uv run python scripts/verify_install.py dist/trainguard-0.3.1-py3-none-any.whl
```

See the [full closure assessment](docs/analysis/closure-assessment-2026-09-26.md) for the supported boundary, infrastructure gates and next acceptance steps.
The [0.2.2 local acceptance report](docs/experiments/closure-2026-09-26.md) contains the test, campaign, build and package evidence.

## Version 0.3.0 local simulation closure

Run the complete local gate from the repository root:

```bash
uv sync --locked --group dev
uv run python scripts/run_simulation_closure.py --output-root runs/simulation-closure --previous-ref a18ae9a
```

The gate retains static checks, the full test suite, JUnit and raw test recovery directories, a fresh ten-case CPU acceptance campaign, a wheel and source distribution, and a wheel/source identity check in a uniquely named result directory. Its `result.json` and `report.md` record the outcome even if a gate fails. The CPU campaign requires the intended fault, exactly one recovery, the expected attempt statuses, and either exact agreement with the uninterrupted reference or the specified negative-control difference.

The suite also exercises two independent local launchers with a real two-rank Gloo group, hard process exits across checkpoint publication boundaries, rank-coordinated async cancellation and deadlines, event-log durability ordering, and candidate fallback when DCP metadata or payload cannot be loaded. An isolated in-memory protocol model exercises immutable remote generations, conditional head publication, response-loss reconciliation, and epoch takeover after an independent isolation assertion. Bridge tests publish actual DCP checkpoint bytes through the model, damage the newer version, and restore the older one; a two-rank training run resumes from the downloaded version after all local candidates are removed. The model is not a production remote backend or a real object service.

See the [simulation analysis](docs/analysis/simulation-closure-2026-09-26.md) and [local result](docs/experiments/simulation-closure-2026-09-26.md). Actual CUDA devices, multiple hosts, a selected object service and isolation authority, and controlled host power loss still require their own acceptance runs.

## Version 0.3.1 commercial evaluation boundary

The package now includes an installed CPU template, a guarded configuration profile, explicit authorization for fault and omitted-state experiments at the controller entry, and a read-only allowlisted support export. The local gate installs the built wheel with locked dependencies in a new environment, verifies a real checkpoint recovery against a reference, checks the support export, and confirms the complete run directory survives uninstall. It also rehearses a source/runtime upgrade boundary using separate old and new locked environments: a new wheel rejects an interrupted old run, while the old environment resumes it to the reference result.

These checks support a limited, isolated evaluation package. Run the [release readiness checker](docs/commercial/operations-runbook.md) against the candidate wheel, source distribution, lock and reviewed gate receipts. A `BLOCKED` result prevents a production claim. The current real GPU, multi-host, customer-workload, object-service, security operations and paid-customer gates remain open.
The [version 0.3.1 local validation record](docs/commercial/evidence/local-validation-0.3.1.json) links the complete local gate outcomes to the candidate artifacts. An evaluation bundle remains labeled `EVALUATION_ONLY`; machine-checked receipts never authorize production release.
