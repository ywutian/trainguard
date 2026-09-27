# TrainGuard

TrainGuard tests whether fixed-size distributed PyTorch training resumes from a consistent update boundary. A successful recovery must restore the model, optimizer, scheduler, rank-local random state, completed update count, and next data position. The validator compares the final state and effective sample sequence with an uninterrupted run.

## Status

The CPU recovery path is implemented and tested with two Gloo workers. It supports synchronous and native asynchronous Distributed Checkpoint (DCP), application-level checkpoint commits, bounded full-group restarts, explicit resume, deterministic fault injection, correctness validation, and repeated local benchmarks. CUDA DDP and FSDP2 adapters are implemented but require real-device acceptance. Multi-node/object-storage recovery and host-power-loss durability remain open gates.

Version 0.3.3 added a single-file external CPU DDP workload interface for local evaluation. A nonbuilt-in regression example runs through the same reference, crash/recovery, and omitted-state comparisons. The interface still uses the fixed AdamW optimizer and Cosine scheduler; its source hash covers only the adapter file. Version 0.3.4 restricts the source distribution to reviewed repository files and checks every archive member against the checkout.

Version 0.3.6 is a candidate with a bounded v2 external workload contract for local two-rank CPU/Gloo evaluation. It tests an external one-group SGD optimizer with momentum, external StepLR, complete stream state, and extra loss state. A reference run, worker-exit recovery, and omitted-state controls compare exact final state. This remains a local experiment and does not change customer or production gates.

For the proposed customer deployment, integration, operations, security and commercial acceptance scope, see the [product closure and release gates](docs/plans/product-closure-2026-09-26.md). The current release is an experiment package and has not passed those production gates.

The [customer pilot template](docs/commercial/customer-pilot-template.md), [operations runbook](docs/commercial/operations-runbook.md), and [machine-readable release gates](docs/commercial/release-gates.json) record the commercial scope and evidence required before customer production use. The existing code is MIT licensed; pilot fees cover agreed integration, validation and support services.
[Security reporting](SECURITY.md) uses the repository private vulnerability form. Candidate supply-chain evidence includes a CycloneDX SBOM, declared third-party license inventory, and point-in-time known-vulnerability scan. These are candidate evaluation records and do not establish production approval.

The [market evidence](docs/commercial/market-evidence-2026-09-26.md) separates current competitor facts from unvalidated positioning and pricing hypotheses. The source distribution includes the commercial handoff documents but omits historical raw experiment evidence; the repository retains the full record.

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run trainguard validate-config --config configs/cpu_demo.yaml
uv run trainguard run --config configs/cpu_demo.yaml
uv run trainguard run --config configs/recovery_demo.yaml --allow-experiment
```

The first run is an uninterrupted reference. The second injects a rank-0 exit after step 2, restarts the group from a committed checkpoint, and completes. Each command prints its run directory. Compare the two directories:

An installed wheel can write its packaged CPU example outside the source checkout with `trainguard init-config --output cpu_demo.yaml`. A configuration with `run.profile: guarded` requires checkpointing, explicit capacity budgets, two checkpoint opportunities, and a customer-held sample commitment key. It rejects fault injection or omitted recovery state. This guard does not imply production acceptance.

```bash
uv run trainguard validate --reference runs/<reference-id> --recovered runs/<recovered-id>
```

Validation writes `validation.json` in the recovered run directory. It checks exact SHA-256 digests for model, optimizer, and scheduler state, the completed step, and each rank's effective sample sequence after removing rolled-back work. It also rejects duplicate or out-of-order steps, invalid sample IDs, missing successful-rank logs, and invalid final summaries in either run. A failed attempt may end with a truncated final log line. The fixed CPU comparison tolerance is `atol=0, rtol=0`. A mismatch exits with status 1.

If a controller exits after a committed checkpoint, resume its run directory with:

```bash
uv run trainguard resume runs/<run-id>
```

Resume refuses to start while the previous owned launcher or worker group is still running, including the interval before a launcher PID is recorded. A run with no valid committed checkpoint fails clearly. Configuration is copied into the run directory so later changes to the original YAML do not alter recovery.

`recovery.startup_timeout_seconds` bounds launch, state loading, and the first completed update. After that first update, `recovery.progress_timeout_seconds` bounds stalls between completed updates; `run.timeout_seconds` bounds the whole attempt. A startup timeout fails the attempt and does not count as a successful recovery.

### External CPU DDP workload

For a trusted external CPU DDP workload, copy the CPU example configuration and add `external_workload` with `version: 1`, the Python file path, and its SHA-256. The file must define `WORKLOAD_API_VERSION = 1` and callable `build_model(config)`, `build_stream(config, rank, consumed_batches)`, and `loss(output, batch, config)`. The stream returns sample IDs and a Tensor batch from `next()` and provides `close()`. See [the independent example](examples/external_cpu_workload.py). The source file is checked before launch, copied into the run directory, and checked again by each worker and on resume. Preflight executes this trusted Python code. Imports, data files, and external side effects are outside the single-file digest and need their own inventory and acceptance. This interface does not establish customer-workload or production support.

For v2, set `external_workload.version: 2` and declare at least one direct local Python dependency (`module`, `path`, `sha256`) and one data file (`name`, `path`, `sha256`). Each file must be regular and at most 1 MiB. The [v2 example](examples/external_cpu_workload_v2.py) uses [a helper](examples/external_v2_helper.py) and [a data file](examples/external_v2_data.json). The run binds and freezes their exact bytes, checks both original and frozen files on resume and during exact comparison, and rejects undeclared direct nonstandard imports. V2 uses exactly two CPU/Gloo ranks, FP32, and zero data workers. All inputs and the adapter are trusted code/data; this is not a Python sandbox or a complete inventory of dynamic imports, network reads, environment access, or other side effects.

The v2 file defines `WORKLOAD_API_VERSION = 2`, `build_model(config)`, `build_optimizer(wrapped_model, config)`, `build_scheduler(optimizer, config)`, `build_stream(config, rank, consumed_batches, data_paths)`, `build_extra_state(config, rank)`, and `loss(output, batch, config, extra)`. The optimizer must be one-group `torch.optim.SGD` with momentum; the scheduler must be `torch.optim.lr_scheduler.StepLR` bound to it. The stream supplies `next()`, `close()`, `state_dict()`, and `load_state_dict()`; its JSON state includes `consumed_batches`. The extra object supplies `state_dict()` and `load_state_dict()`. Both states must be finite JSON and together at most 64 KiB per rank. They are saved after a completed optimizer update and loaded before the next batch is consumed. Checkpoint RNG is restored after the stream and extra constructors and load hooks, so those hooks cannot advance the resumed process RNG. The v2 restore path allocates SGD momentum buffers without calling `optimizer.step()`; the worker rejects any step during restore. This path has only been proven for the tested SGD/StepLR shape and does not support arbitrary optimizers, schedulers, data loaders, distributed topologies, or live customer workloads. Deliberate `recovery.omit_state: stream` and `extra` are experiment-only negative controls.

## Checkpoints and records

Each candidate is stored under `runs/<run-id>/checkpoints/step-<step>-<attempt>/`. DCP writes model and optimizer state. Every rank writes its scheduler, Python/NumPy/Torch CPU and optional CUDA RNG, scaler, update and consumed-batch counters, and compatibility fingerprints. Rank 0 validates all expected files, records sizes and SHA-256 hashes in `manifest.json`, then publishes `COMMITTED`. Recovery scans these files and ignores incomplete, incompatible, or corrupted candidates, falling back to the newest valid older checkpoint.

The run directory also contains `run.json`, a SQLite index (`run.sqlite3`), `launcher.log`, per-attempt rank event logs and summaries, and a final `summary.json`. In the default local mode, the committed manifest is the checkpoint validity source if the controller stops before updating SQLite. In the same-host reference experiment, only HEAD-referenced generations are eligible for recovery.

### Same-host reference checkpoint experiment

The optional reference mode connects real two-rank CPU/Gloo synchronous DCP saves to a separate same-host SQLite store:

```bash
mkdir -p runs/reference-private
chmod 700 runs/reference-private
uv run trainguard run --config configs/recovery_demo.yaml --allow-experiment \
  --reference-store runs/reference-private/objects.sqlite3
```

This mode accepts only the experiment profile, two CPU/Gloo DDP ranks, synchronous checkpoints, and no local retention setting. The database must remain outside the run directory, inside a private directory owned by the current user. The controller uploads each fully committed checkpoint, then confirms a conditional HEAD publication before recording it as published. A later `trainguard resume <run-directory>` reads the saved store path and selects only HEAD-referenced generations. Removing `<run-directory>/checkpoints/` after a stopped worker group still permits recovery from HEAD; the rest of the run directory and the separate database must remain. The protocol retains at most eight HEAD candidates. A self-consistent candidate that fails the actual distributed load is marked by durable restore evidence and the next candidate is tried. If HEAD has no usable candidate, recovery fails closed even when local COMMITTED directories exist.

SQLite WAL, the run-directory process lock, and the persisted local epoch identity are same-host experiment mechanisms. This mode does not establish a production object service, cross-host fencing, independent isolation after host failure, or power-loss durability. The reference store holds complete checkpoint bytes without an automatic capacity budget; use bounded test workloads and preserve the database for review.

Experiment run evidence can contain raw sample identifiers and paths. Guarded rank events replace raw sample IDs with ordered, customer-keyed HMAC commitments and record only a key identifier; the customer keeps the key in a private file outside the run directory. The key holder can re-sign evidence, and custom workload output or unstructured launcher logs are outside this protection. Keep all run data in a restricted customer-controlled directory. `trainguard support-bundle runs/<run-id> --output support.json` produces a read-only summary with allowlisted scalar fields; the customer should review and approve it before sharing.

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
uv build --wheel --sdist --build-constraints build-constraints.txt --require-hashes --out-dir dist
uv run python scripts/verify_wheel.py dist/trainguard-0.3.6-py3-none-any.whl
uv run python scripts/verify_install.py dist/trainguard-0.3.6-py3-none-any.whl
```

Current runs also record every installed distribution's normalized name, version, and digests of its `RECORD` and optional `direct_url.json` metadata. At run and resume boundaries, every `RECORD`-listed file other than `RECORD` itself must be a regular file under the active environment with the listed size and SHA-256/384/512. Resume, acceptance, benchmark, and exact run comparison reject a changed or missing inventory. A separate digest binds Python's ordered import paths, startup `.pth` files and hooks, hash seed, Python/CUDA path controls, and all regular files in explicit `PYTHONPATH` roots without recording raw paths. A `PYTHONPATH` entry that shadows the application package, contains a link or special file, or exceeds the import identity limits is rejected; workers do not write bytecode. Runs reject installed packages without `RECORD`, unhashed listed files, and third-party editable installs whose source is not frozen; the project's own editable install uses its package source fingerprint. Origin paths are not written to the run record. These boundary checks do not cover unlisted bytecode or modules added to other unchanged import paths, dynamic imports outside the recorded roots, or files changed after the boundary; deployment still requires an immutable, trusted runtime.

See the [full closure assessment](docs/analysis/closure-assessment-2026-09-26.md) for the supported boundary, infrastructure gates and next acceptance steps.
The [0.2.2 local acceptance report](docs/experiments/closure-2026-09-26.md) contains the test, campaign, build and package evidence.

## Local simulation closure

Run the complete local gate from the repository root:

```bash
uv sync --locked --group dev
uv run python scripts/run_simulation_closure.py --output-root runs/simulation-closure --previous-ref ad53d3e854419caf0bab6c3bff80ae2da6890ce9
```

The gate retains static checks, the full test suite, JUnit and raw test recovery directories, a fresh ten-case CPU acceptance campaign, a wheel and source distribution, and a wheel/source identity check in a uniquely named result directory. Its `result.json` and `report.md` record the outcome even if a gate fails. The CPU campaign requires the intended fault, exactly one recovery, the expected attempt statuses, and either exact agreement with the uninterrupted reference or the specified negative-control difference.

The suite also exercises two independent local launchers with a real two-rank Gloo group, hard process exits across checkpoint publication boundaries, rank-coordinated async cancellation and deadlines, event-log durability ordering, and candidate fallback when DCP metadata or payload cannot be loaded. An isolated in-memory protocol model exercises immutable remote generations, conditional head publication, response-loss reconciliation, and epoch takeover after an independent isolation assertion. Bridge tests publish actual DCP checkpoint bytes through the model, damage the newer version, and restore the older one; a two-rank training run resumes from the downloaded version after all local candidates are removed. The model is not a production remote backend or a real object service.

See the [simulation analysis](docs/analysis/simulation-closure-2026-09-26.md) and [local result](docs/experiments/simulation-closure-2026-09-26.md). Actual CUDA devices, multiple hosts, a selected object service and isolation authority, and controlled host power loss still require their own acceptance runs.

## Version 0.3.1 commercial evaluation boundary

The package now includes an installed CPU template, a guarded configuration profile, explicit authorization for fault and omitted-state experiments at the controller entry, and a read-only allowlisted support export. The local gate installs the built wheel with locked dependencies in a new environment, verifies a real checkpoint recovery against a reference, checks the support export, and confirms the complete run directory survives uninstall. It also rehearses a source/runtime upgrade boundary using separate old and new locked environments: a new wheel rejects an interrupted old run, while the old environment resumes it to the reference result.

These checks support a limited, isolated evaluation package. Run the [release readiness checker](docs/commercial/operations-runbook.md) against the candidate wheel, source distribution, lock and reviewed gate receipts. A `BLOCKED` result prevents a production claim. The current real GPU, multi-host, customer-workload, object-service, security operations and paid-customer gates remain open.
The [version 0.3.1 local validation record](docs/commercial/evidence/local-validation-0.3.1.json) links its local outcomes to its artifacts. The [Linux verification failure](docs/commercial/evidence/linux-check-0.3.1.json) is retained separately; those artifacts and receipts are not used for the current candidate. An evaluation bundle remains labeled `EVALUATION_ONLY`; machine-checked receipts never authorize production release.

Version 0.3.2 closes the Linux process-start ownership race and makes the CLI option check independent of terminal width. Its [local validation record](docs/commercial/evidence/local-validation-0.3.2.json) identifies separate artifacts and evidence; the version 0.3.1 record remains historical.

Version 0.3.3 verifies a selected checkpoint again on every worker before and after loading, rejects inconsistent rank scheduler and final DDP state, and strengthens local test and delivery-bundle integrity checks. Its [local validation record](docs/commercial/evidence/local-validation-0.3.3.json) binds the local artifacts and raw results. [Linux artifact inspection](docs/commercial/evidence/linux-check-0.3.3.json) later found generated test output in its source distribution, so those artifacts are superseded. Version 0.3.4 checks an explicit source archive file set and rejects generated members before delivery. Recipients must authenticate the delivery manifest and bundled verifier with a manifest digest received through an independent trusted channel before executing bundle code; the [operations runbook](docs/commercial/operations-runbook.md) gives the exact sequence. Bundle consistency alone does not authenticate the sender. The external workload example and same-host storage reference do not change the nine blocked customer and production gates.

Version 0.3.5 strengthens checkpoint load checks, records per-rank restore progress so a stopped group cannot repeatedly select a checkpoint whose load did not finish, and keeps ambiguous candidates for diagnosis. Newly created run records are private, atomic report writes resist planted temporary links, and local release evidence is bound to collected test identities and a fixed prior release. Each local gate has a timeout, records timeout as failure, and stops its discovered child process groups. A separate offline check compares the downloaded Linux 3.11 and 3.12 test evidence and package bytes with the local candidate. These controls require a new full validation record; earlier version results cannot be reused for this candidate.

Version 0.3.6 candidate adds full end-of-run checkpoint auditing, detached-worker fencing, external v2 state restoration, guarded local capacity and keyed sample evidence. A guarded run only reports success if its final-step candidate and at least two structurally verified candidates remain, with the checkpoint tree's regular-file logical bytes under its configured local byte budget and free-space floor. It also binds installed package metadata and checks recorded installed file bytes at run and resume boundaries. Structural verification does not prove every candidate can be loaded on a future device. The release checker binds local gates to execution inputs, then requires a live authenticated Linux 3.11/3.12 workflow fetch before constructing an `EVALUATION_ONLY` customer bundle. Production and customer-environment gates remain blocked until separately tested and signed off.
