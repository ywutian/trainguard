# Experiment protocol

## Correctness

1. Run a fixed-seed, uninterrupted CPU reference with the same model, data settings, and two-worker world size.
2. Run a fault-injected configuration with a committed checkpoint. Faults can exit a worker, interrupt a save before commit, corrupt a committed DCP file, or stall a worker after a completed step.
3. Validate the recovered run with `trainguard validate --reference <dir> --recovered <dir>`. The validator reconstructs each rank's effective sample sequence by discarding events after every selected rollback step. It compares that sequence and exact final model, optimizer, scheduler, and completed-step digests with the reference. It independently audits both runs: steps must be consecutive within each attempt, sample IDs must be nonnegative integers with the configured batch length, and successful attempts must retain each rank log. Duplicate steps, malformed events, invalid summaries, and missing effective steps fail with diagnostics. Events belonging to another run, attempt, or rank are ignored. Only an incomplete final line from an unsuccessful attempt may be discarded as a crash tail. The comparison uses `atol=0`, `rtol=0`, fixed before the run.
4. For negative controls, set `recovery.omit_state` to `rng`, `optimizer`, or `cursor` in separate recovered configurations. Enable model dropout for the RNG control. Each control must finish training but fail validation.

The automated suite runs these cases and checks fallback to an older version when the newest checkpoint is corrupted. It also covers controller exits after file commit but before index update, after worker completion but before top-level success publication, and after a launcher leaves a live worker. Resume must refuse an overlapping worker group and reconcile finished attempts without spending another retry. `run.sqlite3` records each attempt and recovery; rank event files preserve the observed sample IDs.

## Performance

Run `trainguard benchmark --config configs/cpu_benchmark.yaml --output-root runs --repetitions 5 --warmups 1`, then repeat with `configs/cpu_benchmark_large.yaml`. Both use 3000 steps and a 100-step checkpoint interval. The first uses hidden size 64 with one layer; the second uses hidden size 128 with two layers. Compare three modes within each workload: no checkpoint, synchronous DCP, and native asynchronous DCP. Each mode gets one warm-up followed by five measured runs, with mode order rotating across repetitions. All runs are validated against the first uninterrupted reference for that workload. `results.json` stores `warmup_runs` separately from `raw_runs`; summary statistics exclude warm-ups. `--warmups 0` disables warm-up runs. At least three measured repetitions per mode are required.

Wall time includes process launch and training. Worker training time starts after process-group and model initialization and ends after the final training barrier; it includes checkpoint work but excludes launch. Per-checkpoint events record staging, completion lag, and checksum/commit time. Completion lag (`writing_seconds`) measures elapsed time after staging until the trainer observes completion at the rank barrier. It includes rank-state writes and, for async, overlapping training; it is not pure I/O duration. The benchmark also reports restart time and recomputed steps, which are zero for runs without injected failures. Phase times are not additive with wall time. Report median and full observed range alongside every raw measurement. Keep CPU/local-filesystem, GPU, and simulated storage-delay results separate.

Committed payload bytes sum all file sizes listed in checkpoint manifests over a run, excluding manifests and commit markers. Divide by checkpoint count for mean payload size per checkpoint. The benchmark records host load averages over 1, 5, and 15 minutes immediately before and after each run. Use an isolated, stable host for performance conclusions; these snapshots cannot establish isolation or control cache and thermal effects. Warm-up runs prepare the host and filesystem cache, but every measured run still launches fresh workers and includes initial training steps. If observed ranges overlap or noise approaches the mode difference, report measurements without claiming a stable speed advantage.

The [two-size CPU report](experiments/cpu-sizes-2026-09-25.md) contains 18 measured runs and 6 separately recorded warm-ups, all validated. Its sync/async timing ranges overlap in each workload. The earlier [extended CPU report](experiments/cpu-extended-2026-09-25.md) contains 15 measured and validated runs without the new warm-up protocol; the [four-step report](experiments/cpu-2026-09-25.md) is a smoke test. These reports do not establish a general async speed advantage, GPU or multi-node performance, or host-power-loss durability.

## Version 0.2 protocol

Use `acceptance` for reference/fault/negative-control closure and `acceptance-resume` for continuation. Ten cases cover sync/async exit and save interruption, sync/async corruption, a sync hang, and three omitted-state controls. The campaign forces dropout >=0.2 and save interval 1 so the controls have a defined checkpoint/RNG boundary; copied configurations record these changes. Separate controller exit windows and data/prefetch/accumulation/scaler boundary cases are in the automated suite.

`benchmark-resume` requires identical source digest, Python/dependency versions and copied configurations. Successful slots are revalidated; failures keep reason/history and are retried once on continuation. Warm-ups remain separate. Six repetitions cover every mode permutation; use initial target 12 repetitions per mode on a dedicated host for performance inference, then replicate an independent batch. Raw rows include rank/resource peaks, paired within-repeat differences and actual run paths. A paused or failed experiment never publishes a completed report.

New phase definitions:

- Preparation: constructing model/optimizer state dictionaries.
- Staging: synchronous submission/staging completion before model mutation can resume.
- Upload: staging-return/submission to Future callback (or synchronous DCP duration), including DCP planning/coordination and I/O. Callback execution is a timestamp observation, not an internal storage profiler.
- Main-thread wait: coordinated forced wait and readiness barrier; excludes the separately reported preparation/staging phase.
- Checksum/commit: rank-0 full file validation, hashing and publication.
- Eligibility lag: largest local upload-completion-to-publication-notification duration across ranks, including control notification; an upper bound on time to COMMITTED publication.
- Recoverable step lag: current update at commit minus saved update.
- RTO: local controller fault observation to all ranks' first resumed update completion; CUDA synchronizes this one measurement boundary. A restore directly to the final update has no first new update and RTO is undefined, not zero.
- `attempt_gap_seconds`: old attempt completion to new attempt allocation; `restart_seconds` is retained as its legacy alias. It is not RTO.

Phase maxima can come from different ranks and overlap training; they must not be summed into wall time. All timestamps subtracted for RTO belong to the same host in this implementation. Do not apply the method to unsynchronized multi-host clocks.

`storage-benchmark` uses one Gloo rank and 64/256 MiB tensors. It checks exact DCP round trips and reports staging/upload/hash/load separately, with immediate raw-row persistence. Its throughput is not full training speed. Host/cache state is not isolated.

`checkpoint-budget` accepts measured equivalent overhead, upload, commit lag, RTO, an explicit job MTBF assumption and a rollback budget. It uses the restricted sparse-failure approximation documented in the [assessment](analysis/upgrade-assessment-2026-09-25.md), constrains one-upload scheduling and reports infeasibility. It does not infer a failure rate from injected faults or automatically tune training.
