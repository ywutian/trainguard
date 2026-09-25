# Experiment protocol

## Correctness

1. Run a fixed-seed, uninterrupted CPU reference with the same model, data settings, and two-worker world size.
2. Run a fault-injected configuration with a committed checkpoint. Faults can exit a worker, interrupt a save before commit, corrupt a committed DCP file, or stall a worker after a completed step.
3. Validate the recovered run with `trainguard validate --reference <dir> --recovered <dir>`. The validator reconstructs each rank's effective sample sequence by discarding events after every selected rollback step. It compares that sequence and exact final model, optimizer, scheduler, and completed-step digests with the reference. The comparison uses `atol=0`, `rtol=0`, fixed before the run.
4. For negative controls, set `recovery.omit_state` to `rng`, `optimizer`, or `cursor` in separate recovered configurations. Enable model dropout for the RNG control. Each control must finish training but fail validation.

The automated suite runs these cases and checks fallback to an older version when the newest checkpoint is corrupted. `run.sqlite3` records each attempt and recovery; rank event files preserve the observed sample IDs.

## Performance

Run `trainguard benchmark --config configs/cpu_benchmark.yaml --output-root runs --repetitions 5`. The command uses one 3000-step workload and a 100-step checkpoint interval for three modes: no checkpoint, synchronous DCP, and native asynchronous DCP. It runs each mode five times in rotating order, validates every completed run against the first uninterrupted reference, and writes raw `results.json` plus `report.md`.

Wall time includes process launch and training. Worker training time starts after process-group and model initialization and ends after the final training barrier; it includes checkpoint work but excludes launch. Per-checkpoint events record staging, elapsed save completion, and checksum/commit time. The benchmark also reports restart time and recomputed steps, which are zero for runs without injected failures. Async writing can overlap training, so its phase times are not additive with wall time. Report median and full observed range alongside every raw measurement. Keep CPU/local-filesystem, GPU, and simulated storage-delay results separate.

The [extended CPU report](experiments/cpu-extended-2026-09-25.md) contains 15 measured and validated runs. The earlier [four-step report](experiments/cpu-2026-09-25.md) is a smoke test. Observed ranges in the extended run overlap, so it does not establish a general async speed advantage. Neither report establishes GPU or multi-node performance or host-power-loss durability.
