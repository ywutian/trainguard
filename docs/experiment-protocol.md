# Experiment protocol

## Correctness

1. Run a fixed-seed, uninterrupted CPU reference with the same model, data settings, and two-worker world size.
2. Run a fault-injected configuration with a committed checkpoint. Faults can exit a worker, interrupt a save before commit, corrupt a committed DCP file, or stall a worker after a completed step.
3. Validate the recovered run with `trainguard validate --reference <dir> --recovered <dir>`. The validator reconstructs each rank's effective sample sequence by discarding events after every selected rollback step. It compares that sequence and exact final model, optimizer, scheduler, and completed-step digests with the reference. The comparison uses `atol=0`, `rtol=0`, fixed before the run.
4. For negative controls, set `recovery.omit_state` to `rng`, `optimizer`, or `cursor` in separate recovered configurations. Enable model dropout for the RNG control. Each control must finish training but fail validation.

The automated suite runs these cases and checks fallback to an older version when the newest checkpoint is corrupted. `run.sqlite3` records each attempt and recovery; rank event files preserve the observed sample IDs.

## Performance

Run `trainguard benchmark --config configs/cpu_demo.yaml --output-root runs --repetitions 3`. The command uses one workload and checkpoint interval for three modes: no checkpoint, synchronous DCP, and native asynchronous DCP. It runs each mode at least three times, validates every completed run against the first uninterrupted reference, and writes raw `results.json` plus `report.md`.

Wall time includes process launch and training. Per-checkpoint events record staging, elapsed writing, and checksum/commit time. The benchmark also reports restart time and recomputed steps, which are zero for runs without injected failures. Async writing can overlap training, so its phase times are not additive with wall time. Report median and full observed range alongside every raw measurement. Keep CPU/local-filesystem, GPU, and simulated storage-delay results separate.

The [2026-09-25 CPU report](experiments/cpu-2026-09-25.md) contains the first measured run and its limits. Results from this short local CPU workload do not establish GPU or multi-node performance, and no host-power-loss durability claim is made.
