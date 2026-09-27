# CPU checkpoint benchmark

Recorded: 2026-09-25T21:44:02.711387+00:00
Workload fingerprint: `43b8e203e17c69a06949b12d09d946b96cd69222431a259bc01ba8dfea10d14d`
Environment: Python 3.12.12; PyTorch 2.10.0; macOS-26.5.1-arm64-arm-64bit; 10 logical CPUs.
World size: 2; worker threads: 1.

## Elapsed time

| Mode | Median (s) | Range (s) | Runs |
| --- | ---: | ---: | ---: |
| none | 8.478 | 7.223–11.103 | 5 |
| sync | 9.382 | 8.114–14.070 | 5 |
| async | 8.535 | 8.319–12.707 | 5 |

## Worker training window

| Mode | Median (s) | Range (s) | Runs |
| --- | ---: | ---: | ---: |
| none | 5.792 | 4.824–6.962 | 5 |
| sync | 6.714 | 5.825–11.210 | 5 |
| async | 6.170 | 5.668–9.479 | 5 |

## Raw measurements

| Mode | Repeat | Elapsed (s) | Training (s) | Checkpoints | Staging (s) | Writing elapsed (s) | Hash + commit (s) | Restart (s) | Recomputed steps | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| none | 1 | 7.223 | 4.824 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | yes |
| sync | 1 | 8.114 | 5.825 | 30 | 0.021 | 0.436 | 0.173 | 0.000 | 0 | yes |
| async | 1 | 8.535 | 6.170 | 30 | 0.100 | 5.689 | 0.192 | 0.000 | 0 | yes |
| sync | 2 | 14.070 | 11.210 | 30 | 0.040 | 0.924 | 0.314 | 0.000 | 0 | yes |
| async | 2 | 8.319 | 5.781 | 30 | 0.076 | 5.317 | 0.178 | 0.000 | 0 | yes |
| none | 2 | 8.478 | 5.945 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | yes |
| async | 3 | 12.707 | 9.479 | 30 | 0.126 | 8.738 | 0.282 | 0.000 | 0 | yes |
| none | 3 | 11.103 | 6.962 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | yes |
| sync | 3 | 12.272 | 8.833 | 30 | 0.030 | 0.640 | 0.253 | 0.000 | 0 | yes |
| none | 4 | 10.304 | 5.573 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | yes |
| sync | 4 | 9.382 | 6.714 | 30 | 0.023 | 0.546 | 0.204 | 0.000 | 0 | yes |
| async | 4 | 9.156 | 6.191 | 30 | 0.081 | 5.594 | 0.191 | 0.000 | 0 | yes |
| sync | 5 | 8.534 | 6.151 | 30 | 0.022 | 0.464 | 0.191 | 0.000 | 0 | yes |
| async | 5 | 8.321 | 5.668 | 30 | 0.073 | 5.220 | 0.177 | 0.000 | 0 | yes |
| none | 5 | 8.280 | 5.792 | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | yes |

Each raw row points to its run directory in `results.json`. Validation uses exact hashes and effective sample IDs (atol=0, rtol=0). Mode order rotates across repetitions. The exact base configuration is in `results.json`, and mode configurations are copied beside this report.

## Measurement limits

Native asynchronous save can overlap writing with training. Its writing elapsed measurement includes that overlap, so phase totals must not be added to wall time. Worker training time starts after process-group and model initialization and ends after the final training barrier; it includes checkpoint work but excludes launch. These runs use a local CPU filesystem; they do not measure GPU, multi-node, storage-delay, disk-loss, or host-power-failure behavior. Short workloads and few repetitions cannot establish a general performance advantage.

## Interpretation

All 15 runs passed exact final-state and effective-sample validation. Each checkpointed run committed 30 checkpoints. The worker training medians were 5.792 s without checkpoints, 6.714 s with synchronous saving, and 6.170 s with asynchronous saving. The observed ranges overlap substantially, and the host showed sizeable run-to-run variation. These measurements do not establish a consistent speed advantage for either checkpoint mode.

The workload uses a small model and approximately 1.4 MB per checkpoint. The five repetitions ran on a local machine without isolation from other activity. The earlier four-step result remains useful as a smoke test but is dominated by process launch.

## Reproduce

From the project root:

```bash
uv sync
uv run trainguard benchmark --config configs/cpu_benchmark.yaml --output-root runs --repetitions 5
```

The [companion JSON](cpu-extended-2026-09-25.json) preserves the exact configuration, environment, summary statistics, and every raw measurement. Run IDs map to the retained local directory `runs/benchmark-8bf4ca3277e5/runs/`.
