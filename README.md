# TrainGuard

TrainGuard is a project for testing whether distributed PyTorch training resumes **correctly** after process and checkpoint failures. A run that continues is only correct when model, optimizer, scheduler, random state, and the next data position describe the same completed update.

## Current status

The repository currently contains the first runnable baseline: a fixed-size CPU DDP training attempt using two Gloo workers, deterministic synthetic token data, validated YAML configuration, per-rank JSONL events, and a bounded launcher. **Checkpoint saving, recovery, fault injection, and correctness comparison are not implemented yet.** See [the roadmap](docs/roadmap.md) for acceptance criteria.

| Capability | Status |
| --- | --- |
| CPU DDP reference run | Implemented |
| Deterministic sample IDs and input tokens | Implemented |
| Config fingerprint and per-rank event logs | Implemented |
| Complete-state checkpoint and manifest | Planned |
| Automatic group restart and explicit resume | Planned |
| Fault matrix and recovery validator | Planned |
| Sync versus async DCP benchmark | Planned |

## Quick start

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run trainguard validate-config --config configs/cpu_demo.yaml
uv run trainguard run --config configs/cpu_demo.yaml
uv run pytest
```

Each run creates `runs/<run-id>/` with `run.json`, `launcher.log`, `summary.json` on success, and one JSONL event file per rank under `attempts/attempt-001/`. Generated run data is excluded from version control. The demo uses four steps to keep the first smoke test short; the config can be copied and adjusted for larger experiments.

## Design boundaries

The target system is single-node and fixed-size for its first recovery implementation. The controller will own restart decisions while `torchrun` owns worker creation and rendezvous, with `--max-restarts=0`. Only a validated, application-level committed checkpoint will be eligible for recovery. Each checkpoint must contain model, optimizer, scheduler, global step, per-rank RNG, and per-rank data cursor state from the same update boundary.

The current launcher runs a single attempt and does **not** claim recovery correctness. The design and planned failure cases are documented in [architecture](docs/architecture.md), [recovery semantics](docs/recovery-semantics.md), and [experiment protocol](docs/experiment-protocol.md).

## Development

```bash
uv run ruff check .
uv run pytest
```

The baseline is designed to run locally without a GPU or data download. GPU, multi-node, FSDP, and storage durability across host power loss are outside the first implementation scope.

## References

- [PyTorch torchrun documentation](https://docs.pytorch.org/docs/2.10/elastic/run.html)
- [PyTorch distributed checkpoint documentation](https://docs.pytorch.org/docs/2.10/distributed.checkpoint.html)

## License

MIT. See [LICENSE](LICENSE).
