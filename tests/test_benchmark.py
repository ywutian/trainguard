import json
from pathlib import Path

import pytest

from trainguard import benchmark
from trainguard.benchmark import mode_order, run_benchmark, summarize_rows
from trainguard.run_store import RunStore


def test_benchmark_summary_keeps_raw_measurements_and_median_range() -> None:
    rows = [
        {"mode": "sync", "elapsed_seconds": 3.0},
        {"mode": "sync", "elapsed_seconds": 1.0},
        {"mode": "sync", "elapsed_seconds": 2.0},
        {"mode": "async", "elapsed_seconds": 4.0},
        {"mode": "async", "elapsed_seconds": 5.0},
        {"mode": "async", "elapsed_seconds": 6.0},
    ]
    summary = summarize_rows(rows)
    assert summary["sync"] == {"median_seconds": 2.0, "min_seconds": 1.0, "max_seconds": 3.0}
    assert summary["async"] == {"median_seconds": 5.0, "min_seconds": 4.0, "max_seconds": 6.0}


def test_repetition_order_rotates_modes() -> None:
    assert mode_order(1) == ("none", "sync", "async")
    assert mode_order(2) == ("sync", "async", "none")
    assert mode_order(3) == ("async", "none", "sync")


def test_summary_can_measure_training_window_separately_from_launch() -> None:
    rows = [
        {"mode": "sync", "elapsed_seconds": 10.0, "training_seconds": 2.0},
        {"mode": "sync", "elapsed_seconds": 11.0, "training_seconds": 4.0},
        {"mode": "sync", "elapsed_seconds": 12.0, "training_seconds": 3.0},
    ]
    assert summarize_rows(rows, "training_seconds") == {
        "sync": {"median_seconds": 3.0, "min_seconds": 2.0, "max_seconds": 4.0}
    }


def test_warmup_runs_are_validated_but_excluded_from_statistics(tmp_path: Path, monkeypatch) -> None:
    launched = []
    validated = []

    def fake_run(config, output_root):
        mode = json.loads(config.read_text())["checkpoint"]["mode"]
        launched.append(mode)
        directory = output_root / f"run-{len(launched)}"
        directory.mkdir(parents=True)
        duration = 100.0 if len(launched) <= 3 else 2.0
        (directory / "summary.json").write_text(json.dumps({"training_elapsed_seconds": duration}))
        return directory, True

    def fake_validate(reference, recovered):
        validated.append(recovered)
        return {"passed": True, "differences": []}

    monkeypatch.setattr(benchmark, "run", fake_run)
    monkeypatch.setattr(benchmark, "validate_runs", fake_validate)
    monkeypatch.setattr(benchmark, "_run_metrics", lambda directory: {
        "checkpoint_count": 1, "checkpoint_bytes": 1024,
        "staging_seconds": 0.1, "writing_seconds": 0.2,
        "checksum_commit_seconds": 0.1, "restart_seconds": 0.0, "recomputed_steps": 0,
    })
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    directory = run_benchmark(source, tmp_path, repetitions=3, warmups=1)
    result = json.loads((directory / "results.json").read_text())
    assert len(result["warmup_runs"]) == 3
    assert len(result["raw_runs"]) == 9
    assert len(validated) == 12
    assert all(row["training_seconds"] == 100.0 for row in result["warmup_runs"])
    assert all(value["median_seconds"] == 2.0 for value in result["training_summary"].values())
    assert all(row["checkpoint_bytes"] == 1024 for row in result["raw_runs"])
    assert "Warm-up repetitions per mode: 1" in (directory / "report.md").read_text()
    assert all("load_average_before" in row for row in result["raw_runs"])


def test_negative_warmup_count_is_rejected(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    with pytest.raises(ValueError, match="warmup"):
        run_benchmark(source, tmp_path, warmups=-1)


def test_checkpoint_bytes_count_committed_payloads(tmp_path: Path) -> None:
    log = tmp_path / "attempts" / "attempt-001" / "rank-0.jsonl"
    log.parent.mkdir(parents=True)
    events = []
    for step in (1, 2):
        checkpoint = tmp_path / "checkpoints" / f"step-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "manifest.json").write_text(json.dumps({
            "files": [{"path": "dcp/payload", "size": 100}, {"path": "rank-0", "size": 25}],
        }))
        events.append({
            "event_type": "checkpoint_committed", "checkpoint_path": str(checkpoint),
            "staging_seconds": 0.1, "writing_seconds": 0.2, "checksum_commit_seconds": 0.3,
        })
    log.write_text("".join(json.dumps(event) + "\n" for event in events))
    RunStore(tmp_path / "run.sqlite3").close()
    metrics = benchmark._run_metrics(tmp_path)
    assert metrics["checkpoint_count"] == 2
    assert metrics["checkpoint_bytes"] == 250
