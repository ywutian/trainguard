import json
from pathlib import Path

import pytest

from trainguard import benchmark


def install_runs(monkeypatch, fail_at=None):
    calls = []

    def launch(config, root):
        calls.append(json.loads(config.read_text())["checkpoint"]["mode"])
        path = root / f"run-{len(calls)}"
        path.mkdir(parents=True)
        (path / "config.json").write_text(config.read_text())
        (path / "run.json").write_text(json.dumps({
            "status": "FAILED" if len(calls) == fail_at else "SUCCEEDED",
        }))
        if len(calls) == fail_at:
            return path, False
        (path / "summary.json").write_text(json.dumps({"training_elapsed_seconds": 2.0}))
        return path, True

    monkeypatch.setattr(benchmark, "run", launch)
    monkeypatch.setattr(
        benchmark, "resume",
        lambda path: json.loads((path / "run.json").read_text())["status"] == "SUCCEEDED",
    )
    monkeypatch.setattr(
        benchmark, "validate_runs", lambda a, b: {"passed": True, "differences": []}
    )
    monkeypatch.setattr(
        benchmark,
        "_run_metrics",
        lambda p: {
            "checkpoint_count": 0,
            "checkpoint_bytes": 0,
            "staging_seconds": 0,
            "writing_seconds": 0,
            "checksum_commit_seconds": 0,
            "restart_seconds": 0,
            "recomputed_steps": 0,
        },
    )
    return calls


def test_failed_experiment_can_continue_without_duplicate_rows(tmp_path, monkeypatch):
    calls = install_runs(monkeypatch, fail_at=4)
    config = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    with pytest.raises(RuntimeError):
        benchmark.run_benchmark(config, tmp_path, repetitions=3, warmups=0)
    directory = next(tmp_path.iterdir())
    assert (directory / "results.json").exists(), "completed rows must survive a failed run"
    results = json.loads((directory / "results.json").read_text())
    assert len(results["raw_runs"]) == 3
    assert results["status"] == "FAILED"
    assert any(slot["status"] == "FAILED" and slot["reason"] for slot in results["slots"])
    resume = getattr(benchmark, "resume_benchmark", None)
    assert callable(resume)
    resume(directory)
    final = json.loads((directory / "results.json").read_text())
    assert len(final["raw_runs"]) == 9
    assert len(calls) == 10
    assert final["status"] == "SUCCEEDED"
    assert len({(r["repeat"], r["mode"]) for r in final["raw_runs"]}) == 9
    resume(directory)
    assert len(calls) == 10


def test_resume_rejects_source_changes(tmp_path, monkeypatch):
    install_runs(monkeypatch, fail_at=2)
    config = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    with pytest.raises(RuntimeError):
        benchmark.run_benchmark(config, tmp_path, repetitions=3, warmups=0)
    directory = next(tmp_path.iterdir())
    assert (directory / "results.json").exists()
    result = json.loads((directory / "results.json").read_text())
    result["environment"]["source_sha256"] = "changed"
    (directory / "results.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="source"):
        benchmark.resume_benchmark(directory)


def test_six_orders_are_balanced():
    orders = [benchmark.mode_order(repeat) for repeat in range(1, 7)]
    assert len(set(orders)) == 6
    for position in range(3):
        assert all(
            sum(order[position] == mode for order in orders) == 2
            for mode in ("none", "sync", "async")
        )


def pending_slot(tmp_path, monkeypatch, measurement=None):
    calls = install_runs(monkeypatch)
    config = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    directory = benchmark.run_benchmark(config, tmp_path, repetitions=3, warmups=0)
    results = json.loads((directory / "results.json").read_text())
    slot = results["slots"][-1]
    run_dir = Path(slot["run_dir"])
    (run_dir / "config.json").write_text((directory / f"{slot['mode']}-config.json").read_text())
    (run_dir / "run.json").write_text(json.dumps({"status": "SUCCEEDED", "measurement": measurement}))
    slot.pop("row")
    slot["status"] = "RUNNING"
    (directory / "results.json").write_text(json.dumps(results))
    return directory, calls


def test_completed_run_reuses_original_measurements(tmp_path, monkeypatch):
    measurement = {
        "elapsed_seconds": 23.5,
        "load_average_before": [1, 2, 3],
        "load_average_after": [4, 5, 6],
        "method": "controller_monotonic",
    }
    directory, calls = pending_slot(tmp_path, monkeypatch, measurement)
    monkeypatch.setattr(benchmark, "resume", lambda path: True)
    benchmark.resume_benchmark(directory)
    result = json.loads((directory / "results.json").read_text())
    row = result["slots"][-1]["row"]
    assert row["elapsed_seconds"] == 23.5
    assert row["load_average_before"] == [1, 2, 3]
    assert row["load_average_after"] == [4, 5, 6]
    assert len(calls) == 9


def test_missing_original_measurement_requires_new_formal_run(tmp_path, monkeypatch):
    directory, calls = pending_slot(tmp_path, monkeypatch)
    monkeypatch.setattr(benchmark, "resume", lambda path: True)
    benchmark.resume_benchmark(directory)
    result = json.loads((directory / "results.json").read_text())
    assert len(calls) == 10
    assert "measurement" in result["slots"][-1]["history"][0]["reason"]


def test_repeated_resume_cannot_bypass_live_worker_group(tmp_path, monkeypatch):
    from trainguard.controller import RunActiveError

    directory, calls = pending_slot(tmp_path, monkeypatch)

    def active(path):
        raise RunActiveError("worker group remains active")

    monkeypatch.setattr(benchmark, "resume", active)
    for _ in range(2):
        with pytest.raises(RunActiveError):
            benchmark.resume_benchmark(directory)
    assert len(calls) == 9
    result = json.loads((directory / "results.json").read_text())
    assert result["status"] == "BLOCKED"
    assert result["slots"][-1]["status"] == "RUNNING"


def test_controller_preserves_original_duration_on_completion_audit(tmp_path, monkeypatch):
    from trainguard import controller

    monkeypatch.setattr(controller.time, "monotonic", lambda: 125.0)
    monkeypatch.setattr(controller.os, "getloadavg", lambda: (4.0, 5.0, 6.0))
    status = {
        "status": "RUNNING", "execution_started_monotonic": 100.0,
        "execution_load_before": [1.0, 2.0, 3.0],
    }
    controller._set_status(tmp_path, status, "SUCCEEDED", "completed")
    assert status["measurement"]["elapsed_seconds"] == 25.0
    finished_at = status["finished_at"]
    monkeypatch.setattr(controller.time, "monotonic", lambda: 200.0)
    controller._set_status(tmp_path, status, "SUCCEEDED", "completion audited")
    assert status["measurement"]["elapsed_seconds"] == 25.0
    assert status["finished_at"] == finished_at
