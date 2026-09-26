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
        if len(calls) == fail_at:
            return path, False
        (path / "summary.json").write_text(json.dumps({"training_elapsed_seconds": 2.0}))
        return path, True

    monkeypatch.setattr(benchmark, "run", launch)
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
