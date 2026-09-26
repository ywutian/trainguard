import json
from pathlib import Path

import pytest

from trainguard.config import load_config
from trainguard.data import sample_ids_for_step
from trainguard.run_store import RunStore
from trainguard.validation import validate_runs


def _run(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir()
    config = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    (directory / "config.json").write_text(json.dumps(config.model_dump()))
    (directory / "run.json").write_text(json.dumps({"run_id": name, "status": "SUCCEEDED"}))
    (directory / "summary.json").write_text(json.dumps({
        "run_id": name, "attempt_id": "attempt-001",
        "workload_fingerprint": config.workload_fingerprint(),
        "world_size": config.run.world_size, "global_step": config.training.total_steps,
        "model_sha256": "a" * 64, "optimizer_sha256": "b" * 64,
        "scheduler_sha256": "c" * 64,
    }))
    store = RunStore(directory / "run.sqlite3")
    try:
        store.create_run(name, config.fingerprint(), "2026-09-25T00:00:00+00:00")
        store.start_attempt(name, "attempt-001", 1, None, 0)
        store.finish_attempt("attempt-001", "SUCCEEDED", 0, "completed")
    finally:
        store.close()
    for rank in range(config.run.world_size):
        events = [{
            "run_id": name, "attempt_id": "attempt-001", "rank": rank,
            "event_type": "step_completed", "global_step": step,
            "sample_ids": sample_ids_for_step(
                step - 1, rank, config.run.world_size, config.training.batch_size_per_rank
            ),
        } for step in range(1, config.training.total_steps + 1)]
        _write_events(directory, rank, events)
    return directory


def _events(directory: Path, rank: int = 0) -> list[dict]:
    path = directory / "attempts" / "attempt-001" / f"rank-{rank}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_events(directory: Path, rank: int, events: list[dict]) -> None:
    path = directory / "attempts" / "attempt-001" / f"rank-{rank}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


@pytest.mark.parametrize("fault,reason", [
    ("duplicate", "duplicate step"),
    ("order", "step order"),
    ("sample_type", "sample IDs"),
    ("step_type", "invalid step"),
    ("json", "invalid JSON"),
])
def test_invalid_event_evidence_cannot_pass(tmp_path: Path, fault: str, reason: str) -> None:
    reference = _run(tmp_path, "reference")
    recovered = _run(tmp_path, "recovered")
    events = _events(recovered)
    if fault == "duplicate":
        events.insert(1, events[0].copy())
    elif fault == "order":
        events[0], events[1] = events[1], events[0]
    elif fault == "sample_type":
        events[0]["sample_ids"][1] = True
    elif fault == "step_type":
        events[0]["global_step"] = True
    _write_events(recovered, 0, events)
    if fault == "json":
        with (recovered / "attempts/attempt-001/rank-0.jsonl").open("a") as stream:
            stream.write("{broken}\n")
    result = validate_runs(reference, recovered)
    assert not result["passed"], fault
    assert any(reason in value for value in result["differences"]), result


def test_missing_rank_log_is_reported(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference")
    recovered = _run(tmp_path, "recovered")
    (recovered / "attempts/attempt-001/rank-1.jsonl").unlink()
    result = validate_runs(reference, recovered)
    assert not result["passed"]
    assert any("missing rank log" in value for value in result["differences"]), result


@pytest.mark.parametrize("field,value", [("model_sha256", None), ("global_step", 3)])
def test_equal_invalid_summaries_cannot_pass(tmp_path: Path, field: str, value: object) -> None:
    reference = _run(tmp_path, "reference")
    recovered = _run(tmp_path, "recovered")
    for directory in (reference, recovered):
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary[field] = value
        path.write_text(json.dumps(summary))
    result = validate_runs(reference, recovered)
    assert not result["passed"]
    assert any(field in difference for difference in result["differences"])


def test_stale_records_are_ignored(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference")
    recovered = _run(tmp_path, "recovered")
    events = _events(recovered)
    stale = events[0] | {"attempt_id": "attempt-old", "global_step": 99}
    _write_events(recovered, 0, [stale, *events])
    assert validate_runs(reference, recovered)["passed"]


def test_rollback_and_truncated_failed_attempt_tail_remain_valid(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference")
    recovered = _run(tmp_path, "recovered")
    store = RunStore(recovered / "run.sqlite3")
    try:
        store.finish_attempt("attempt-001", "FAILED", 71, "worker exited")
        store.start_attempt("recovered", "attempt-002", 2, "checkpoint", 2)
        store.finish_attempt("attempt-002", "SUCCEEDED", 0, "completed")
    finally:
        store.close()
    for rank in (0, 1):
        source = recovered / "attempts/attempt-001" / f"rank-{rank}.jsonl"
        with source.open("a") as stream:
            stream.write('{"partial"')
        destination = recovered / "attempts/attempt-002" / f"rank-{rank}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(
            json.dumps(event | {"attempt_id": "attempt-002"}) + "\n"
            for event in _events_from_complete_lines(source)[2:]
        ))
    assert validate_runs(reference, recovered)["passed"]


def _events_from_complete_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True)
            if line.endswith("\n")]


def test_missing_attempt_index_returns_failed_report(tmp_path: Path) -> None:
    reference = _run(tmp_path, "reference")
    recovered = _run(tmp_path, "recovered")
    (recovered / "run.sqlite3").unlink()
    result = validate_runs(reference, recovered)
    assert not result["passed"]
    assert any("attempt index is unreadable" in value for value in result["differences"])
