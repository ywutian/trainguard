import json
from pathlib import Path

from trainguard.controller import _read_events


def test_stale_attempt_and_run_events_cannot_advance_progress(tmp_path: Path) -> None:
    path = tmp_path / "attempts" / "attempt-002" / "rank-0.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {"run_id": "run-a", "attempt_id": "attempt-001", "rank": 0,
         "event_type": "step_completed", "global_step": 99},
        {"run_id": "run-b", "attempt_id": "attempt-002", "rank": 0,
         "event_type": "step_completed", "global_step": 99},
        {"run_id": "run-a", "attempt_id": "attempt-002", "rank": 0,
         "event_type": "step_completed", "global_step": 2},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    offsets: dict[int, int] = {}
    steps: dict[int, int] = {}
    completed: set[int] = set()
    assert _read_events(tmp_path, "attempt-002", "run-a", 1, offsets, steps, completed)
    assert steps == {0: 2}
    assert not _read_events(tmp_path, "attempt-002", "run-a", 1, offsets, steps, completed)
