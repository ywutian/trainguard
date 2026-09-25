import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from trainguard.controller import (
    _owned_process_alive,
    _pid_identity,
    _read_events,
    _stop_process_group,
)


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


def test_orphaned_owned_worker_is_detected_after_launcher_exits(tmp_path: Path) -> None:
    run_id = "run-orphan"
    attempt_id = "attempt-001"
    script = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',"
        "'trainguard.trainer','--run-dir',sys.argv[1],"
        "'--run-id',sys.argv[2],'--attempt-id',sys.argv[3]])"
    )
    launcher = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), run_id, attempt_id],
        start_new_session=True,
    )
    identity = _pid_identity(launcher.pid)
    try:
        assert launcher.wait(timeout=5) == 0
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if _owned_process_alive(launcher.pid, identity, run_id, attempt_id, tmp_path):
                break
            time.sleep(0.05)
        assert _owned_process_alive(launcher.pid, identity, run_id, attempt_id, tmp_path)
        _stop_process_group(launcher, tmp_path, run_id, attempt_id)
        assert not _owned_process_alive(launcher.pid, identity, run_id, attempt_id, tmp_path)
    finally:
        try:
            os.killpg(launcher.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
