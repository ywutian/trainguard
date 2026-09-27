"""Real CPU boundaries for guarded local checkpoint capacity."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.checkpoint import candidate_path, validate_checkpoint
from trainguard.config import load_config
from trainguard.controller import RunActiveError
from trainguard.events import append_event
from trainguard.run_store import RunStore


def _guarded_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **limits: int
) -> Path:
    key_path = tmp_path / "customer-sample.key"
    key_path.write_bytes(b"a-private-customer-key-with-at-least-32-bytes")
    key_path.chmod(0o600)
    monkeypatch.setenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE", str(key_path))
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["run"]["profile"] = "guarded"
    raw["training"]["total_steps"] = 3
    raw["recovery"]["max_restarts"] = 0
    raw["checkpoint"].update(
        mode="sync", interval_steps=1, keep_last_k=2,
        max_checkpoint_bytes=10_000_000, max_retained_bytes=20_000_000,
        min_free_bytes=1_000_000, max_event_log_bytes=1_000_000,
    )
    raw["checkpoint"].update(limits)
    source = tmp_path / "guarded.json"
    source.write_text(json.dumps(raw))
    return source


def test_event_log_refuses_bytes_above_its_bound(tmp_path: Path) -> None:
    path = tmp_path / "rank-0.jsonl"
    append_event(path, run_id="run", event_type="first")
    original = path.read_bytes()
    with pytest.raises(OSError, match="event log byte budget"):
        append_event(path, max_bytes=len(original), run_id="run", event_type="second")
    assert path.read_bytes() == original


def _indexed_statuses(run_dir: Path) -> tuple[str, list[tuple[str, str]]]:
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        run_status = database.execute("SELECT status FROM runs").fetchone()[0]
        attempts = database.execute(
            "SELECT attempt_id, status FROM attempts ORDER BY number"
        ).fetchall()
    return run_status, attempts


def _assert_no_workers(run_dir: Path) -> None:
    status = json.loads((run_dir / "run.json").read_text())
    store = RunStore(run_dir / "run.sqlite3", existing_only=True)
    try:
        controller._assert_no_owned_workers(
            run_dir, status["run_id"], store.attempts(status["run_id"])
        )
    finally:
        store.close()


def test_guarded_launch_log_exhaustion_closes_unlaunched_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch, max_event_log_bytes=1)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FAILED"
    assert "event log byte budget is exhausted" in status["reason"]
    assert _indexed_statuses(run_dir) == ("FAILED", [("attempt-001", "FAILED")])
    assert not (run_dir / "launcher.log").exists()


def test_guarded_cleanup_log_exhaustion_does_not_claim_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    original = controller.append_event
    saw_group_stopped = False

    def exhaust_on_group_stopped(path, *, max_bytes=None, **fields):
        nonlocal saw_group_stopped
        if fields.get("event_type") == "group_stopped":
            saw_group_stopped = True
            max_bytes = path.stat().st_size
        return original(path, max_bytes=max_bytes, **fields)

    monkeypatch.setattr(controller, "append_event", exhaust_on_group_stopped)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert saw_group_stopped
    assert not succeeded
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FAILED"
    assert "event log byte budget is exhausted" in status["reason"]
    assert _indexed_statuses(run_dir) == ("FAILED", [("attempt-001", "FAILED")])
    _assert_no_workers(run_dir)


def test_guarded_running_log_exhaustion_stops_workers_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    original_read = controller._read_events
    injected = False

    def exhaust_during_poll(run_dir, attempt_id, run_id, world_size, offsets, steps, completed):
        nonlocal injected
        if not injected:
            injected = True
            path = run_dir / "controller.jsonl"
            controller.append_event(
                path, max_bytes=path.stat().st_size, run_id=run_id,
                attempt_id=attempt_id, event_type="progress_observed",
            )
        return original_read(
            run_dir, attempt_id, run_id, world_size, offsets, steps, completed
        )

    monkeypatch.setattr(controller, "_read_events", exhaust_during_poll)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert injected
    assert not succeeded
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FAILED"
    assert "event log byte budget is exhausted" in status["reason"]
    assert _indexed_statuses(run_dir) == ("FAILED", [("attempt-001", "FAILED")])
    assert (run_dir / "attempts/attempt-001/worker-group-ended.json").is_file()
    _assert_no_workers(run_dir)


def test_resume_checkpoint_selection_log_exhaustion_closes_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    raw = json.loads(source.read_text())
    raw["training"]["total_steps"] = 5
    raw["recovery"]["max_restarts"] = 1
    source.write_text(json.dumps(raw))
    original_read = controller._read_events
    interrupted = False

    def interrupt_after_checkpoint(run_dir, attempt_id, run_id, world_size, offsets, steps, completed):
        nonlocal interrupted
        if not interrupted and list((run_dir / "checkpoints").glob("*/COMMITTED")):
            interrupted = True
            raise KeyboardInterrupt
        return original_read(
            run_dir, attempt_id, run_id, world_size, offsets, steps, completed
        )

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_read_events", interrupt_after_checkpoint)
        run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert interrupted
    assert not succeeded
    assert json.loads((run_dir / "run.json").read_text())["status"] == "INTERRUPTED"
    assert list((run_dir / "checkpoints").glob("*/COMMITTED"))
    original_limit = controller.event_log_limit

    def exhausted_limit(config):
        assert original_limit(config) is not None
        return (run_dir / "controller.jsonl").stat().st_size

    monkeypatch.setattr(controller, "event_log_limit", exhausted_limit)
    assert not controller.resume(run_dir)
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FAILED"
    assert "event log byte budget is exhausted" in status["reason"]
    assert _indexed_statuses(run_dir) == ("FAILED", [("attempt-001", "FAILED")])
    _assert_no_workers(run_dir)


def test_event_budget_does_not_close_run_with_live_owned_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch, max_event_log_bytes=1)
    original = controller.append_event
    live_worker = None

    def leave_owned_worker(path, *, max_bytes=None, **fields):
        nonlocal live_worker
        if fields.get("event_type") == "launch_requested" and live_worker is None:
            run_dir = path.parent
            live_worker = subprocess.Popen(
                [
                    sys.executable, "-c", "import time; time.sleep(30)",
                    "trainguard.trainer", "--run-dir", str(run_dir),
                    "--run-id", fields["run_id"], "--attempt-id", fields["attempt_id"],
                ],
                start_new_session=True,
            )
            for _ in range(20):
                if controller._owned_group_members(
                    run_dir, fields["run_id"], fields["attempt_id"]
                ):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("owned test worker did not appear")
        return original(path, max_bytes=max_bytes, **fields)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "append_event", leave_owned_worker)
        try:
            with pytest.raises(RunActiveError, match="still owns a worker"):
                controller.run(source, tmp_path / "runs")
            run_dir = next((tmp_path / "runs").iterdir())
            assert json.loads((run_dir / "run.json").read_text())["status"] == "RUNNING"
            assert _indexed_statuses(run_dir) == ("RUNNING", [("attempt-001", "RUNNING")])
        finally:
            if live_worker is not None:
                live_worker.terminate()
                live_worker.wait(timeout=5)
    assert not controller.resume(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["status"] == "FAILED"
    assert _indexed_statuses(run_dir) == ("FAILED", [("attempt-001", "FAILED")])


def test_guarded_run_keeps_two_valid_backups_within_declared_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert succeeded, (run_dir / "launcher.log").read_text()
    config = load_config(run_dir / "config.json")
    candidates = sorted((run_dir / "checkpoints").glob("*/COMMITTED"))
    assert len(candidates) >= 2
    for marker in candidates:
        record = validate_checkpoint(
            marker.parent, config, run_dir.name,
            decode_payload=True, require_trainable_state=True,
        )
        assert sum(item["size"] for item in record.manifest["files"]) <= (
            config.checkpoint.max_checkpoint_bytes
        )
        assert sum(
            file.stat().st_size for file in marker.parent.rglob("*") if file.is_file()
        ) <= config.checkpoint.max_checkpoint_bytes
    for path in (run_dir / "controller.jsonl", *sorted((run_dir / "attempts").glob("*/*.jsonl"))):
        assert path.stat().st_size <= config.checkpoint.max_event_log_bytes


def test_one_rank_low_space_rejects_before_candidate_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os, shutil\n"
        "if os.environ.get('RANK') == '0':\n"
        "    original = shutil.disk_usage\n"
        "    def limited(path):\n"
        "        result = original(path)\n"
        "        return type(result)(result.total, result.used, 1)\n"
        "    shutil.disk_usage = limited\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, (str(hook), os.environ.get("PYTHONPATH"))))
    )
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    assert "another rank rejected checkpoint capacity" in (run_dir / "launcher.log").read_text()
    assert not list((run_dir / "checkpoints").glob("step-*"))


def test_oversized_checkpoint_never_publishes_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(
        tmp_path, monkeypatch, max_checkpoint_bytes=1, max_retained_bytes=2,
    )
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    assert "checkpoint exceeds its configured byte limit" in (
        run_dir / "launcher.log"
    ).read_text()
    assert list((run_dir / "checkpoints").glob("step-*"))
    assert not list((run_dir / "checkpoints").glob("*/COMMITTED"))


def test_guarded_checkpoint_limit_includes_manifest_and_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    reference, succeeded = controller.run(source, tmp_path / "reference")
    assert succeeded
    candidate = next((reference / "checkpoints").glob("*/COMMITTED")).parent
    config = load_config(reference / "config.json")
    record = validate_checkpoint(candidate, config, reference.name)
    payload_bytes = sum(item["size"] for item in record.manifest["files"])
    complete_bytes = sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
    assert complete_bytes > payload_bytes
    limit = payload_bytes + (complete_bytes - payload_bytes) // 2
    source = _guarded_config(
        tmp_path, monkeypatch, max_checkpoint_bytes=limit, max_retained_bytes=2 * limit,
    )
    run_dir, succeeded = controller.run(source, tmp_path / "bounded")
    assert not succeeded
    assert "checkpoint exceeds its configured byte limit" in (run_dir / "launcher.log").read_text()
    assert not list((run_dir / "checkpoints").glob("*/COMMITTED"))


def test_guarded_config_needs_two_checkpoint_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    raw = json.loads(source.read_text())
    raw["training"]["total_steps"] = 1
    raw["checkpoint"]["interval_steps"] = 10
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="two distinct checkpoint boundaries"):
        load_config(source)


def test_guarded_completion_rejects_only_one_verified_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    original_scan = controller._scan_checkpoints
    damaged_marker = None

    def damage_older_at_audit(run_dir, *args, **kwargs):
        nonlocal damaged_marker
        if kwargs.get("audit") and damaged_marker is None:
            marker = candidate_path(run_dir, "attempt-001", 2) / "COMMITTED"
            damaged_marker = marker.read_bytes()
            marker.write_bytes(b"invalid backup\n")
        return original_scan(run_dir, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_scan_checkpoints", damage_older_at_audit)
        with pytest.raises(RuntimeError, match="fewer than two verified"):
            controller.run(source, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FINALIZING"
    assert status["post_run_audit"]["status"] == "FAILED"
    assert status["post_run_audit"]["valid_retained_count"] == 1
    assert damaged_marker is not None
    (candidate_path(run_dir, "attempt-001", 2) / "COMMITTED").write_bytes(damaged_marker)
    assert controller.resume(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["post_run_audit"][
        "valid_retained_count"
    ] >= 2


def test_guarded_audit_counts_uncommitted_candidate_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(
        tmp_path, monkeypatch, max_checkpoint_bytes=2_000_000,
        max_retained_bytes=4_000_000,
    )
    original_scan = controller._scan_checkpoints

    def add_uncommitted_at_audit(run_dir, *args, **kwargs):
        if kwargs.get("audit"):
            candidate = candidate_path(run_dir, "attempt-999", 999)
            if not candidate.exists():
                candidate.mkdir()
                (candidate / "unfinished.bin").write_bytes(b"x" * 2_000_000)
        return original_scan(run_dir, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_scan_checkpoints", add_uncommitted_at_audit)
        with pytest.raises(RuntimeError, match="capacity budget is unsatisfied"):
            controller.run(source, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    audit = json.loads((run_dir / "run.json").read_text())["post_run_audit"]
    assert audit["status"] == "FAILED"
    assert audit["valid_retained_count"] >= 2
    assert audit["unverified_candidate_count"] >= 1
    assert audit["unverified_candidate_bytes"] >= 2_000_000
    assert audit["checkpoint_file_bytes"] > 4_000_000
    assert audit["checkpoint_file_budget_satisfied"] is False
