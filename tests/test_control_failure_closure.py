"""Real local failure boundaries for run and attempt state convergence."""

from __future__ import annotations

import errno
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.config import load_config
from trainguard.controller import RunActiveError
from trainguard.run_store import RunStore

CPU_DEMO = Path(__file__).parents[1] / "configs/cpu_demo.yaml"


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text())


def _index(run_dir: Path) -> tuple[list[str], list[tuple[str, str]]]:
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        runs = [row[0] for row in database.execute("SELECT status FROM runs")]
        attempts = database.execute(
            "SELECT attempt_id, status FROM attempts ORDER BY number"
        ).fetchall()
    return runs, attempts


def _no_owned_workers(run_dir: Path) -> bool:
    status = _status(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        attempt_ids = [
            row[0] for row in database.execute("SELECT attempt_id FROM attempts")
        ]
    return all(
        not controller._owned_group_members(run_dir, status["run_id"], attempt_id)
        for attempt_id in attempt_ids or ["attempt-001"]
    )


def test_index_mode_failure_after_lock_converges_before_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = Path.chmod

    def deny_index_mode(path, mode, *args, **kwargs):
        if path.name == "run.sqlite3":
            raise PermissionError("injected index mode failure")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", deny_index_mode)
    run_dir, succeeded = controller.run(CPU_DEMO, tmp_path / "runs")
    assert not succeeded
    status = _status(run_dir)
    assert status["status"] == "FAILED"
    assert status["run_index_terminal_unverified"] is True
    assert "injected index mode failure" in status["reason"]
    assert _index(run_dir) == ([], [])
    assert _no_owned_workers(run_dir)
    assert not controller.resume(run_dir)


@pytest.mark.parametrize(
    "error", [PermissionError("injected lock denial"), RunActiveError("lock held")]
)
def test_lock_acquisition_failure_cannot_publish_running_or_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception,
) -> None:
    def deny_lock(_run_dir):
        raise error

    monkeypatch.setattr(controller, "_controller_lock", deny_lock)
    with pytest.raises(type(error), match=str(error)):
        controller.run(CPU_DEMO, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    assert not (run_dir / "run.json").exists()
    assert _index(run_dir) == ([], [])
    assert not (run_dir / "attempts").exists()


@pytest.mark.parametrize("after_commit", [False, True])
def test_run_index_creation_failure_records_actual_row_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_commit: bool
) -> None:
    original = RunStore.create_run

    def fail_creation(self, *args, **kwargs):
        if after_commit:
            original(self, *args, **kwargs)
        raise sqlite3.OperationalError("injected run insertion failure")

    monkeypatch.setattr(RunStore, "create_run", fail_creation)
    run_dir, succeeded = controller.run(CPU_DEMO, tmp_path / "runs")
    assert not succeeded
    status = _status(run_dir)
    assert status["status"] == "FAILED"
    assert "injected run insertion failure" in status["reason"]
    assert _index(run_dir) == (
        ["FAILED"] if after_commit else [], []
    )
    if not after_commit:
        assert status["run_index_terminal_unverified"] is True
        assert "run index row was not established" in status["reason"]
        assert not controller.resume(run_dir)
    else:
        assert "run_index_terminal_unverified" not in status
    assert _no_owned_workers(run_dir)


def test_resume_index_reconstruction_failure_records_missing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(RunStore, "create_run", lambda *a, **k: (_ for _ in ()).throw(SystemExit(73)))
        with pytest.raises(SystemExit):
            controller.run(CPU_DEMO, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    assert _status(run_dir)["status"] == "RUNNING"
    with monkeypatch.context() as patch:
        patch.setattr(
            RunStore, "create_run",
            lambda *a, **k: (_ for _ in ()).throw(
                sqlite3.OperationalError("injected index reconstruction failure")
            ),
        )
        assert not controller.resume(run_dir)
    status = _status(run_dir)
    assert status["status"] == "FAILED"
    assert status["run_index_terminal_unverified"] is True
    assert "run index row was not established" in status["reason"]
    assert _index(run_dir) == ([], [])
    assert not controller.resume(run_dir)


def test_unverified_terminal_index_write_blocks_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_status = RunStore.set_run_status

    def fail_prune(*args, **kwargs):
        raise RuntimeError("injected prelaunch failure")

    def refuse_terminal_status(self, run_id, status):
        if status == "FAILED":
            raise sqlite3.OperationalError("injected terminal index failure")
        return original_status(self, run_id, status)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "prune_checkpoints", fail_prune)
        patch.setattr(RunStore, "set_run_status", refuse_terminal_status)
        run_dir, succeeded = controller.run(CPU_DEMO, tmp_path / "runs")
    assert not succeeded
    status = _status(run_dir)
    assert status["status"] == "FAILED"
    assert status["run_index_terminal_unverified"] is True
    assert "run index terminal update failed" in status["reason"]
    assert _index(run_dir) == (["RUNNING"], [])
    assert not controller.resume(run_dir)
    assert _no_owned_workers(run_dir)


@pytest.mark.parametrize(
    ("failure_point", "expected_status"),
    [
        ("prune_runtime", "FAILED"),
        ("prune_value", "FAILED"),
        ("set_run_status", "FAILED"),
        ("start_attempt", "FAILED"),
        ("prelaunch_interrupt", "INTERRUPTED"),
    ],
)
def test_prelaunch_failure_converges_without_phantom_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    failure_point: str, expected_status: str,
) -> None:
    if failure_point.startswith("prune") or failure_point == "prelaunch_interrupt":
        error: BaseException
        if failure_point == "prune_runtime":
            error = RuntimeError("injected retention inspection failure")
        elif failure_point == "prune_value":
            error = ValueError("injected checkpoint format failure")
        else:
            error = KeyboardInterrupt()

        def fail_prune(*args, **kwargs):
            raise error

        monkeypatch.setattr(controller, "prune_checkpoints", fail_prune)
    elif failure_point == "set_run_status":
        original = RunStore.set_run_status
        failed = False

        def fail_status_once(self, *args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise sqlite3.OperationalError("injected run status failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(RunStore, "set_run_status", fail_status_once)
    else:
        def fail_attempt_once(self, *args, **kwargs):
            raise sqlite3.OperationalError("injected attempt insertion failure")

        monkeypatch.setattr(RunStore, "start_attempt", fail_attempt_once)

    run_dir, succeeded = controller.run(CPU_DEMO, tmp_path / "runs")
    assert not succeeded
    status = _status(run_dir)
    assert status["status"] == expected_status
    assert status["attempt_id"] is None
    assert _index(run_dir) == ([expected_status], [])
    assert _no_owned_workers(run_dir)
    if failure_point == "start_attempt":
        assert "attempt row was not established" in status["reason"]


def _recoverable_stopped_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw = load_config(CPU_DEMO).model_dump()
    raw["checkpoint"].update(mode="sync", interval_steps=1)
    raw["fault"].update(kind="worker_exit", step=3, rank=0)
    raw["recovery"].update(max_restarts=1, progress_timeout_seconds=30)
    source = tmp_path / "recoverable.json"
    source.write_text(json.dumps(raw))
    original_launch = controller._launch_attempt

    class SimulatedControllerExit(BaseException):
        pass

    def stop_after_first(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        if args[3] == "attempt-001":
            assert not result.succeeded
            raise SimulatedControllerExit
        return result

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_launch_attempt", stop_after_first)
        with pytest.raises(SimulatedControllerExit):
            controller.run(source, tmp_path / "runs", allow_experiment=True)
    run_dir = next((tmp_path / "runs").iterdir())
    assert list((run_dir / "checkpoints").glob("*/COMMITTED"))
    assert _no_owned_workers(run_dir)
    return run_dir


@pytest.mark.parametrize("failure_point", ["checkpoint_selected", "record_recovery"])
def test_resume_prelaunch_failure_converges_after_real_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    run_dir = _recoverable_stopped_run(tmp_path, monkeypatch)
    if failure_point == "checkpoint_selected":
        original = controller.append_event

        def fail_selection(path, *, max_bytes=None, **fields):
            if fields.get("event_type") == "checkpoint_selected":
                raise PermissionError("injected selection log denial")
            return original(path, max_bytes=max_bytes, **fields)

        monkeypatch.setattr(controller, "append_event", fail_selection)
    else:
        def fail_recovery_once(self, *args, **kwargs):
            raise sqlite3.OperationalError("injected recovery decision failure")

        monkeypatch.setattr(RunStore, "record_recovery", fail_recovery_once)

    assert not controller.resume(run_dir)
    status = _status(run_dir)
    assert status["status"] == "FAILED"
    runs, attempts = _index(run_dir)
    assert runs == ["FAILED"]
    if failure_point == "checkpoint_selected":
        assert "injected selection log denial" in status["reason"]
        assert attempts == [("attempt-001", "INTERRUPTED")]
    else:
        assert "injected recovery decision failure" in status["reason"]
        assert attempts == [
            ("attempt-001", "INTERRUPTED"), ("attempt-002", "FAILED")
        ]
    assert _no_owned_workers(run_dir)


def test_launcher_creation_failure_closes_attempt_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subprocess.Popen

    def fail_launcher(command, *args, **kwargs):
        if isinstance(command, list) and "torch.distributed.run" in command:
            raise OSError(errno.EAGAIN, "injected process creation failure")
        return original(command, *args, **kwargs)

    monkeypatch.setattr(controller.subprocess, "Popen", fail_launcher)
    run_dir, succeeded = controller.run(CPU_DEMO, tmp_path / "runs")
    assert not succeeded
    status = _status(run_dir)
    assert status["status"] == "FAILED"
    assert "injected process creation failure" in status["reason"]
    assert _index(run_dir) == (["FAILED"], [("attempt-001", "FAILED")])
    assert _no_owned_workers(run_dir)


def test_launcher_exception_with_live_worker_cannot_mark_run_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subprocess.Popen
    worker = None

    def leave_worker_then_fail(command, *args, **kwargs):
        nonlocal worker
        if isinstance(command, list) and "torch.distributed.run" in command:
            run_dir = Path(command[command.index("--run-dir") + 1])
            run_id = command[command.index("--run-id") + 1]
            attempt_id = command[command.index("--attempt-id") + 1]
            worker = original(
                [
                    sys.executable, "-c", "import time; time.sleep(30)",
                    "trainguard.trainer", "--run-dir", str(run_dir),
                    "--run-id", run_id, "--attempt-id", attempt_id,
                ],
                start_new_session=True,
            )
            for _ in range(30):
                if controller._owned_group_members(run_dir, run_id, attempt_id):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("owned test worker did not appear")
            raise OSError(errno.EAGAIN, "injected launch acknowledgement failure")
        return original(command, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller.subprocess, "Popen", leave_worker_then_fail)
        try:
            with pytest.raises(RunActiveError, match="still owns a worker"):
                controller.run(CPU_DEMO, tmp_path / "runs")
            run_dir = next((tmp_path / "runs").iterdir())
            assert _status(run_dir)["status"] == "RUNNING"
            assert _index(run_dir) == (["RUNNING"], [("attempt-001", "RUNNING")])
        finally:
            if worker is not None:
                worker.terminate()
                worker.wait(timeout=5)
    assert controller.resume(run_dir)
    assert _index(run_dir) == (["SUCCEEDED"], [("attempt-001", "SUCCEEDED")])


def test_attempt_index_completion_failure_cannot_claim_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RunStore.finish_attempt
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise sqlite3.OperationalError("injected attempt completion failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RunStore, "finish_attempt", fail_once)
    run_dir, succeeded = controller.run(CPU_DEMO, tmp_path / "runs")
    assert failed
    assert not succeeded
    assert "injected attempt completion failure" in _status(run_dir)["reason"]
    assert _index(run_dir) == (["FAILED"], [("attempt-001", "FAILED")])
    assert _no_owned_workers(run_dir)
