"""Bounded single-node controller for fixed-size process-group recovery."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from trainguard.checkpoint import CheckpointInvalid, CheckpointRecord, validate_checkpoint
from trainguard.config import ProjectConfig, load_config
from trainguard.events import utc_now, write_json_atomic
from trainguard.records import parse_event
from trainguard.run_store import RunStore
from trainguard.validation import completion_errors


class RunActiveError(RuntimeError):
    """A controller or an owned worker group still runs for this directory."""


@dataclass
class AttemptResult:
    succeeded: bool
    reason: str
    exit_code: int | None
    max_step: int


def _available_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _owned_group_members(
    run_dir: Path, run_id: str, attempt_id: str, group_id: int | None = None
) -> list[int]:
    result = subprocess.run(
        ["ps", "axww", "-o", "pid=", "-o", "pgid=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("cannot inspect worker process ownership")
    members = []
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_text, group_text, command = parts
        if group_id is not None and int(group_text) != group_id:
            continue
        if (
            ("trainguard.trainer" in command or "torch.distributed.run" in command)
            and f"--run-dir {run_dir}" in command
            and f"--run-id {run_id}" in command
            and f"--attempt-id {attempt_id}" in command
        ):
            members.append(int(pid_text))
    return members


def _stop_process_group(
    process: subprocess.Popen[bytes], run_dir: Path, run_id: str, attempt_id: str
) -> None:
    if process.poll() is not None and not _owned_group_members(
        run_dir, run_id, attempt_id, process.pid
    ):
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _owned_group_members(run_dir, run_id, attempt_id, process.pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def _controller_lock(run_dir: Path) -> Iterator[None]:
    with (run_dir / ".controller.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunActiveError("another controller owns this run") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _pid_identity(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _owned_process_alive(
    pid: int | None,
    identity: str | None,
    run_id: str,
    attempt_id: str,
    run_dir: Path,
) -> bool:
    if pid is not None and identity and _pid_identity(pid) == identity:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and f"--run-id {run_id}" in result.stdout:
            return True
    return bool(_owned_group_members(run_dir, run_id, attempt_id, pid))


def _read_events(
    run_dir: Path,
    attempt_id: str,
    run_id: str,
    world_size: int,
    offsets: dict[int, int],
    steps: dict[int, int],
    completed: set[int],
) -> bool:
    progressed = False
    for rank in range(world_size):
        path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            stream.seek(offsets.get(rank, 0))
            while line := stream.readline():
                if not line.endswith("\n"):
                    break
                offsets[rank] = stream.tell()
                try:
                    event = parse_event(line, run_id, attempt_id, rank)
                except ValueError as exc:
                    raise ValueError(f"{path.name}: {exc}") from exc
                if event is None:
                    continue
                if event.get("event_type") == "step_completed":
                    step = event.get("global_step")
                    if type(step) is int and step > steps.get(rank, 0):
                        steps[rank] = step
                        progressed = True
                if event.get("event_type") == "training_completed":
                    completed.add(rank)
    return progressed


def _max_step(run_dir: Path, attempt_id: str, run_id: str, world_size: int) -> int:
    steps: dict[int, int] = {}
    _read_events(run_dir, attempt_id, run_id, world_size, {}, steps, set())
    return max(steps.values(), default=0)


def _scan_checkpoints(
    run_dir: Path, config: ProjectConfig, run_id: str, store: RunStore
) -> CheckpointRecord | None:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return None
    valid = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            record = validate_checkpoint(path, config, run_id)
        except (CheckpointInvalid, OSError) as exc:
            store.record_checkpoint(str(path), run_id, None, None, "INVALID", str(exc))
        else:
            store.record_checkpoint(
                str(path), run_id, record.attempt_id, record.global_step, "VALID", None
            )
            valid.append(record)
    return max(valid, key=lambda item: (item.global_step, item.path.name), default=None)


def _valid_attempt_summary(
    run_dir: Path, attempt_id: str, config: ProjectConfig, run_id: str
) -> dict | None:
    path = run_dir / "attempts" / attempt_id / "summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeError):
        return None
    if completion_errors(run_dir, attempt_id, config, run_id, summary):
        return None
    return summary


def _launch_attempt(
    run_dir: Path,
    config: ProjectConfig,
    run_id: str,
    attempt_id: str,
    resume_checkpoint: CheckpointRecord | None,
    store: RunStore,
) -> AttemptResult:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={config.run.world_size}",
        "--max-restarts=0",
        "--master-addr=127.0.0.1",
        f"--master-port={_available_local_port()}",
        "-m",
        "trainguard.trainer",
        "--config",
        str(run_dir / "config.json"),
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--attempt-id",
        attempt_id,
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint.path)])
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    offsets: dict[int, int] = {}
    steps: dict[int, int] = {}
    completed: set[int] = set()
    started = time.monotonic()
    last_progress = started
    reason = "launcher exited before completion"
    exit_code = None
    with (run_dir / "launcher.log").open("ab") as log:
        log.write(f"\n=== {attempt_id} ===\n".encode())
        log.flush()
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            store.set_pid(attempt_id, process.pid, _pid_identity(process.pid))
            while True:
                if _read_events(
                    run_dir,
                    attempt_id,
                    run_id,
                    config.run.world_size,
                    offsets,
                    steps,
                    completed,
                ):
                    last_progress = time.monotonic()
                exit_code = process.poll()
                if exit_code is not None:
                    break
                if time.monotonic() - started > config.run.timeout_seconds:
                    reason = f"attempt exceeded {config.run.timeout_seconds} seconds"
                    _stop_process_group(process, run_dir, run_id, attempt_id)
                    exit_code = process.poll()
                    break
                if time.monotonic() - last_progress > config.recovery.progress_timeout_seconds:
                    reason = "step progress stalled"
                    _stop_process_group(process, run_dir, run_id, attempt_id)
                    exit_code = process.poll()
                    break
                time.sleep(0.2)
            orphaned_workers = bool(_owned_group_members(run_dir, run_id, attempt_id, process.pid))
            if orphaned_workers:
                reason = "launcher exited while owned workers remained"
            _read_events(
                run_dir, attempt_id, run_id, config.run.world_size, offsets, steps, completed
            )
            summary = _valid_attempt_summary(run_dir, attempt_id, config, run_id)
            if exit_code == 0 and not orphaned_workers and summary is not None:
                return AttemptResult(
                    True, "completed all training steps", 0, max(steps.values(), default=0)
                )
            if reason == "launcher exited before completion":
                reason = f"launcher exit code {exit_code}; completion evidence missing or invalid"
        except KeyboardInterrupt:
            reason = "interrupted by user"
        except Exception as exc:  # noqa: BLE001 - cleanup covers every controller failure
            reason = f"controller {type(exc).__name__}: {exc}"
            write_json_atomic(
                run_dir / "attempts" / attempt_id / "controller-error.json",
                {"reason": reason, "time": utc_now()},
            )
        finally:
            try:
                _stop_process_group(process, run_dir, run_id, attempt_id)
            except BaseException as cleanup:
                raise RunActiveError(f"{reason}; worker cleanup failed: {cleanup}") from cleanup
    return AttemptResult(False, reason, process.poll(), max(steps.values(), default=0))


def _set_status(run_dir: Path, status: dict, value: str, reason: str) -> None:
    status.update(status=value, reason=reason)
    if value in {"SUCCEEDED", "FAILED", "INTERRUPTED"}:
        status["finished_at"] = utc_now()
    write_json_atomic(run_dir / "run.json", status)


def _drive(run_dir: Path, status: dict, store: RunStore) -> bool:
    run_id = status["run_id"]
    config = load_config(run_dir / "config.json")
    attempts = store.attempts(run_id)
    if attempts:
        last = attempts[-1]
        if _owned_process_alive(
            last["pid"],
            last["pid_identity"],
            run_id,
            last["attempt_id"],
            run_dir,
        ):
            raise RunActiveError(f"attempt {last['attempt_id']} still owns a worker group")
        summary = (
            _valid_attempt_summary(run_dir, last["attempt_id"], config, run_id)
            if last["status"] in {"RUNNING", "SUCCEEDED"}
            else None
        )
        if summary is not None:
            write_json_atomic(run_dir / "summary.json", summary)
            if last["status"] == "RUNNING":
                store.finish_attempt(
                    last["attempt_id"], "SUCCEEDED", 0, "completed before controller exit"
                )
            store.set_run_status(run_id, "SUCCEEDED")
            _set_status(run_dir, status, "SUCCEEDED", "completed before controller exit")
            return True
        if last["status"] == "RUNNING":
            attempt_dir = run_dir / "attempts" / last["attempt_id"]
            if last["pid"] is None and (not attempt_dir.exists() or not any(attempt_dir.iterdir())):
                store.discard_unlaunched_attempt(last["attempt_id"])
            else:
                store.finish_attempt(last["attempt_id"], "INTERRUPTED", None, "controller exited")
            attempts = store.attempts(run_id)

    while True:
        number = len(attempts) + 1
        selected = None
        if attempts:
            if number > config.recovery.max_restarts + 1:
                reason = "restart limit exhausted"
                store.set_run_status(run_id, "FAILED")
                _set_status(run_dir, status, "FAILED", reason)
                return False
            selected = _scan_checkpoints(run_dir, config, run_id, store)
            if selected is None:
                reason = "no valid checkpoint remains for recovery"
                store.set_run_status(run_id, "FAILED")
                _set_status(run_dir, status, "FAILED", reason)
                return False

        attempt_id = f"attempt-{number:03d}"
        attempt_dir = run_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if any(attempt_dir.iterdir()):
            raise RunActiveError(f"unrecorded attempt directory contains files: {attempt_dir}")
        status["attempt_id"] = attempt_id
        _set_status(run_dir, status, "RUNNING", "training in progress")
        store.set_run_status(run_id, "RUNNING")
        store.start_attempt(
            run_id,
            attempt_id,
            number,
            str(selected.path) if selected else None,
            selected.global_step if selected else 0,
        )
        if selected is not None:
            previous = attempts[-1]
            previous_max = _max_step(run_dir, previous["attempt_id"], run_id, config.run.world_size)
            store.record_recovery(
                run_id,
                previous["attempt_id"],
                attempt_id,
                str(selected.path),
                selected.global_step,
                max(0, previous_max - selected.global_step),
            )
        result = _launch_attempt(run_dir, config, run_id, attempt_id, selected, store)
        store.finish_attempt(
            attempt_id,
            "SUCCEEDED" if result.succeeded else "FAILED",
            result.exit_code,
            result.reason,
        )
        _scan_checkpoints(run_dir, config, run_id, store)
        if result.succeeded:
            summary = _valid_attempt_summary(run_dir, attempt_id, config, run_id)
            assert summary is not None
            write_json_atomic(run_dir / "summary.json", summary)
            store.set_run_status(run_id, "SUCCEEDED")
            _set_status(run_dir, status, "SUCCEEDED", result.reason)
            return True
        if result.reason == "interrupted by user":
            store.set_run_status(run_id, "INTERRUPTED")
            _set_status(run_dir, status, "INTERRUPTED", result.reason)
            return False
        attempts = store.attempts(run_id)


def run(config_path: Path, output_root: Path) -> tuple[Path, bool]:
    config = load_config(config_path.resolve())
    run_id = uuid.uuid4().hex[:12]
    run_dir = (output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(run_dir / "config.json", config.model_dump())
    started_at = utc_now()
    status = {
        "run_id": run_id,
        "attempt_id": None,
        "status": "RUNNING",
        "config_fingerprint": config.fingerprint(),
        "started_at": started_at,
        "config": config.model_dump(),
    }
    write_json_atomic(run_dir / "run.json", status)
    store = RunStore(run_dir / "run.sqlite3")
    try:
        store.create_run(run_id, config.fingerprint(), started_at)
        with _controller_lock(run_dir):
            succeeded = _drive(run_dir, status, store)
    finally:
        store.close()
    return run_dir, succeeded


def resume(run_dir: Path) -> bool:
    run_dir = run_dir.resolve()
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.json")
    if config.fingerprint() != status["config_fingerprint"]:
        raise ValueError("saved config fingerprint differs from run metadata")
    store = RunStore(run_dir / "run.sqlite3")
    try:
        with _controller_lock(run_dir):
            if status["status"] == "SUCCEEDED":
                attempts = store.attempts(status["run_id"])
                if (
                    not attempts
                    or _valid_attempt_summary(
                        run_dir, attempts[-1]["attempt_id"], config, status["run_id"]
                    )
                    is None
                ):
                    store.set_run_status(status["run_id"], "FAILED")
                    _set_status(run_dir, status, "FAILED", "completion evidence missing or invalid")
                    return False
            return _drive(run_dir, status, store)
    finally:
        store.close()
