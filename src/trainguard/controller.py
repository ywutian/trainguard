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
from trainguard.run_store import RunStore


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


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
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
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _owned_process_alive(pid: int | None, identity: str | None, run_id: str) -> bool:
    if pid is None or not identity or _pid_identity(pid) != identity:
        return False
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True, text=True, check=False,
    )
    return result.returncode == 0 and f"--run-id {run_id}" in result.stdout


def _read_events(
    run_dir: Path, attempt_id: str, run_id: str, world_size: int,
    offsets: dict[int, int], steps: dict[int, int], completed: set[int],
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
                    event = json.loads(line)
                except ValueError:
                    continue
                if (event.get("run_id"), event.get("attempt_id"), event.get("rank")) != (
                    run_id, attempt_id, rank
                ):
                    continue
                if event.get("event_type") == "step_completed":
                    step = event.get("global_step")
                    if isinstance(step, int) and step > steps.get(rank, 0):
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
    except ValueError:
        return None
    expected = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "config_fingerprint": config.fingerprint(),
        "global_step": config.training.total_steps,
        "world_size": config.run.world_size,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None
    return summary


def _launch_attempt(
    run_dir: Path, config: ProjectConfig, run_id: str, attempt_id: str,
    resume_checkpoint: CheckpointRecord | None, store: RunStore,
) -> AttemptResult:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
        f"--nproc-per-node={config.run.world_size}", "--max-restarts=0",
        "--master-addr=127.0.0.1", f"--master-port={_available_local_port()}",
        "-m", "trainguard.trainer", "--config", str(run_dir / "config.json"),
        "--run-dir", str(run_dir), "--run-id", run_id, "--attempt-id", attempt_id,
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
            command, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        store.set_pid(attempt_id, process.pid, _pid_identity(process.pid))
        try:
            while True:
                if _read_events(
                    run_dir, attempt_id, run_id, config.run.world_size,
                    offsets, steps, completed,
                ):
                    last_progress = time.monotonic()
                exit_code = process.poll()
                if exit_code is not None:
                    break
                if time.monotonic() - started > config.run.timeout_seconds:
                    reason = f"attempt exceeded {config.run.timeout_seconds} seconds"
                    _stop_process_group(process)
                    exit_code = process.poll()
                    break
                if time.monotonic() - last_progress > config.recovery.progress_timeout_seconds:
                    reason = "step progress stalled"
                    _stop_process_group(process)
                    exit_code = process.poll()
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            reason = "interrupted by user"
            _stop_process_group(process)
            exit_code = process.poll()
    _read_events(run_dir, attempt_id, run_id, config.run.world_size, offsets, steps, completed)
    summary = _valid_attempt_summary(run_dir, attempt_id, config, run_id)
    if exit_code == 0 and summary is not None and len(completed) == config.run.world_size:
        return AttemptResult(True, "completed all training steps", 0, max(steps.values(), default=0))
    if reason == "launcher exited before completion":
        reason = f"launcher exit code {exit_code}; summary or rank completion missing"
    return AttemptResult(False, reason, exit_code, max(steps.values(), default=0))


def _set_status(run_dir: Path, status: dict, value: str, reason: str) -> None:
    status.update(status=value, reason=reason)
    if value in {"SUCCEEDED", "FAILED", "INTERRUPTED"}:
        status["finished_at"] = utc_now()
    write_json_atomic(run_dir / "run.json", status)


def _drive(run_dir: Path, status: dict, store: RunStore) -> bool:
    run_id = status["run_id"]
    config = load_config(run_dir / "config.json")
    attempts = store.attempts(run_id)
    if attempts and attempts[-1]["status"] == "RUNNING":
        last = attempts[-1]
        if _owned_process_alive(last["pid"], last["pid_identity"], run_id):
            raise RunActiveError(f"attempt {last['attempt_id']} still owns a worker group")
        summary = _valid_attempt_summary(run_dir, last["attempt_id"], config, run_id)
        if summary is not None:
            write_json_atomic(run_dir / "summary.json", summary)
            store.finish_attempt(last["attempt_id"], "SUCCEEDED", 0, "completed before controller exit")
            store.set_run_status(run_id, "SUCCEEDED")
            _set_status(run_dir, status, "SUCCEEDED", "completed before controller exit")
            return True
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
        (run_dir / "attempts" / attempt_id).mkdir(parents=True, exist_ok=False)
        status["attempt_id"] = attempt_id
        _set_status(run_dir, status, "RUNNING", "training in progress")
        store.set_run_status(run_id, "RUNNING")
        store.start_attempt(
            run_id, attempt_id, number,
            str(selected.path) if selected else None,
            selected.global_step if selected else 0,
        )
        if selected is not None:
            previous = attempts[-1]
            previous_max = _max_step(run_dir, previous["attempt_id"], run_id, config.run.world_size)
            store.record_recovery(
                run_id, previous["attempt_id"], attempt_id, str(selected.path),
                selected.global_step, max(0, previous_max - selected.global_step),
            )
        result = _launch_attempt(run_dir, config, run_id, attempt_id, selected, store)
        store.finish_attempt(
            attempt_id, "SUCCEEDED" if result.succeeded else "FAILED",
            result.exit_code, result.reason,
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
    if status["status"] == "SUCCEEDED":
        return True
    store = RunStore(run_dir / "run.sqlite3")
    try:
        with _controller_lock(run_dir):
            return _drive(run_dir, status, store)
    finally:
        store.close()
