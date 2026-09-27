"""Bounded single-node controller for fixed-size process-group recovery."""

from __future__ import annotations

import fcntl
import json
import os
import re
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

from trainguard.checkpoint import (
    CheckpointInvalid,
    CheckpointRecord,
    ordered_candidates,
    validate_checkpoint,
)
from trainguard.config import ProjectConfig, load_config
from trainguard.environment import environment_snapshot
from trainguard.events import append_event, utc_now, write_json_atomic
from trainguard.lifecycle import prune_checkpoints
from trainguard.records import parse_event
from trainguard.restore_failures import (
    failed_restore_candidates,
    record_group_ended,
    record_restore_incomplete,
)
from trainguard.run_store import RunStore
from trainguard.strategy import preflight
from trainguard.validation import completion_errors


class RunActiveError(RuntimeError):
    """A controller or owned worker process still runs for this directory."""


class ExperimentNotAuthorizedError(ValueError):
    """Fault injection or omitted recovery state needs explicit authorization."""


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
            re.search(r"(?:^|\s)(?:trainguard.trainer|torch.distributed.run)(?:\s|$)", command)
            and re.search(r"--run-dir " + re.escape(str(run_dir)) + r"(?=\s--|$)", command)
            and re.search(r"--run-id " + re.escape(run_id) + r"(?=\s|$)", command)
            and re.search(r"--attempt-id " + re.escape(attempt_id) + r"(?=\s|$)", command)
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
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(run_dir / ".controller.lock", flags, 0o600), "a+") as lock:
        os.fchmod(lock.fileno(), 0o600)
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
        # The launcher may still be between fork and exec. A matching process
        # identity is enough to block a second owner until its exit is known.
        return True
    return bool(_owned_group_members(run_dir, run_id, attempt_id))


def _assert_no_owned_workers(run_dir: Path, run_id: str, attempts: list) -> None:
    for attempt in attempts:
        if _owned_process_alive(
            attempt["pid"], attempt["pid_identity"], run_id, attempt["attempt_id"], run_dir
        ):
            raise RunActiveError(
                f"attempt {attempt['attempt_id']} still owns a worker group or process"
            )


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
    run_dir: Path, config: ProjectConfig, run_id: str, store: RunStore, audit: bool = False
) -> CheckpointRecord | None:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return None
    failed_restores = failed_restore_candidates(
        run_dir, run_id, config.run.world_size, store.attempts(run_id)
    )
    records = []
    selected = None
    for path in ordered_candidates(run_dir):
        try:
            record = validate_checkpoint(
                path, config, run_id, decode_payload=True, require_trainable_state=True
            )
        except (CheckpointInvalid, OSError) as exc:
            records.append((str(path), run_id, None, None, "INVALID", str(exc)))
        else:
            if record.manifest_sha256 in failed_restores.explicit.get(path, set()):
                records.append((
                    str(path), run_id, record.attempt_id, record.global_step,
                    "INVALID", "worker restore failed for this manifest",
                ))
            elif record.manifest_sha256 in failed_restores.incomplete.get(path, set()):
                records.append((
                    str(path), run_id, record.attempt_id, record.global_step,
                    "INVALID", "worker restore incomplete for this manifest",
                ))
            else:
                records.append(
                    (str(path), run_id, record.attempt_id, record.global_step, "VALID", None)
                )
                if selected is None:
                    selected = record
                if not audit:
                    break
    store.record_checkpoints(records)
    return selected


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
        "--expected-config-fingerprint",
        config.fingerprint(),
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--attempt-id",
        attempt_id,
    ]
    if resume_checkpoint is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", resume_checkpoint.manifest_sha256):
            raise ValueError("selected checkpoint manifest digest is invalid")
        command.extend([
            "--resume-checkpoint", str(resume_checkpoint.path),
            "--expected-checkpoint-sha256", resume_checkpoint.manifest_sha256,
        ])
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    offsets: dict[int, int] = {}
    steps: dict[int, int] = {}
    completed: set[int] = set()
    started = time.monotonic()
    last_progress = started
    reason = "launcher exited before completion"
    exit_code = None
    commits_seen: set[str] = set()

    def milestone(event_type: str, **fields) -> None:
        append_event(
            run_dir / "controller.jsonl",
            run_id=run_id,
            attempt_id=attempt_id,
            event_type=event_type,
            **fields,
        )

    milestone("launch_requested", resumed=resume_checkpoint is not None)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(run_dir / "launcher.log", flags, 0o600), "ab") as log:
        os.fchmod(log.fileno(), 0o600)
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
                if config.checkpoint.keep_last_k is not None:
                    commits = {
                        str(path.parent) for path in (run_dir / "checkpoints").glob("*/COMMITTED")
                    }
                    if commits != commits_seen:
                        prune_checkpoints(
                            run_dir,
                            config,
                            run_id,
                            protected={resume_checkpoint.path} if resume_checkpoint else set(),
                        )
                        commits_seen = commits
                exit_code = process.poll()
                if exit_code is not None:
                    if exit_code != 0:
                        milestone("fault_observed", reason=f"launcher exit {exit_code}")
                    break
                if time.monotonic() - started > config.run.timeout_seconds:
                    reason = f"attempt exceeded {config.run.timeout_seconds} seconds"
                    milestone("fault_observed", reason=reason)
                    _stop_process_group(process, run_dir, run_id, attempt_id)
                    exit_code = process.poll()
                    break
                if not steps and (
                    time.monotonic() - started > config.recovery.startup_timeout_seconds
                ):
                    reason = "worker startup or first update stalled"
                    milestone("fault_observed", reason=reason)
                    _stop_process_group(process, run_dir, run_id, attempt_id)
                    exit_code = process.poll()
                    break
                if steps and (
                    time.monotonic() - last_progress > config.recovery.progress_timeout_seconds
                ):
                    reason = "step progress stalled"
                    milestone("fault_observed", reason=reason)
                    _stop_process_group(process, run_dir, run_id, attempt_id)
                    exit_code = process.poll()
                    break
                time.sleep(0.2)
            orphaned_workers = bool(_owned_group_members(run_dir, run_id, attempt_id))
            if orphaned_workers:
                reason = "launcher exited while owned workers remained"
            _read_events(
                run_dir, attempt_id, run_id, config.run.world_size, offsets, steps, completed
            )
            summary = _valid_attempt_summary(run_dir, attempt_id, config, run_id)
            if exit_code == 0 and not orphaned_workers and summary is not None:
                reason = "completed all training steps"
                return AttemptResult(
                    True, reason, 0, max(steps.values(), default=0)
                )
            if reason == "launcher exited before completion":
                reason = f"launcher exit code {exit_code}; completion evidence missing or invalid"
        except KeyboardInterrupt:
            reason = "interrupted by user"
            milestone("fault_observed", reason=reason)
        except Exception as exc:  # noqa: BLE001 - cleanup covers every controller failure
            reason = f"controller {type(exc).__name__}: {exc}"
            milestone("fault_observed", reason=reason)
            write_json_atomic(
                run_dir / "attempts" / attempt_id / "controller-error.json",
                {"reason": reason, "time": utc_now()},
            )
        finally:
            try:
                _stop_process_group(process, run_dir, run_id, attempt_id)
                if process.poll() is None or _owned_group_members(run_dir, run_id, attempt_id):
                    raise RunActiveError(
                        f"attempt {attempt_id} still owns a worker process"
                    )
                record_group_ended(
                    run_dir, run_id, attempt_id, reason, "controller_cleanup"
                )
                milestone("group_stopped")
            except BaseException as cleanup:
                raise RunActiveError(f"{reason}; worker cleanup failed: {cleanup}") from cleanup
    return AttemptResult(False, reason, process.poll(), max(steps.values(), default=0))


def _set_status(run_dir: Path, status: dict, value: str, reason: str) -> None:
    previous = status.get("status")
    status.update(status=value, reason=reason)
    if value in {"SUCCEEDED", "FAILED", "INTERRUPTED"}:
        if previous != value or "finished_at" not in status:
            status["finished_at"] = utc_now()
        started = status.pop("execution_started_monotonic", None)
        if started is not None:
            status["measurement"] = {
                "elapsed_seconds": time.monotonic() - started,
                "load_average_before": status.pop("execution_load_before"),
                "load_average_after": list(os.getloadavg()),
                "method": "controller_monotonic",
            }
    else:
        status.pop("finished_at", None)
    write_json_atomic(run_dir / "run.json", status)


def _drive(run_dir: Path, status: dict, store: RunStore, config: ProjectConfig) -> bool:
    run_id = status["run_id"]
    attempts = store.attempts(run_id)
    _assert_no_owned_workers(run_dir, run_id, attempts)
    if attempts:
        last = attempts[-1]
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
                if last["resume_checkpoint"] is not None:
                    record_group_ended(
                        run_dir, run_id, last["attempt_id"],
                        "previous owner exited after worker group ended", "no_owned_workers",
                    )
                    record_restore_incomplete(
                        run_dir, run_id, last["attempt_id"],
                        Path(last["resume_checkpoint"]), config.run.world_size,
                    )
                store.finish_attempt(last["attempt_id"], "INTERRUPTED", None, "controller exited")
            attempts = store.attempts(run_id)

    while True:
        _assert_no_owned_workers(run_dir, run_id, attempts)
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

        if selected is not None:
            append_event(
                run_dir / "controller.jsonl",
                run_id=run_id,
                attempt_id=f"attempt-{number:03d}",
                event_type="checkpoint_selected",
                checkpoint_path=str(selected.path),
                manifest_sha256=selected.manifest_sha256,
                global_step=selected.global_step,
            )
        prune_checkpoints(run_dir, config, run_id, protected={selected.path} if selected else set())
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
            json.loads((selected.path / "rank-0.json").read_text())["consumed_batches"]
            if selected
            else 0,
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
        if not result.succeeded and selected is not None:
            record_restore_incomplete(
                run_dir, run_id, attempt_id, selected.path, config.run.world_size,
                selected.manifest_sha256,
            )
        store.finish_attempt(
            attempt_id,
            "SUCCEEDED" if result.succeeded else "FAILED",
            result.exit_code,
            result.reason,
        )
        if result.succeeded:
            _scan_checkpoints(run_dir, config, run_id, store)
            prune_checkpoints(run_dir, config, run_id)
            summary = _valid_attempt_summary(run_dir, attempt_id, config, run_id)
            assert summary is not None
            write_json_atomic(run_dir / "summary.json", summary)
            store.set_run_status(run_id, "SUCCEEDED")
            _set_status(run_dir, status, "SUCCEEDED", result.reason)
            return True
        if result.reason.startswith("controller "):
            store.set_run_status(run_id, "FAILED")
            _set_status(run_dir, status, "FAILED", result.reason)
            return False
        if result.reason == "interrupted by user":
            store.set_run_status(run_id, "INTERRUPTED")
            _set_status(run_dir, status, "INTERRUPTED", result.reason)
            return False
        attempts = store.attempts(run_id)


def run(
    config_path: Path, output_root: Path, *, allow_experiment: bool = False
) -> tuple[Path, bool]:
    execution_started = time.monotonic()
    execution_load_before = list(os.getloadavg())
    config = load_config(config_path.resolve())
    if (config.fault.kind != "none" or config.recovery.omit_state != "none") and not allow_experiment:
        raise ExperimentNotAuthorizedError(
            "fault injection or omitted recovery state requires explicit experiment authorization"
        )
    workload_source = preflight(config)
    run_id = uuid.uuid4().hex[:12]
    run_dir = (output_root / run_id).resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    if workload_source is not None:
        from trainguard.external_workload import freeze_source, read_verified_source

        frozen = freeze_source(run_dir, workload_source)
        read_verified_source(config, path=frozen)
    write_json_atomic(run_dir / "config.json", config.model_dump())
    started_at = utc_now()
    status = {
        "run_id": run_id,
        "attempt_id": None,
        "status": "RUNNING",
        "config_fingerprint": config.fingerprint(),
        "experiment_authorized": allow_experiment,
        "started_at": started_at,
        "config": config.model_dump(),
        "run_schema_version": 2,
        "environment": environment_snapshot(config.run.world_size, config.run.device, run_dir),
        "execution_started_monotonic": execution_started,
        "execution_load_before": execution_load_before,
    }
    write_json_atomic(run_dir / "run.json", status)
    store = RunStore(run_dir / "run.sqlite3")
    try:
        (run_dir / "run.sqlite3").chmod(0o600)
        store.create_run(run_id, config.fingerprint(), started_at)
        with _controller_lock(run_dir):
            succeeded = _drive(run_dir, status, store, config)
    finally:
        store.close()
    return run_dir, succeeded


def resume(run_dir: Path) -> bool:
    run_dir = run_dir.resolve()
    status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = load_config(run_dir / "config.json")
    if status.get("run_schema_version") != 2:
        raise ValueError("run schema is unsupported; use its original source and runtime")
    if (
        config.fault.kind != "none" or config.recovery.omit_state != "none"
    ) and status.get("experiment_authorized") is not True:
        raise ExperimentNotAuthorizedError("saved experiment authorization is missing")
    current = environment_snapshot(config.run.world_size, config.run.device, run_dir)
    for field in ("source_sha256", "python", "torch", "versions"):
        if current[field] != status["environment"].get(field):
            raise ValueError(f"saved run source or runtime {field} differs")
    if config.external_workload is not None:
        from trainguard.external_workload import frozen_workload_path

        preflight(config, workload_source=frozen_workload_path(run_dir))
    else:
        preflight(config)
    if config.fingerprint() != status["config_fingerprint"]:
        raise ValueError("saved config fingerprint differs from run metadata")
    store = RunStore(run_dir / "run.sqlite3")
    try:
        with _controller_lock(run_dir):
            identity = store.run_identity(status["run_id"])
            if identity is None:
                # The controller can exit after publishing run.json but before inserting
                # the first SQLite row. Rebuild only that empty, never-launched state.
                attempts_root = run_dir / "attempts"
                checkpoints_root = run_dir / "checkpoints"
                pristine = (
                    status.get("attempt_id") is None
                    and status.get("status") == "RUNNING"
                    and not (run_dir / "controller.jsonl").exists()
                    and not (run_dir / "summary.json").exists()
                    and (not attempts_root.exists() or not any(attempts_root.iterdir()))
                    and (not checkpoints_root.exists() or not any(checkpoints_root.iterdir()))
                )
                if not pristine:
                    raise ValueError("run index is missing after training may have started")
                store.create_run(
                    status["run_id"], status["config_fingerprint"], status["started_at"]
                )
            elif (
                identity["config_fingerprint"] != status["config_fingerprint"]
                or identity["started_at"] != status["started_at"]
            ):
                raise ValueError("run index identity differs from run metadata")
            _assert_no_owned_workers(run_dir, status["run_id"], store.attempts(status["run_id"]))
            if status["status"] != "SUCCEEDED":
                # A different controller cannot reconstruct the original wall-time window.
                status.pop("execution_started_monotonic", None)
                status.pop("execution_load_before", None)
                status.pop("measurement", None)
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
            return _drive(run_dir, status, store, config)
    finally:
        store.close()
