"""Launch and bound a single fixed-size training attempt."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import uuid
from pathlib import Path

from trainguard.config import load_config
from trainguard.events import utc_now, write_json_atomic


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


def run(config_path: Path, output_root: Path) -> tuple[Path, bool]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    run_id = uuid.uuid4().hex[:12]
    attempt_id = "attempt-001"
    run_dir = (output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run.json"
    started_at = utc_now()
    status = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": "RUNNING",
        "config_fingerprint": config.fingerprint(),
        "started_at": started_at,
        "config": config.model_dump(),
    }
    write_json_atomic(status_path, status)

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
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--run-id",
        run_id,
        "--attempt-id",
        attempt_id,
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    result = "FAILED"
    reason = "launcher exited before completion"
    log_path = run_dir / "launcher.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=config.run.timeout_seconds)
            if exit_code == 0 and (run_dir / "summary.json").is_file():
                result = "SUCCEEDED"
                reason = "completed all training steps"
            else:
                reason = f"launcher exit code {exit_code}"
        except subprocess.TimeoutExpired:
            reason = f"training exceeded {config.run.timeout_seconds} seconds"
            _stop_process_group(process)
        except KeyboardInterrupt:
            reason = "interrupted by user"
            _stop_process_group(process)

    status.update(status=result, reason=reason, finished_at=utc_now())
    write_json_atomic(status_path, status)
    return run_dir, result == "SUCCEEDED"
