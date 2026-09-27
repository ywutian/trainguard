import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.config import load_config
from trainguard.run_store import RunStore


def config():
    return load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml")


def test_incomplete_summary_cannot_publish_success(tmp_path):
    settings = config()
    path = tmp_path / "attempts/attempt-001/summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "run_id": "run",
                "attempt_id": "attempt-001",
                "config_fingerprint": settings.fingerprint(),
                "global_step": settings.training.total_steps,
                "world_size": settings.run.world_size,
            }
        )
    )
    assert controller._valid_attempt_summary(tmp_path, "attempt-001", settings, "run") is None


def test_succeeded_run_is_reaudited(tmp_path):
    source = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    directory, ok = controller.run(source, tmp_path)
    assert ok
    (directory / "attempts/attempt-001/rank-1.jsonl").unlink()
    assert not controller.resume(directory)
    assert json.loads((directory / "run.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("failure", ["pid", "events"])
def test_exception_after_spawn_cleans_actual_process(tmp_path, monkeypatch, failure):
    settings = config()
    (tmp_path / "attempts/attempt-001").mkdir(parents=True)
    (tmp_path / "config.json").write_text(json.dumps(settings.model_dump()))
    store = RunStore(tmp_path / "run.sqlite3")
    store.create_run("run", settings.fingerprint(), "now")
    store.start_attempt("run", "attempt-001", 1, None, 0)
    spawned = []
    original = controller.subprocess.Popen

    def track(*args, **kwargs):
        child = original(*args, **kwargs)
        if "torch.distributed.run" in args[0]:
            spawned.append(child)
        return child

    def fail(*args, **kwargs):
        raise RuntimeError("injected audit failure")

    monkeypatch.setattr(controller.subprocess, "Popen", track)
    if failure == "pid":
        monkeypatch.setattr(store, "set_pid", fail)
    else:
        monkeypatch.setattr(controller, "_read_events", fail)
    try:
        result = controller._launch_attempt(tmp_path, settings, "run", "attempt-001", None, store)
        assert not result.succeeded
        assert "injected audit failure" in result.reason
        assert spawned[0].poll() is not None
        assert not controller._owned_group_members(tmp_path, "run", "attempt-001", spawned[0].pid)
    finally:
        for child in spawned:
            controller._stop_process_group(child, tmp_path, "run", "attempt-001")
        store.close()


def test_detached_worker_prevents_group_ended_evidence(tmp_path, monkeypatch):
    settings = config()
    (tmp_path / "attempts/attempt-001").mkdir(parents=True)
    (tmp_path / "config.json").write_text(json.dumps(settings.model_dump()))
    store = RunStore(tmp_path / "run.sqlite3")
    store.create_run("run", settings.fingerprint(), "now")
    store.start_attempt("run", "attempt-001", 1, None, 0)
    worker_pid_path = tmp_path / "detached-worker.pid"
    launcher_pid = None
    real_popen = subprocess.Popen
    script = (
        "import subprocess,sys; from pathlib import Path; "
        "worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)',"
        "'trainguard.trainer','--run-dir',sys.argv[1],"
        "'--run-id',sys.argv[2],'--attempt-id',sys.argv[3]],"
        "start_new_session=True); "
        "Path(sys.argv[4]).write_text(str(worker.pid))"
    )

    def detached_launcher(command, *args, **kwargs):
        nonlocal launcher_pid
        if "torch.distributed.run" in command:
            command = [
                sys.executable, "-c", script, str(tmp_path), "run", "attempt-001",
                str(worker_pid_path),
            ]
        process = real_popen(command, *args, **kwargs)
        launcher_pid = process.pid
        return process

    monkeypatch.setattr(controller.subprocess, "Popen", detached_launcher)
    try:
        with pytest.raises(controller.RunActiveError, match="worker cleanup failed"):
            controller._launch_attempt(tmp_path, settings, "run", "attempt-001", None, store)
        worker_pid = int(worker_pid_path.read_text())
        assert launcher_pid is not None
        assert os.getpgid(worker_pid) != launcher_pid
        assert not controller._owned_group_members(tmp_path, "run", "attempt-001", launcher_pid)
        assert controller._owned_group_members(tmp_path, "run", "attempt-001") == [worker_pid]
        assert not (tmp_path / "attempts/attempt-001/worker-group-ended.json").exists()
    finally:
        if worker_pid_path.exists():
            worker_pid = int(worker_pid_path.read_text())
            if worker_pid in controller._owned_group_members(tmp_path, "run", "attempt-001"):
                os.kill(worker_pid, signal.SIGTERM)
        store.close()


def test_controller_failure_diagnostic_is_not_replaced_by_retry_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        controller,
        "_launch_attempt",
        lambda *a: controller.AttemptResult(
            False, "controller ValueError: invalid current event", 1, 0
        ),
    )
    source = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    directory, ok = controller.run(source, tmp_path)
    assert not ok
    assert (
        json.loads((directory / "run.json").read_text())["reason"]
        == "controller ValueError: invalid current event"
    )
