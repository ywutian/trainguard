import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.checkpoint import candidate_path
from trainguard.config import load_config
from trainguard.run_store import RunStore
from trainguard.support import build_support_bundle


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
    assert json.loads((directory / "run.json").read_text())["post_run_audit"]["status"] == (
        "NOT_APPLICABLE"
    )
    (directory / "attempts/attempt-001/rank-1.jsonl").unlink()
    assert not controller.resume(directory)
    assert json.loads((directory / "run.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("failure", ["scan", "prune"])
def test_post_run_audit_failure_cannot_publish_success_or_repeat_training(
    tmp_path, monkeypatch, failure
):
    settings = config().model_dump()
    settings["checkpoint"].update(mode="sync", interval_steps=1, keep_last_k=2)
    source = tmp_path / "guarded-config.json"
    source.write_text(json.dumps(settings))
    function_name = "_scan_checkpoints" if failure == "scan" else "prune_checkpoints"
    original = getattr(controller, function_name)

    def fail_after_training(run_dir, *args, **kwargs):
        if json.loads((run_dir / "run.json").read_text())["status"] == "FINALIZING":
            raise RuntimeError(f"injected {failure} failure")
        return original(run_dir, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, function_name, fail_after_training)
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            controller.run(source, tmp_path / "runs")
        run_dir = next((tmp_path / "runs").iterdir())
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            controller.resume(run_dir)
        status = json.loads((run_dir / "run.json").read_text())
        assert status["status"] == "FINALIZING"
        assert status["post_run_audit"]["status"] == "FAILED"
        assert build_support_bundle(run_dir)["status"] == "FINALIZING"
        with sqlite3.connect(run_dir / "run.sqlite3") as database:
            assert database.execute("SELECT status FROM runs").fetchone()[0] == "FINALIZING"
            assert database.execute("SELECT status FROM attempts").fetchall() == [
                ("SUCCEEDED",)
            ]

    assert controller.resume(run_dir)
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "SUCCEEDED"
    assert status["post_run_audit"]["status"] == "PASSED"
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1

    with monkeypatch.context() as patch:
        patch.setattr(controller, function_name, fail_after_training)
        with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
            controller.resume(run_dir)
        assert json.loads((run_dir / "run.json").read_text())["status"] == "FINALIZING"
        assert build_support_bundle(run_dir)["status"] == "FINALIZING"
    assert controller.resume(run_dir)


def test_unsatisfied_retention_budget_cannot_pass_post_run_audit(tmp_path):
    settings = config().model_dump()
    settings["checkpoint"].update(
        mode="sync", interval_steps=1, keep_last_k=2, max_retained_bytes=1
    )
    source = tmp_path / "tight-budget.json"
    source.write_text(json.dumps(settings))
    with pytest.raises(RuntimeError, match="capacity budget is unsatisfied"):
        controller.run(source, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FINALIZING"
    assert status["post_run_audit"]["status"] == "FAILED"
    assert status["post_run_audit"]["budget_satisfied"] is False
    with pytest.raises(RuntimeError, match="capacity budget is unsatisfied"):
        controller.resume(run_dir)
    resumed_status = json.loads((run_dir / "run.json").read_text())
    assert resumed_status["status"] == "FINALIZING"
    assert resumed_status["post_run_audit"]["status"] == "FAILED"
    assert resumed_status["post_run_audit"]["budget_satisfied"] is False
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT status FROM attempts").fetchall() == [("SUCCEEDED",)]


def test_older_checkpoint_cannot_certify_completed_attempt(tmp_path, monkeypatch):
    settings = config().model_dump()
    settings["training"]["total_steps"] = 4
    settings["checkpoint"].update(mode="sync", interval_steps=1, keep_last_k=2)
    source = tmp_path / "final-checkpoint-config.json"
    source.write_text(json.dumps(settings))
    original = controller._scan_checkpoints
    final_marker = None

    def damage_final_at_audit(run_dir, *args, **kwargs):
        nonlocal final_marker
        if kwargs.get("audit") and final_marker is None:
            final = candidate_path(run_dir, "attempt-001", 4)
            final_marker = (final / "COMMITTED").read_bytes()
            (final / "COMMITTED").write_bytes(b"invalid final checkpoint\n")
        return original(run_dir, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_scan_checkpoints", damage_final_at_audit)
        with pytest.raises(RuntimeError, match="no valid final checkpoint"):
            controller.run(source, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FINALIZING"
    assert status["post_run_audit"]["status"] == "FAILED"
    assert final_marker is not None
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        checkpoints = {
            Path(path).name: state for path, state in database.execute(
                "SELECT path, status FROM checkpoints"
            )
        }
        assert checkpoints["step-000004-attempt-001"] == "INVALID"
        assert checkpoints["step-000003-attempt-001"] == "VALID"
    with pytest.raises(RuntimeError, match="no valid final checkpoint"):
        controller.resume(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["status"] == "FINALIZING"

    (candidate_path(run_dir, "attempt-001", 4) / "COMMITTED").write_bytes(final_marker)
    assert controller.resume(run_dir)
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "SUCCEEDED"
    assert status["post_run_audit"]["status"] == "PASSED"
    assert status["post_run_audit"]["final_checkpoint"]["global_step"] == 4


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
