import json
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
