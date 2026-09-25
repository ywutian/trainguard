import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.config import load_config
from trainguard.controller import RunActiveError, resume, run
from trainguard.validation import validate_runs


def _config(tmp_path: Path, *, checkpoint: str, fault: str = "none", step: int | None = None) -> Path:
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    raw = load_config(source).model_dump()
    raw["model"]["dropout"] = 0.2
    raw["checkpoint"] = {"mode": checkpoint, "interval_steps": 1}
    raw["fault"] = {"kind": fault, "step": step, "rank": 0}
    raw["recovery"] = {"max_restarts": 2, "progress_timeout_seconds": 5}
    path = tmp_path / f"{checkpoint}-{fault}.json"
    path.write_text(json.dumps(raw))
    return path


def _summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def test_worker_exit_recovery_matches_reference(tmp_path: Path) -> None:
    reference, reference_ok = run(_config(tmp_path, checkpoint="none"), tmp_path / "runs")
    recovered, recovered_ok = run(
        _config(tmp_path, checkpoint="sync", fault="worker_exit", step=2),
        tmp_path / "runs",
    )
    assert reference_ok
    assert recovered_ok, (recovered / "launcher.log").read_text()
    for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256", "global_step"):
        assert _summary(reference)[field] == _summary(recovered)[field]
    validation = validate_runs(reference, recovered)
    assert validation["passed"], validation
    assert validation["comparison"]["tensor_atol"] == 0.0
    with sqlite3.connect(recovered / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2
        assert database.execute("SELECT COUNT(*) FROM recoveries").fetchone()[0] == 1


def test_corrupted_newest_checkpoint_uses_previous_step(tmp_path: Path) -> None:
    reference, reference_ok = run(_config(tmp_path, checkpoint="none"), tmp_path / "runs")
    recovered, recovered_ok = run(
        _config(tmp_path, checkpoint="sync", fault="corrupt", step=2),
        tmp_path / "runs",
    )
    assert reference_ok
    assert recovered_ok, (recovered / "launcher.log").read_text()
    assert _summary(reference)["model_sha256"] == _summary(recovered)["model_sha256"]
    with sqlite3.connect(recovered / "run.sqlite3") as database:
        step = database.execute("SELECT resume_step FROM attempts WHERE number=2").fetchone()[0]
    assert step == 1


def test_no_valid_checkpoint_fails_without_restart(tmp_path: Path) -> None:
    config = _config(tmp_path, checkpoint="sync", fault="worker_exit", step=1)
    raw = json.loads(config.read_text())
    raw["checkpoint"]["interval_steps"] = 2
    config.write_text(json.dumps(raw))
    run_dir, succeeded = run(config, tmp_path / "runs")
    assert not succeeded
    status = json.loads((run_dir / "run.json").read_text())
    assert "no valid checkpoint" in status["reason"]
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_async_save_interruption_and_stall_recover(tmp_path: Path) -> None:
    reference, reference_ok = run(_config(tmp_path, checkpoint="none"), tmp_path / "runs")
    assert reference_ok
    for mode, fault, step in (
        ("async", "worker_exit", 3),
        ("async", "corrupt", 2),
        ("sync", "save_interrupt", 2),
        ("sync", "hang", 2),
    ):
        recovered, succeeded = run(
            _config(tmp_path, checkpoint=mode, fault=fault, step=step),
            tmp_path / "runs",
        )
        assert succeeded, (recovered / "launcher.log").read_text()
        for field in ("model_sha256", "optimizer_sha256", "scheduler_sha256"):
            assert _summary(reference)[field] == _summary(recovered)[field]
        with sqlite3.connect(recovered / "run.sqlite3") as database:
            assert database.execute("SELECT COUNT(*) FROM recoveries").fetchone()[0] == 1


def test_validator_detects_omitted_recovery_state(tmp_path: Path) -> None:
    reference, reference_ok = run(_config(tmp_path, checkpoint="none"), tmp_path / "runs")
    assert reference_ok
    for omitted in ("rng", "optimizer", "cursor"):
        config = _config(tmp_path, checkpoint="sync", fault="worker_exit", step=2)
        raw = json.loads(config.read_text())
        raw["recovery"]["omit_state"] = omitted
        config.write_text(json.dumps(raw))
        recovered, recovered_ok = run(config, tmp_path / "runs")
        assert recovered_ok, (recovered / "launcher.log").read_text()
        validation = validate_runs(reference, recovered)
        assert not validation["passed"], omitted
        assert validation["differences"], omitted
        if omitted == "cursor":
            assert any("sample" in difference for difference in validation["differences"])


def test_explicit_resume_rejects_live_owner_then_recovers(tmp_path: Path, monkeypatch) -> None:
    original = controller._launch_attempt

    class SimulatedControllerExit(Exception):
        pass

    def exit_after_first(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == "attempt-001":
            raise SimulatedControllerExit
        return result

    monkeypatch.setattr(controller, "_launch_attempt", exit_after_first)
    with pytest.raises(SimulatedControllerExit):
        run(
            _config(tmp_path, checkpoint="sync", fault="worker_exit", step=2),
            tmp_path / "runs",
        )
    monkeypatch.setattr(controller, "_launch_attempt", original)
    run_dir = next((tmp_path / "runs").iterdir())
    run_id = json.loads((run_dir / "run.json").read_text())["run_id"]
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "--run-id", run_id],
        start_new_session=True,
    )
    try:
        identity = controller._pid_identity(process.pid)
        with sqlite3.connect(run_dir / "run.sqlite3") as database:
            database.execute(
                "UPDATE attempts SET pid=?, pid_identity=? WHERE attempt_id='attempt-001'",
                (process.pid, identity),
            )
        with pytest.raises(RunActiveError, match="owns a worker group"):
            resume(run_dir)
    finally:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    assert resume(run_dir), (run_dir / "launcher.log").read_text()
    assert json.loads((run_dir / "run.json").read_text())["status"] == "SUCCEEDED"
