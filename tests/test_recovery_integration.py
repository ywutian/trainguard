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
from trainguard.controller import RunActiveError, resume
from trainguard.controller import run as run_controller
from trainguard.run_store import RunStore
from trainguard.validation import validate_runs


def run(config: Path, output: Path):
    return run_controller(config, output, allow_experiment=True)


def _config(tmp_path: Path, *, checkpoint: str, fault: str = "none", step: int | None = None) -> Path:
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    raw = load_config(source).model_dump()
    raw["model"]["dropout"] = 0.2
    raw["checkpoint"] = {"mode": checkpoint, "interval_steps": 1}
    raw["fault"] = {"kind": fault, "step": step, "rank": 0}
    # Full-suite CPU load can make a cold two-rank launcher take over five seconds.
    raw["recovery"] = {"max_restarts": 2, "progress_timeout_seconds": 20}
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

    rank_log = recovered / "attempts" / "attempt-002" / "rank-0.jsonl"
    original = rank_log.read_text()
    rank_log.write_text(
        "".join(
            line for line in original.splitlines(keepends=True)
            if json.loads(line).get("event_type") != "state_loaded"
        )
    )
    missing_load = validate_runs(reference, recovered)
    assert not missing_load["passed"]
    assert any("state_loaded" in item for item in missing_load["differences"])
    rank_log.write_text(original)

    with sqlite3.connect(recovered / "run.sqlite3") as database:
        original_step = database.execute(
            "SELECT resume_step FROM recoveries WHERE to_attempt='attempt-002'"
        ).fetchone()[0]
        database.execute("UPDATE recoveries SET resume_step=0 WHERE to_attempt='attempt-002'")
    broken_decision = validate_runs(reference, recovered)
    assert not broken_decision["passed"]
    assert any("recovery decision" in item for item in broken_decision["differences"])
    with sqlite3.connect(recovered / "run.sqlite3") as database:
        database.execute(
            "UPDATE recoveries SET resume_step=? WHERE to_attempt='attempt-002'",
            (original_step,),
        )

    run_path = recovered / "run.json"
    status = json.loads(run_path.read_text())
    status["environment"]["source_sha256"] = "0" * 64
    run_path.write_text(json.dumps(status))
    wrong_environment = validate_runs(reference, recovered)
    assert not wrong_environment["passed"]
    assert any("run environment source_sha256 differs" in item for item in wrong_environment["differences"])
    status["environment"]["source_sha256"] = json.loads(
        (reference / "run.json").read_text()
    )["environment"]["source_sha256"]
    status["environment"]["installed_distributions"] = []
    run_path.write_text(json.dumps(status))
    wrong_dependencies = validate_runs(reference, recovered)
    assert not wrong_dependencies["passed"]
    assert any(
        "run environment installed_distributions differs" in item
        for item in wrong_dependencies["differences"]
    )
    del status["environment"]["installed_distributions"]
    run_path.write_text(json.dumps(status))
    missing_dependencies = validate_runs(reference, recovered)
    assert not missing_dependencies["passed"]
    assert any(
        "run environment dependency identity is invalid" in item
        for item in missing_dependencies["differences"]
    )


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


def test_selected_checkpoint_mutation_before_worker_load_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    original_scan = controller._scan_checkpoints
    modified = False

    def mutate_after_selection(run_dir, config, run_id, store, audit=False):
        nonlocal modified
        selected = original_scan(run_dir, config, run_id, store, audit)
        if selected is not None and not modified:
            sidecar = selected.path / "rank-0.json"
            state = json.loads(sidecar.read_text())
            state["scheduler"]["T_max"] += 1
            sidecar.write_text(json.dumps(state))
            modified = True
        return selected

    monkeypatch.setattr(controller, "_scan_checkpoints", mutate_after_selection)
    run_dir, succeeded = run(
        _config(tmp_path, checkpoint="sync", fault="worker_exit", step=2),
        tmp_path / "runs",
    )
    assert modified
    assert not succeeded
    assert not (run_dir / "summary.json").exists()
    assert "checkpoint changed before load" in (run_dir / "launcher.log").read_text()
    assert json.loads((run_dir / "run.json").read_text())["status"] == "FAILED"


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


def test_resume_after_run_metadata_before_sqlite_insert(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path, checkpoint="sync")
    with monkeypatch.context() as patch:
        patch.setattr(RunStore, "create_run", lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit(73)))
        with pytest.raises(SystemExit):
            run(config, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    assert json.loads((run_dir / "run.json").read_text())["attempt_id"] is None
    assert resume(run_dir), (run_dir / "launcher.log").read_text()
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    assert validate_runs(run_dir, run_dir)["passed"]


def test_missing_index_after_training_started_fails_closed(tmp_path: Path) -> None:
    run_dir, succeeded = run(_config(tmp_path, checkpoint="sync"), tmp_path / "runs")
    assert succeeded
    index_path = run_dir / "run.sqlite3"
    index_path.unlink()
    assert not resume(run_dir)
    assert not index_path.exists()
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["status"] == "FAILED"
    assert "run index is missing" in saved["reason"]


def test_async_save_interruption_and_stall_recover(tmp_path: Path) -> None:
    reference, reference_ok = run(_config(tmp_path, checkpoint="none"), tmp_path / "runs")
    assert reference_ok
    for mode, fault, step in (
        ("async", "worker_exit", 3),
        ("async", "corrupt", 2),
        ("async", "save_interrupt", 2),
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
    reference, reference_ok = run(
        _config(tmp_path, checkpoint="none"), tmp_path / "reference-runs"
    )
    assert reference_ok
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
    assert list((run_dir / "checkpoints").glob("*/COMMITTED"))
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
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
    assert validate_runs(reference, run_dir)["passed"]
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT resume_step FROM attempts WHERE number=2").fetchone()[0] == 1


@pytest.mark.parametrize("exit_at", ["attempt_return", "status_publication"])
def test_resume_reconciles_real_completion_before_controller_publication(
    tmp_path: Path, monkeypatch, exit_at: str
) -> None:
    class SimulatedControllerExit(Exception):
        pass

    raw = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml").model_dump()
    raw["recovery"]["max_restarts"] = 0
    config = tmp_path / "completion.json"
    config.write_text(json.dumps(raw))
    original_launch = controller._launch_attempt
    original_status = controller._set_status

    def exit_after_completion(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        assert result.succeeded
        raise SimulatedControllerExit

    def exit_before_status(run_dir, status, value, reason):
        if value == "SUCCEEDED":
            raise SimulatedControllerExit
        return original_status(run_dir, status, value, reason)

    with monkeypatch.context() as patch:
        if exit_at == "attempt_return":
            patch.setattr(controller, "_launch_attempt", exit_after_completion)
        else:
            patch.setattr(controller, "_set_status", exit_before_status)
        with pytest.raises(SimulatedControllerExit):
            run(config, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "attempts/attempt-001/summary.json").is_file()
    status = json.loads((run_dir / "run.json").read_text())
    if exit_at == "attempt_return":
        assert status["status"] == "RUNNING"
    else:
        assert status["status"] == "FINALIZING"
        assert status["post_run_audit"]["status"] == "PENDING"
    assert resume(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        assert database.execute("SELECT status FROM attempts").fetchone()[0] == "SUCCEEDED"
    assert json.loads((run_dir / "run.json").read_text())["status"] == "SUCCEEDED"


def test_resume_reconciles_completed_attempt_at_retry_limit(tmp_path: Path) -> None:
    raw = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml").model_dump()
    raw["recovery"]["max_restarts"] = 0
    config = tmp_path / "no-restarts.json"
    config.write_text(json.dumps(raw))
    run_dir, succeeded = run(config, tmp_path / "runs")
    assert succeeded
    status = json.loads((run_dir / "run.json").read_text())
    status["status"] = "RUNNING"
    (run_dir / "run.json").write_text(json.dumps(status))
    (run_dir / "summary.json").unlink()
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        database.execute("UPDATE runs SET status='RUNNING'")
    assert resume(run_dir)
    assert (run_dir / "summary.json").is_file()
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_resume_reuses_unrecorded_attempt_directory(tmp_path: Path, monkeypatch) -> None:
    original = controller._drive

    class SimulatedControllerExit(Exception):
        pass

    def exit_after_directory(run_dir, status, store, config):
        (run_dir / "attempts" / "attempt-001").mkdir(parents=True)
        raise SimulatedControllerExit

    monkeypatch.setattr(controller, "_drive", exit_after_directory)
    with pytest.raises(SimulatedControllerExit):
        run(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml", tmp_path / "runs")
    monkeypatch.setattr(controller, "_drive", original)
    run_dir = next((tmp_path / "runs").iterdir())
    assert resume(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_resume_reuses_recorded_attempt_before_workers_launch(tmp_path: Path, monkeypatch) -> None:
    original = controller._launch_attempt

    class SimulatedControllerExit(Exception):
        pass

    def exit_before_launch(*args, **kwargs):
        raise SimulatedControllerExit

    monkeypatch.setattr(controller, "_launch_attempt", exit_before_launch)
    with pytest.raises(SimulatedControllerExit):
        run(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml", tmp_path / "runs")
    monkeypatch.setattr(controller, "_launch_attempt", original)
    run_dir = next((tmp_path / "runs").iterdir())
    assert resume(run_dir)
    with sqlite3.connect(run_dir / "run.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


@pytest.mark.parametrize("owner_kind", ["launcher", "orphan"])
def test_resume_rejects_unrecorded_live_owner(
    tmp_path: Path, monkeypatch, owner_kind: str
) -> None:
    original = controller._launch_attempt
    launcher = None

    class SimulatedControllerExit(Exception):
        pass

    def exit_after_spawn(run_dir, config, run_id, attempt_id, selected, store):
        nonlocal launcher
        if owner_kind == "launcher":
            command = [
                sys.executable, "-c", "import time; time.sleep(30)",
                "-m", "torch.distributed.run", "--run-dir", str(run_dir),
                "--run-id", run_id, "--attempt-id", attempt_id,
            ]
        else:
            script = (
                "import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',"
                "'trainguard.trainer','--run-dir',sys.argv[1],"
                "'--run-id',sys.argv[2],'--attempt-id',sys.argv[3]])"
            )
            command = [sys.executable, "-c", script, str(run_dir), run_id, attempt_id]
        launcher = subprocess.Popen(command, start_new_session=True)
        if owner_kind == "orphan":
            assert launcher.wait(timeout=5) == 0
        raise SimulatedControllerExit

    monkeypatch.setattr(controller, "_launch_attempt", exit_after_spawn)
    try:
        with pytest.raises(SimulatedControllerExit):
            run(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml", tmp_path / "runs")
        monkeypatch.setattr(controller, "_launch_attempt", original)
        run_dir = next((tmp_path / "runs").iterdir())
        with pytest.raises(RunActiveError, match="owns a worker group"):
            resume(run_dir)
    finally:
        monkeypatch.setattr(controller, "_launch_attempt", original)
        if launcher is not None:
            controller._stop_process_group(
                launcher, run_dir, run_dir.name, "attempt-001"
            )

    assert resume(run_dir)


def test_detached_worker_from_older_attempt_blocks_resume_but_not_other_run(
    tmp_path: Path, monkeypatch
) -> None:
    source = _config(tmp_path, checkpoint="none")
    original_launch = controller._launch_attempt
    worker_pid_path = tmp_path / "detached-worker.pid"
    launcher_pid = None
    run_dir = None

    class SimulatedControllerExit(Exception):
        pass

    def exit_after_detached_worker(run_dir, config, run_id, attempt_id, selected, store):
        nonlocal launcher_pid
        script = (
            "import subprocess,sys; from pathlib import Path; "
            "worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)',"
            "'trainguard.trainer','--run-dir',sys.argv[1],"
            "'--run-id',sys.argv[2],'--attempt-id',sys.argv[3]],"
            "start_new_session=True); "
            "Path(sys.argv[4]).write_text(str(worker.pid))"
        )
        launcher = subprocess.Popen(
            [
                sys.executable, "-c", script, str(run_dir), run_id, attempt_id,
                str(worker_pid_path),
            ],
            start_new_session=True,
        )
        launcher_pid = launcher.pid
        assert launcher.wait(timeout=5) == 0
        raise SimulatedControllerExit

    monkeypatch.setattr(controller, "_launch_attempt", exit_after_detached_worker)
    try:
        with pytest.raises(SimulatedControllerExit):
            run(source, tmp_path / "interrupted-runs")
        monkeypatch.setattr(controller, "_launch_attempt", original_launch)
        run_dir = next((tmp_path / "interrupted-runs").iterdir())
        run_id = json.loads((run_dir / "run.json").read_text())["run_id"]
        worker_pid = int(worker_pid_path.read_text())
        assert launcher_pid is not None
        assert os.getpgid(worker_pid) != launcher_pid
        assert not controller._owned_group_members(run_dir, run_id, "attempt-001", launcher_pid)

        # An older attempt must remain fenced even after a newer attempt was recorded.
        store = RunStore(run_dir / "run.sqlite3")
        try:
            store.finish_attempt("attempt-001", "FAILED", 74, "controller exited")
            store.start_attempt(run_id, "attempt-002", 2, None, 0)
            store.finish_attempt("attempt-002", "FAILED", 74, "controller exited")
        finally:
            store.close()
        with pytest.raises(RunActiveError, match="attempt attempt-001 still owns"):
            resume(run_dir)
        assert not (run_dir / "attempts/attempt-001/worker-group-ended.json").exists()

        other_run, other_ok = run(source, tmp_path / "independent-runs")
        assert other_ok, (other_run / "launcher.log").read_text()
        assert worker_pid in controller._owned_group_members(run_dir, run_id, "attempt-001")
        assert not controller._owned_group_members(other_run, other_run.name, "attempt-001")
    finally:
        monkeypatch.setattr(controller, "_launch_attempt", original_launch)
        if worker_pid_path.exists() and run_dir is not None:
            worker_pid = int(worker_pid_path.read_text())
            if worker_pid in controller._owned_group_members(run_dir, run_dir.name, "attempt-001"):
                os.kill(worker_pid, signal.SIGTERM)
