import json
import os
import stat
from pathlib import Path

from trainguard.controller import run
from trainguard.events import append_event, write_json_atomic


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_run_artifacts_are_private_with_a_permissive_umask(tmp_path: Path) -> None:
    config = Path(__file__).parents[1] / "configs/cpu_demo.yaml"
    previous = os.umask(0o022)
    try:
        run_dir, succeeded = run(config, tmp_path / "runs")
    finally:
        os.umask(previous)

    assert succeeded
    assert _mode(run_dir) == 0o700
    for path in (
        run_dir / "config.json",
        run_dir / "run.json",
        run_dir / "run.sqlite3",
        run_dir / ".controller.lock",
        run_dir / "launcher.log",
        run_dir / "attempts/attempt-001/rank-0.jsonl",
    ):
        assert _mode(path) == 0o600, path


def test_atomic_json_ignores_planted_temporary_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("preserve this file", encoding="utf-8")
    output = tmp_path / "report.json"
    planted = tmp_path / f".report.json.{os.getpid()}.tmp"
    planted.symlink_to(target)

    write_json_atomic(output, {"result": "safe"})

    assert target.read_text(encoding="utf-8") == "preserve this file"
    assert planted.is_symlink()
    assert not output.is_symlink()
    assert json.loads(output.read_text(encoding="utf-8")) == {"result": "safe"}
    assert _mode(output) == 0o600


def test_event_log_is_private_and_rejects_symlinks(tmp_path: Path) -> None:
    path = tmp_path / "rank-0.jsonl"
    append_event(path, event_type="step_completed", global_step=1)
    assert _mode(path) == 0o600
    target = tmp_path / "target.txt"
    target.write_text("preserve this file", encoding="utf-8")
    path.unlink()
    path.symlink_to(target)

    try:
        append_event(path, event_type="step_completed", global_step=2)
    except OSError:
        pass
    else:
        raise AssertionError("event append followed a symbolic link")
    assert target.read_text(encoding="utf-8") == "preserve this file"
