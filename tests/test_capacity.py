"""Real CPU boundaries for guarded local checkpoint capacity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.checkpoint import validate_checkpoint
from trainguard.config import load_config
from trainguard.events import append_event


def _guarded_config(tmp_path: Path, **limits: int) -> Path:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["run"]["profile"] = "guarded"
    raw["training"]["total_steps"] = 3
    raw["recovery"]["max_restarts"] = 0
    raw["checkpoint"].update(
        mode="sync", interval_steps=1, keep_last_k=2,
        max_checkpoint_bytes=10_000_000, max_retained_bytes=20_000_000,
        min_free_bytes=1_000_000, max_event_log_bytes=1_000_000,
    )
    raw["checkpoint"].update(limits)
    source = tmp_path / "guarded.json"
    source.write_text(json.dumps(raw))
    return source


def test_event_log_refuses_bytes_above_its_bound(tmp_path: Path) -> None:
    path = tmp_path / "rank-0.jsonl"
    append_event(path, run_id="run", event_type="first")
    original = path.read_bytes()
    with pytest.raises(OSError, match="event log byte budget"):
        append_event(path, max_bytes=len(original), run_id="run", event_type="second")
    assert path.read_bytes() == original


def test_guarded_run_keeps_two_valid_backups_within_declared_budget(tmp_path: Path) -> None:
    source = _guarded_config(tmp_path)
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert succeeded, (run_dir / "launcher.log").read_text()
    config = load_config(run_dir / "config.json")
    candidates = sorted((run_dir / "checkpoints").glob("*/COMMITTED"))
    assert len(candidates) >= 2
    for marker in candidates:
        record = validate_checkpoint(
            marker.parent, config, run_dir.name,
            decode_payload=True, require_trainable_state=True,
        )
        assert sum(item["size"] for item in record.manifest["files"]) <= (
            config.checkpoint.max_checkpoint_bytes
        )
    for path in (run_dir / "controller.jsonl", *sorted((run_dir / "attempts").glob("*/*.jsonl"))):
        assert path.stat().st_size <= config.checkpoint.max_event_log_bytes


def test_one_rank_low_space_rejects_before_candidate_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path)
    hook = tmp_path / "hook"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        "import os, shutil\n"
        "if os.environ.get('RANK') == '0':\n"
        "    original = shutil.disk_usage\n"
        "    def limited(path):\n"
        "        result = original(path)\n"
        "        return type(result)(result.total, result.used, 1)\n"
        "    shutil.disk_usage = limited\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(hook) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    assert "another rank rejected checkpoint capacity" in (run_dir / "launcher.log").read_text()
    assert not list((run_dir / "checkpoints").glob("step-*"))


def test_oversized_checkpoint_never_publishes_commit_marker(tmp_path: Path) -> None:
    source = _guarded_config(
        tmp_path, max_checkpoint_bytes=1, max_retained_bytes=2,
    )
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    assert "checkpoint exceeds its configured byte limit" in (
        run_dir / "launcher.log"
    ).read_text()
    assert list((run_dir / "checkpoints").glob("step-*"))
    assert not list((run_dir / "checkpoints").glob("*/COMMITTED"))
