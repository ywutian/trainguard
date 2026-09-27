"""Real CPU boundaries for guarded local checkpoint capacity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.checkpoint import candidate_path, validate_checkpoint
from trainguard.config import load_config
from trainguard.events import append_event


def _guarded_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **limits: int
) -> Path:
    key_path = tmp_path / "customer-sample.key"
    key_path.write_bytes(b"a-private-customer-key-with-at-least-32-bytes")
    key_path.chmod(0o600)
    monkeypatch.setenv("TRAINGUARD_SAMPLE_HMAC_KEY_FILE", str(key_path))
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


def test_guarded_run_keeps_two_valid_backups_within_declared_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
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
        assert sum(
            file.stat().st_size for file in marker.parent.rglob("*") if file.is_file()
        ) <= config.checkpoint.max_checkpoint_bytes
    for path in (run_dir / "controller.jsonl", *sorted((run_dir / "attempts").glob("*/*.jsonl"))):
        assert path.stat().st_size <= config.checkpoint.max_event_log_bytes


def test_one_rank_low_space_rejects_before_candidate_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
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
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, (str(hook), os.environ.get("PYTHONPATH"))))
    )
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    assert "another rank rejected checkpoint capacity" in (run_dir / "launcher.log").read_text()
    assert not list((run_dir / "checkpoints").glob("step-*"))


def test_oversized_checkpoint_never_publishes_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(
        tmp_path, monkeypatch, max_checkpoint_bytes=1, max_retained_bytes=2,
    )
    run_dir, succeeded = controller.run(source, tmp_path / "runs")
    assert not succeeded
    assert "checkpoint exceeds its configured byte limit" in (
        run_dir / "launcher.log"
    ).read_text()
    assert list((run_dir / "checkpoints").glob("step-*"))
    assert not list((run_dir / "checkpoints").glob("*/COMMITTED"))


def test_guarded_checkpoint_limit_includes_manifest_and_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    reference, succeeded = controller.run(source, tmp_path / "reference")
    assert succeeded
    candidate = next((reference / "checkpoints").glob("*/COMMITTED")).parent
    config = load_config(reference / "config.json")
    record = validate_checkpoint(candidate, config, reference.name)
    payload_bytes = sum(item["size"] for item in record.manifest["files"])
    complete_bytes = sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
    assert complete_bytes > payload_bytes
    limit = payload_bytes + (complete_bytes - payload_bytes) // 2
    source = _guarded_config(
        tmp_path, monkeypatch, max_checkpoint_bytes=limit, max_retained_bytes=2 * limit,
    )
    run_dir, succeeded = controller.run(source, tmp_path / "bounded")
    assert not succeeded
    assert "checkpoint exceeds its configured byte limit" in (run_dir / "launcher.log").read_text()
    assert not list((run_dir / "checkpoints").glob("*/COMMITTED"))


def test_guarded_config_needs_two_checkpoint_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    raw = json.loads(source.read_text())
    raw["training"]["total_steps"] = 1
    raw["checkpoint"]["interval_steps"] = 10
    source.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="two distinct checkpoint boundaries"):
        load_config(source)


def test_guarded_completion_rejects_only_one_verified_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(tmp_path, monkeypatch)
    original_scan = controller._scan_checkpoints
    damaged_marker = None

    def damage_older_at_audit(run_dir, *args, **kwargs):
        nonlocal damaged_marker
        if kwargs.get("audit") and damaged_marker is None:
            marker = candidate_path(run_dir, "attempt-001", 2) / "COMMITTED"
            damaged_marker = marker.read_bytes()
            marker.write_bytes(b"invalid backup\n")
        return original_scan(run_dir, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_scan_checkpoints", damage_older_at_audit)
        with pytest.raises(RuntimeError, match="fewer than two verified"):
            controller.run(source, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    status = json.loads((run_dir / "run.json").read_text())
    assert status["status"] == "FINALIZING"
    assert status["post_run_audit"]["status"] == "FAILED"
    assert status["post_run_audit"]["valid_retained_count"] == 1
    assert damaged_marker is not None
    (candidate_path(run_dir, "attempt-001", 2) / "COMMITTED").write_bytes(damaged_marker)
    assert controller.resume(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["post_run_audit"][
        "valid_retained_count"
    ] >= 2


def test_guarded_audit_counts_uncommitted_candidate_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _guarded_config(
        tmp_path, monkeypatch, max_checkpoint_bytes=2_000_000,
        max_retained_bytes=4_000_000,
    )
    original_scan = controller._scan_checkpoints

    def add_uncommitted_at_audit(run_dir, *args, **kwargs):
        if kwargs.get("audit"):
            candidate = candidate_path(run_dir, "attempt-999", 999)
            if not candidate.exists():
                candidate.mkdir()
                (candidate / "unfinished.bin").write_bytes(b"x" * 2_000_000)
        return original_scan(run_dir, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_scan_checkpoints", add_uncommitted_at_audit)
        with pytest.raises(RuntimeError, match="capacity budget is unsatisfied"):
            controller.run(source, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    audit = json.loads((run_dir / "run.json").read_text())["post_run_audit"]
    assert audit["status"] == "FAILED"
    assert audit["valid_retained_count"] >= 2
    assert audit["unverified_candidate_count"] >= 1
    assert audit["unverified_candidate_bytes"] >= 2_000_000
    assert audit["checkpoint_file_bytes"] > 4_000_000
    assert audit["checkpoint_file_budget_satisfied"] is False
