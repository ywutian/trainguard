"""Process-crash cuts through the local checkpoint publication transaction."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from trainguard import checkpoint, events
from trainguard.checkpoint import (
    candidate_path,
    capture_rank_state,
    commit_checkpoint,
    latest_valid_checkpoint,
)
from trainguard.config import ProjectConfig, load_config
from trainguard.run_store import RunStore

CUTS = {
    "during_payload_fsync": (81, 1),
    "after_rank_sidecar": (82, 1),
    "during_payload_scan": (83, 1),
    "after_manifest": (84, 1),
    "after_marker_replace": (85, 2),
    "after_marker_durable": (86, 2),
    "after_commit_before_index": (87, 2),
}


def _config() -> ProjectConfig:
    raw = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml").model_dump()
    raw["run"]["world_size"] = 1
    return ProjectConfig.model_validate(raw)


def _write_candidate(
    run_dir: Path, config: ProjectConfig, attempt_id: str, step: int
) -> Path:
    path = candidate_path(run_dir, attempt_id, step)
    # The real DCP writer produces a loadable metadata/storage map and syncs its files.
    dcp.save(
        {"model": {"weight": torch.full((4,), float(step))}},
        checkpoint_id=path / "dcp",
    )
    events.write_json_atomic(
        path / "rank-0.json",
        capture_rank_state(config, "run", attempt_id, 0, step, {}),
    )
    return path


def _child(config_path: Path, run_dir: Path, cut: str) -> None:
    config = load_config(config_path)
    path = candidate_path(run_dir, "attempt-002", 2)
    code = CUTS[cut][0]

    if cut == "during_payload_fsync":
        original_fsync = os.fsync

        def crash_on_payload_fsync(descriptor: int) -> None:
            identity = os.fstat(descriptor)
            is_payload = any(
                (entry.stat().st_dev, entry.stat().st_ino)
                == (identity.st_dev, identity.st_ino)
                for entry in (path / "dcp").glob("*.distcp")
            )
            original_fsync(descriptor)
            if is_payload:
                os._exit(code)

        os.fsync = crash_on_payload_fsync

    if cut == "after_rank_sidecar":
        original_sync_directory = events.sync_directory

        def crash_after_sidecar_sync(directory: Path) -> None:
            original_sync_directory(directory)
            if directory == path and (path / "rank-0.json").is_file():
                os._exit(code)

        events.sync_directory = crash_after_sidecar_sync

    _write_candidate(run_dir, config, "attempt-002", 2)

    if cut == "during_payload_scan":
        original_file_record = checkpoint._file_record

        def crash_after_first_payload_read(root: Path, entry: Path):
            result = original_file_record(root, entry)
            if entry.suffix == ".distcp":
                os._exit(code)
            return result

        checkpoint._file_record = crash_after_first_payload_read

    if cut == "after_manifest":
        original_write_json_atomic = checkpoint.write_json_atomic

        def crash_after_manifest_write(destination: Path, value: dict) -> None:
            original_write_json_atomic(destination, value)
            if destination == path / "manifest.json":
                os._exit(code)

        checkpoint.write_json_atomic = crash_after_manifest_write

    if cut == "after_marker_replace":
        original_sync_directory = checkpoint.sync_directory

        def crash_before_marker_dir_sync(directory: Path) -> None:
            if directory == path and (path / "COMMITTED").is_file():
                os._exit(code)
            original_sync_directory(directory)

        checkpoint.sync_directory = crash_before_marker_dir_sync

    if cut == "after_marker_durable":
        original_write_marker_atomic = checkpoint._write_marker_atomic

        def crash_after_marker_sync(marker: Path, value: str) -> None:
            original_write_marker_atomic(marker, value)
            os._exit(code)

        checkpoint._write_marker_atomic = crash_after_marker_sync

    commit_checkpoint(path, config, "run", "attempt-002", 2)
    if cut == "after_commit_before_index":
        # The filesystem transaction is authoritative even if the index is stale.
        os._exit(code)
    os._exit(99)


@pytest.mark.parametrize("cut", list(CUTS))
def test_process_exit_at_checkpoint_publication_boundary(tmp_path: Path, cut: str) -> None:
    config = _config()
    config_path = tmp_path / "config.json"
    events.write_json_atomic(config_path, config.model_dump())

    older = _write_candidate(tmp_path, config, "attempt-001", 1)
    commit_checkpoint(older, config, "run", "attempt-001", 1)
    assert latest_valid_checkpoint(tmp_path, config, "run").path == older

    store = RunStore(tmp_path / "run.sqlite3")
    try:
        store.create_run("run", config.fingerprint(), "now")
        store.record_checkpoint(str(older), "run", "attempt-001", 1, "VALID", None)
    finally:
        store.close()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint-child",
        str(config_path),
        str(tmp_path),
        cut,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    expected_exit, expected_step = CUTS[cut]
    assert result.returncode == expected_exit, result.stdout + result.stderr

    selected = latest_valid_checkpoint(tmp_path, config, "run")
    assert selected is not None
    assert selected.global_step == expected_step
    assert selected.path == candidate_path(
        tmp_path, "attempt-001" if expected_step == 1 else "attempt-002", expected_step
    )
    restored = {"model": {"weight": torch.zeros(4)}}
    dcp.load(restored, checkpoint_id=selected.path / "dcp")
    torch.testing.assert_close(restored["model"]["weight"], torch.full((4,), float(expected_step)))
    # No child-side SQLite write occurred, even when the newer filesystem
    # transaction became eligible immediately before the process exited.
    with sqlite3.connect(tmp_path / "run.sqlite3") as database:
        indexed = database.execute(
            "SELECT global_step FROM checkpoints ORDER BY global_step"
        ).fetchall()
    assert indexed == [(1,)]


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--checkpoint-child":
    _child(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
