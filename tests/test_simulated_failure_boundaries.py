"""Real two-rank probes for recovery and durable evidence boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import socket
import sqlite3
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch.multiprocessing as mp
from torch.distributed.checkpoint import FileSystemReader

from trainguard import controller
from trainguard.checkpoint import (
    CheckpointInvalid,
    latest_valid_checkpoint,
    ordered_candidates,
    validate_checkpoint,
)
from trainguard.config import load_config
from trainguard.events import write_json_atomic
from trainguard.run_store import RunStore
from trainguard.validation import validate_runs


def _configuration(root: Path, *, recover: bool) -> Path:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["model"]["dropout"] = 0.2
    raw["checkpoint"].update(mode="sync" if recover else "none", interval_steps=1)
    raw["recovery"].update(max_restarts=2, progress_timeout_seconds=30)
    if recover:
        raw["fault"].update(kind="worker_exit", step=3, rank=0)
    path = root / ("recovery-config.json" if recover else "reference-config.json")
    path.write_text(json.dumps(raw))
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _damage_dcp_reference(checkpoint: Path, damage: str) -> dict:
    metadata_path = checkpoint / "dcp/.metadata"
    manifest_path = checkpoint / "manifest.json"
    before_manifest_sha256 = _digest(manifest_path)
    metadata = FileSystemReader(checkpoint / "dcp").read_metadata()
    index, location = next(iter(metadata.storage_data.items()))
    old_location = {
        "relative_path": location.relative_path,
        "offset": location.offset,
        "length": location.length,
    }
    if damage == "missing_shard":
        changed = replace(location, relative_path="__missing_0.distcp")
    elif damage == "short_shard":
        shard_bytes = (checkpoint / "dcp" / location.relative_path).stat().st_size
        changed = replace(location, length=shard_bytes + 1)
    else:
        raise ValueError(damage)
    metadata.storage_data[index] = changed
    metadata_path.write_bytes(pickle.dumps(metadata))
    # Model a self-consistent publication, so ordinary hash failure cannot
    # explain why the controller must reject this candidate.
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["files"] if item["path"] == "dcp/.metadata")
    entry.update(size=metadata_path.stat().st_size, sha256=_digest(metadata_path))
    write_json_atomic(manifest_path, manifest)
    (checkpoint / "COMMITTED").write_text(_digest(manifest_path) + "\n")
    assert len(FileSystemReader(checkpoint / "dcp").read_metadata().storage_data) > 0
    return {
        "fault": damage,
        "checkpoint": str(checkpoint),
        "metadata_key": index.fqn,
        "storage_before": old_location,
        "storage_after": {
            "relative_path": changed.relative_path,
            "offset": changed.offset,
            "length": changed.length,
        },
        "manifest_sha256_before": before_manifest_sha256,
        "manifest_sha256_after": _digest(manifest_path),
        "metadata_sha256_after": _digest(metadata_path),
        "commit_marker_after": (checkpoint / "COMMITTED").read_text().strip(),
    }


@pytest.mark.parametrize("damage", ["missing_shard", "short_shard"])
def test_self_consistent_unloadable_newest_candidate_recovers_from_older_real_dcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    reference, reference_ok = controller.run(
        _configuration(tmp_path, recover=False), tmp_path / "reference-runs"
    )
    assert reference_ok, (reference / "launcher.log").read_text()

    original_launch = controller._launch_attempt

    class StopAfterFirstAttempt(BaseException):
        pass

    def stop_after_fault(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        if args[3] == "attempt-001":
            assert not result.succeeded
            raise StopAfterFirstAttempt
        return result

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_launch_attempt", stop_after_fault)
        with pytest.raises(StopAfterFirstAttempt):
            controller.run(
                _configuration(tmp_path, recover=True), tmp_path / "recovery-runs",
                allow_experiment=True,
            )

    recovered = next((tmp_path / "recovery-runs").iterdir())
    config = load_config(recovered / "config.json")
    run_id = json.loads((recovered / "run.json").read_text())["run_id"]
    newest, older = ordered_candidates(recovered)[:2]
    assert (newest.name, older.name) == (
        "step-000002-attempt-001",
        "step-000001-attempt-001",
    )
    assert validate_checkpoint(newest, config, run_id).global_step == 2
    fault = _damage_dcp_reference(newest, damage)
    fault.update(
        run_id=run_id,
        candidate_order_before_resume=[str(path) for path in ordered_candidates(recovered)],
        known_loadable_fallback=str(older),
    )
    write_json_atomic(recovered / "simulated-fault.json", fault)
    with pytest.raises(CheckpointInvalid, match="DCP metadata"):
        validate_checkpoint(newest, config, run_id)
    assert latest_valid_checkpoint(recovered, config, run_id).path == older
    succeeded = controller.resume(recovered)
    with sqlite3.connect(recovered / "run.sqlite3") as database:
        attempts = database.execute(
            "SELECT attempt_id, status, resume_checkpoint, resume_step "
            "FROM attempts ORDER BY number"
        ).fetchall()
        recoveries = database.execute(
            "SELECT to_attempt, checkpoint_path, resume_step FROM recoveries"
        ).fetchall()
    loaded = {}
    for rank in range(config.run.world_size):
        events = [
            json.loads(line)
            for line in (
                recovered / f"attempts/attempt-002/rank-{rank}.jsonl"
            ).read_text().splitlines()
        ]
        loaded[str(rank)] = {
            "state_loaded": [
                event for event in events if event["event_type"] == "state_loaded"
            ],
            "training_started": [
                event for event in events if event["event_type"] == "training_started"
            ],
        }
    comparison = validate_runs(reference, recovered) if succeeded else None
    write_json_atomic(
        recovered / "simulated-result.json",
        {
            "succeeded": succeeded,
            "candidate_order_after_resume": [
                str(path) for path in ordered_candidates(recovered)
            ],
            "attempts": [list(row) for row in attempts],
            "recoveries": [list(row) for row in recoveries],
            "loaded": loaded,
            "comparison": comparison,
        },
    )
    assert succeeded, (recovered / "launcher.log").read_text()
    assert len(attempts) == 2
    assert attempts[1][2:] == (str(older), 1)
    assert recoveries == [("attempt-002", str(older), 1)]
    for evidence in loaded.values():
        assert len(evidence["state_loaded"]) == 1
        assert evidence["state_loaded"][0]["global_step"] == 1
        assert len(evidence["training_started"]) == 1
        assert evidence["training_started"][0]["resumed_from"] == str(older)
    assert comparison is not None and comparison["passed"], comparison
    reference_summary = json.loads((reference / "summary.json").read_text())
    recovered_summary = json.loads((recovered / "summary.json").read_text())
    for field in (
        "model_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "scaler_sha256",
        "global_step",
        "consumed_batches",
    ):
        assert recovered_summary[field] == reference_summary[field]
    assert all(
        row["reference_sha256"] == row["recovered_sha256"]
        for row in comparison["effective_samples"].values()
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _durability_order_worker(rank: int, directory: str, port: int) -> None:
    from trainguard import checkpoint_io, trainer
    from trainguard.events import sync_event_file

    root = Path(directory)
    run_dir = root / "ordering-run"
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
    )

    def observe_sync(path: Path) -> None:
        sync_event_file(path)
        events = [json.loads(line) for line in path.read_text().splitlines()]
        phase = "completion" if events[-1]["event_type"] == "training_completed" else "checkpoint"
        write_json_atomic(
            root / f"{phase}-synced-rank-{rank}.json",
            {
                "rank": rank,
                "phase": phase,
                "event_tail": events[-1]["event_type"],
                "synced_at_ns": time.monotonic_ns(),
            },
        )

    checkpoint_io.sync_event_file = observe_sync
    trainer.sync_event_file = observe_sync
    if rank == 0:
        original_commit = checkpoint_io.commit_checkpoint

        def observe_commit(path, config, run_id, attempt_id, step):
            sync_markers = [root / f"checkpoint-synced-rank-{number}.json" for number in (0, 1)]
            if not all(marker.is_file() for marker in sync_markers):
                raise AssertionError("checkpoint publication preceded rank event sync")
            if (path / "COMMITTED").exists():
                raise AssertionError("commit marker existed before publication")
            record = original_commit(path, config, run_id, attempt_id, step)
            write_json_atomic(
                root / "publication-observed.json",
                {
                    "checkpoint": str(path),
                    "step": step,
                    "synced_ranks": [json.loads(marker.read_text()) for marker in sync_markers],
                    "commit_marker": (path / "COMMITTED").read_text().strip(),
                    "published_at_ns": time.monotonic_ns(),
                },
            )
            return record

        checkpoint_io.commit_checkpoint = observe_commit

    trainer.train(run_dir / "config.json", run_dir, "ordering", "attempt-001")
    completion_markers = [root / f"completion-synced-rank-{number}.json" for number in (0, 1)]
    write_json_atomic(
        root / f"returned-rank-{rank}.json",
        {
            "rank": rank,
            "all_completion_events_synced": all(
                marker.is_file() for marker in completion_markers
            ),
            "returned_at_ns": time.monotonic_ns(),
        },
    )


def test_all_rank_event_sync_precedes_commit_and_success_evidence(tmp_path: Path) -> None:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["training"]["total_steps"] = 1
    raw["checkpoint"].update(mode="sync", interval_steps=1)
    run_dir = tmp_path / "ordering-run"
    (run_dir / "attempts/attempt-001").mkdir(parents=True)
    write_json_atomic(run_dir / "config.json", raw)
    config = load_config(run_dir / "config.json")
    store = RunStore(run_dir / "run.sqlite3")
    try:
        store.create_run("ordering", config.fingerprint(), "now")
        store.start_attempt("ordering", "attempt-001", 1, None, 0)
    finally:
        store.close()

    mp.spawn(_durability_order_worker, args=(str(tmp_path), _free_port()), nprocs=2, join=True)

    publication = json.loads((tmp_path / "publication-observed.json").read_text())
    assert publication["step"] == 1
    assert publication["commit_marker"] == _digest(
        run_dir / "checkpoints/step-000001-attempt-001/manifest.json"
    )
    assert {item["rank"] for item in publication["synced_ranks"]} == {0, 1}
    assert {item["event_tail"] for item in publication["synced_ranks"]} == {
        "step_completed"
    }
    assert all(
        item["synced_at_ns"] < publication["published_at_ns"]
        for item in publication["synced_ranks"]
    )
    for rank in (0, 1):
        completed = json.loads((tmp_path / f"completion-synced-rank-{rank}.json").read_text())
        returned = json.loads((tmp_path / f"returned-rank-{rank}.json").read_text())
        assert completed["event_tail"] == "training_completed"
        assert completed["synced_at_ns"] < returned["returned_at_ns"]
        assert returned["all_completion_events_synced"]
    assert controller._valid_attempt_summary(run_dir, "attempt-001", config, "ordering")
