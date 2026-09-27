import hashlib
import io
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, MetadataIndex

from trainguard import controller, events
from trainguard.checkpoint import (
    candidate_path,
    capture_rank_state,
    commit_checkpoint,
    validate_checkpoint,
)
from trainguard.config import ProjectConfig, load_config
from trainguard.restore_failures import (
    record_group_ended,
    record_restore_incomplete,
    record_restore_progress,
)
from trainguard.run_store import RunStore


def candidate(root, config, step, *, optimizer=True):
    path = candidate_path(root, "attempt-001", step)
    (path / "dcp").mkdir(parents=True)
    payloads = []
    for rank in range(config.run.world_size):
        stream = io.BytesIO()
        torch.save(torch.tensor([rank, step]), stream)
        data = stream.getvalue()
        payloads.append(data)
        (path / f"dcp/__{rank}_0.distcp").write_bytes(data)
    metadata_entries = {"model.placeholder": BytesStorageMetadata()}
    storage_data = {
        MetadataIndex("model.placeholder"): SimpleNamespace(
            relative_path="__0_0.distcp", offset=0, length=len(payloads[0])
        )
    }
    if optimizer:
        metadata_entries["optimizer.placeholder"] = BytesStorageMetadata()
        storage_data[MetadataIndex("optimizer.placeholder")] = SimpleNamespace(
            relative_path="__1_0.distcp", offset=0, length=len(payloads[1])
        )
    metadata = Metadata(metadata_entries, storage_data=storage_data)
    (path / "dcp/.metadata").write_bytes(pickle.dumps(metadata))
    for rank in range(config.run.world_size):
        events.write_json_atomic(
            path / f"rank-{rank}.json",
            capture_rank_state(
                config, "run", "attempt-001", rank, step,
                {"T_max": config.training.total_steps, "last_epoch": step,
                 "_step_count": step + 1, "base_lrs": [0.001], "eta_min": 0.0,
                 "_last_lr": [0.001 * (1 + math.cos(
                     math.pi * step / config.training.total_steps)) / 2]},
            ),
        )
    commit_checkpoint(path, config, "run", "attempt-001", step)
    return path


def settings(keep=None):
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["training"]["total_steps"] = 100
    if keep is not None:
        raw["checkpoint"]["keep_last_k"] = keep
    return ProjectConfig.model_validate(raw)


def test_selection_validates_newest_only(tmp_path, monkeypatch):
    config = settings()
    for step in range(1, 101):
        candidate(tmp_path, config, step)
    calls = []
    original = controller.validate_checkpoint

    def track(path, *args, **kwargs):
        calls.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(controller, "validate_checkpoint", track)
    store = RunStore(tmp_path / "run.sqlite3")
    store.create_run("run", config.fingerprint(), "now")
    try:
        selected = controller._scan_checkpoints(tmp_path, config, "run", store)
        assert selected.global_step == 100
        assert len(calls) == 1
        (selected.path / "dcp/__0_0.distcp").write_bytes(b"bad")
        calls.clear()
        assert controller._scan_checkpoints(tmp_path, config, "run", store).global_step == 99
        assert len(calls) == 2
    finally:
        store.close()


def test_retention_is_retryable_and_protects_loading(tmp_path, monkeypatch):
    from trainguard import lifecycle

    config = settings(2)
    paths = [candidate(tmp_path, config, step) for step in range(1, 5)]
    pending = candidate_path(tmp_path, "attempt-001", 5)
    pending.mkdir()
    original = lifecycle.shutil.rmtree

    def interrupted(path):
        (path / "rank-0.json").unlink()
        raise OSError("simulated deletion interruption")

    monkeypatch.setattr(lifecycle.shutil, "rmtree", interrupted)
    with pytest.raises(OSError):
        lifecycle.prune_checkpoints(tmp_path, config, "run", protected={paths[1]})
    journal = json.loads((tmp_path / "retention.json").read_text())
    assert journal["pending"]
    monkeypatch.setattr(lifecycle.shutil, "rmtree", original)
    lifecycle.prune_checkpoints(tmp_path, config, "run", protected={paths[1]})
    assert not paths[0].exists()
    assert all(path.exists() for path in paths[1:])
    assert pending.exists()
    assert not json.loads((tmp_path / "retention.json").read_text())["pending"]


def test_retention_does_not_count_model_only_candidate_as_fallback(tmp_path):
    from trainguard import lifecycle

    config = settings(2)
    valid = [candidate(tmp_path, config, step) for step in range(1, 4)]
    invalid = candidate(tmp_path, config, 4, optimizer=False)
    lifecycle.prune_checkpoints(tmp_path, config, "run")
    assert not valid[0].exists()
    assert valid[1].exists() and valid[2].exists()
    assert invalid.exists()


def test_retention_preserves_two_loadable_backups_after_two_restore_failures(tmp_path):
    from trainguard import lifecycle

    raw = settings(2).model_dump()
    raw["run"]["world_size"] = 1
    config = ProjectConfig.model_validate(raw)
    paths = []
    for step in range(1, 6):
        path = candidate_path(tmp_path, "attempt-001", step)
        dcp.save(
            {
                "model": {"weight": torch.full((2, 2), float(step))},
                "optimizer": {"slot": torch.full((2, 2), float(step))},
            },
            checkpoint_id=path / "dcp",
        )
        scheduler = {
            "T_max": config.training.total_steps,
            "last_epoch": step,
            "_step_count": step + 1,
            "base_lrs": [0.001],
            "eta_min": 0.0,
            "_last_lr": [0.001 * (1 + math.cos(
                math.pi * step / config.training.total_steps
            )) / 2],
        }
        events.write_json_atomic(
            path / "rank-0.json",
            capture_rank_state(config, "run", "attempt-001", 0, step, scheduler),
        )
        commit_checkpoint(path, config, "run", "attempt-001", step)
        paths.append(path)

    for path, damage in [(paths[3], "wrong_key"), (paths[4], "wrong_shape")]:
        metadata_path = path / "dcp/.metadata"
        metadata = FileSystemReader(path / "dcp").read_metadata()
        name = next(key for key in metadata.state_dict_metadata if key.startswith("model."))
        if damage == "wrong_key":
            changed = f"model.unsupported_{name.removeprefix('model.')}"
            metadata.state_dict_metadata[changed] = metadata.state_dict_metadata.pop(name)
            metadata.planner_data[changed] = metadata.planner_data.pop(name)
            metadata.storage_data = {
                replace(index, fqn=changed) if index.fqn == name else index: location
                for index, location in metadata.storage_data.items()
            }
        else:
            original = metadata.state_dict_metadata[name]
            metadata.state_dict_metadata[name] = replace(
                original, size=torch.Size([original.size[0] + 1, *original.size[1:]])
            )
        metadata_path.write_bytes(pickle.dumps(metadata))
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = next(item for item in manifest["files"] if item["path"] == "dcp/.metadata")
        entry.update(
            size=metadata_path.stat().st_size,
            sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        )
        events.write_json_atomic(manifest_path, manifest)
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (path / "COMMITTED").write_text(manifest_sha256 + "\n")
        assert validate_checkpoint(
            path, config, "run", decode_payload=True, require_trainable_state=True
        ).manifest_sha256 == manifest_sha256

    store = RunStore(tmp_path / "run.sqlite3")
    store.create_run("run", config.fingerprint(), "now")
    store.start_attempt("run", "attempt-001", 1, None, 0)
    for number, path in [(2, paths[3]), (3, paths[4])]:
        attempt_id = f"attempt-{number:03d}"
        store.start_attempt("run", attempt_id, number, str(path), number + 2)
        events.write_json_atomic(
            tmp_path / "attempts" / attempt_id / "rank-0-restore-failure.json",
            {
                "schema_version": 1,
                "run_id": "run",
                "attempt_id": attempt_id,
                "rank": 0,
                "checkpoint_path": str(path),
                "manifest_sha256": hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest(),
                "error_type": "CheckpointException",
            },
        )
    assert controller._scan_checkpoints(tmp_path, config, "run", store).path == paths[2]
    store.close()

    lifecycle.prune_checkpoints(tmp_path, config, "run", protected={paths[4]})
    assert paths[4].exists()
    assert not paths[3].exists()
    lifecycle.prune_checkpoints(tmp_path, config, "run")
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists()
    assert not paths[3].exists() and not paths[4].exists()
    assert all(
        (tmp_path / "attempts" / f"attempt-{number:03d}" / "rank-0-restore-failure.json").exists()
        for number in (2, 3)
    )
    assert validate_checkpoint(
        paths[1], config, "run", decode_payload=True, require_trainable_state=True
    ).global_step == 2
    assert validate_checkpoint(
        paths[2], config, "run", decode_payload=True, require_trainable_state=True
    ).global_step == 3
    for step, path in [(2, paths[1]), (3, paths[2])]:
        state = {
            "model": {"weight": torch.zeros((2, 2))},
            "optimizer": {"slot": torch.zeros((2, 2))},
        }
        dcp.load(state, checkpoint_id=path / "dcp")
        assert torch.equal(state["model"]["weight"], torch.full((2, 2), float(step)))
        assert torch.equal(state["optimizer"]["slot"], torch.full((2, 2), float(step)))


def test_incomplete_restore_is_excluded_but_retained_for_diagnosis(tmp_path):
    from trainguard import lifecycle

    config = settings(2)
    paths = [candidate(tmp_path, config, step) for step in range(1, 5)]
    digest = hashlib.sha256((paths[3] / "manifest.json").read_bytes()).hexdigest()
    store = RunStore(tmp_path / "run.sqlite3")
    store.create_run("run", config.fingerprint(), "now")
    store.start_attempt("run", "attempt-001", 1, None, 0)
    store.start_attempt("run", "attempt-002", 2, str(paths[3]), 4)
    for rank in range(2):
        record_restore_progress(
            tmp_path, "run", "attempt-002", rank, paths[3], digest, "restore_started"
        )
    record_group_ended(
        tmp_path, "run", "attempt-002", "launcher exit code 74", "controller_cleanup"
    )
    assert record_restore_incomplete(
        tmp_path, "run", "attempt-002", paths[3], 2, digest
    )
    assert controller._scan_checkpoints(tmp_path, config, "run", store).path == paths[2]
    store.close()

    lifecycle.prune_checkpoints(tmp_path, config, "run")
    assert not paths[0].exists()
    assert all(path.exists() for path in paths[1:])


def test_atomic_publication_syncs_parent_directory(tmp_path, monkeypatch):
    calls = []
    original = events.os.fsync

    def sync(fd):
        import stat

        calls.append(stat.S_ISDIR(events.os.fstat(fd).st_mode))
        original(fd)

    monkeypatch.setattr(events.os, "fsync", sync)
    events.write_json_atomic(tmp_path / "state.json", {"state": 1})
    assert calls == [False, True]


def test_deletion_intent_survives_loss_of_fallbacks(tmp_path, monkeypatch):
    from trainguard import lifecycle

    config = settings(2)
    paths = [candidate(tmp_path, config, step) for step in range(1, 4)]
    original = lifecycle.shutil.rmtree

    def interrupted(path):
        (path / "rank-0.json").unlink()
        raise OSError("interrupted")

    monkeypatch.setattr(lifecycle.shutil, "rmtree", interrupted)
    with pytest.raises(OSError):
        lifecycle.prune_checkpoints(tmp_path, config, "run")
    (paths[1] / "COMMITTED").unlink()
    monkeypatch.setattr(lifecycle.shutil, "rmtree", original)
    lifecycle.prune_checkpoints(tmp_path, config, "run")
    assert paths[0].exists()
    assert json.loads((tmp_path / "retention.json").read_text())["pending"]


def test_retention_refuses_linked_checkpoint_root(tmp_path):
    from trainguard import lifecycle

    config = settings(2)
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint = candidate(outside, config, 1)
    marker = outside / "preserve.txt"
    marker.write_text("keep")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoints").symlink_to(outside / "checkpoints", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        lifecycle.prune_checkpoints(run_dir, config, "run")
    assert checkpoint.exists()
    assert marker.read_text() == "keep"
    assert not (run_dir / "retention.json").exists()
