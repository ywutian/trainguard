import hashlib
import io
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, MetadataIndex

from trainguard.checkpoint import (
    CheckpointInvalid,
    candidate_path,
    capture_rank_state,
    commit_checkpoint,
    latest_valid_checkpoint,
    restore_rng,
    validate_checkpoint,
)
from trainguard.config import load_config
from trainguard.events import write_json_atomic


def _candidate(root: Path, step: int, config=None) -> tuple[Path, object]:
    config = config or load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    path = candidate_path(root, "attempt-001", step)
    (path / "dcp").mkdir(parents=True)
    payloads = []
    for rank in range(config.run.world_size):
        stream = io.BytesIO()
        torch.save(torch.tensor([rank, step]), stream)
        data = stream.getvalue()
        payloads.append(data)
        (path / "dcp" / f"__{rank}_0.distcp").write_bytes(data)
    metadata = Metadata(
        {"model.placeholder": BytesStorageMetadata()},
        storage_data={
            MetadataIndex("model.placeholder"): SimpleNamespace(
                relative_path="__0_0.distcp", offset=0, length=len(payloads[0])
            )
        },
    )
    (path / "dcp" / ".metadata").write_bytes(pickle.dumps(metadata))
    for rank in range(config.run.world_size):
        state = capture_rank_state(config, "run-one", "attempt-001", rank, step, {})
        write_json_atomic(path / f"rank-{rank}.json", state)
    return path, config


def test_committed_checkpoint_rejects_corruption_and_falls_back(tmp_path: Path) -> None:
    older, config = _candidate(tmp_path, 1)
    commit_checkpoint(older, config, "run-one", "attempt-001", 1)
    newer, _ = _candidate(tmp_path, 2)
    commit_checkpoint(newer, config, "run-one", "attempt-001", 2)
    assert latest_valid_checkpoint(tmp_path, config, "run-one").path == newer
    (newer / "dcp" / "__0_0.distcp").write_bytes(b"corrupt")
    with pytest.raises(CheckpointInvalid, match="hash"):
        validate_checkpoint(newer, config, "run-one")
    assert latest_valid_checkpoint(tmp_path, config, "run-one").path == older


def test_checkpoint_requires_all_rank_state_and_commit_marker(tmp_path: Path) -> None:
    path, config = _candidate(tmp_path, 1)
    (path / "rank-1.json").unlink()
    with pytest.raises(CheckpointInvalid, match="rank"):
        commit_checkpoint(path, config, "run-one", "attempt-001", 1)
    assert latest_valid_checkpoint(tmp_path, config, "run-one") is None
    path, _ = _candidate(tmp_path / "second", 1)
    commit_checkpoint(path, config, "run-one", "attempt-001", 1)
    (path / "COMMITTED").unlink()
    with pytest.raises(CheckpointInvalid, match="commit"):
        validate_checkpoint(path, config, "run-one")


def test_checkpoint_rejects_mixed_steps_and_incompatible_config(tmp_path: Path) -> None:
    path, config = _candidate(tmp_path / "second", 1)
    state_path = path / "rank-1.json"
    state = json.loads(state_path.read_text())
    state["global_step"] = 2
    write_json_atomic(state_path, state)
    with pytest.raises(CheckpointInvalid, match="step"):
        commit_checkpoint(path, config, "run-one", "attempt-001", 1)

    path, config = _candidate(tmp_path, 1)
    commit_checkpoint(path, config, "run-one", "attempt-001", 1)
    raw = config.model_dump()
    raw["run"]["seed"] += 1
    incompatible = type(config).model_validate(raw)
    with pytest.raises(CheckpointInvalid, match="fingerprint"):
        validate_checkpoint(path, incompatible, "run-one")


def test_malformed_newest_rank_state_falls_back(tmp_path: Path) -> None:
    older, config = _candidate(tmp_path, 1)
    commit_checkpoint(older, config, "run-one", "attempt-001", 1)
    newer, _ = _candidate(tmp_path, 2)
    commit_checkpoint(newer, config, "run-one", "attempt-001", 2)
    (newer / "rank-0.json").write_text("[]")
    assert latest_valid_checkpoint(tmp_path, config, "run-one").path == older


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    path, config = _candidate(tmp_path, 1)
    commit_checkpoint(path, config, "run-one", "attempt-001", 1)
    (path / "manifest.json").write_text("[]")
    (path / "COMMITTED").write_text(hashlib.sha256(b"[]").hexdigest() + "\n")
    with pytest.raises(CheckpointInvalid, match="manifest"):
        validate_checkpoint(path, config, "run-one")


def test_unreadable_newest_commit_marker_falls_back(tmp_path: Path) -> None:
    older, config = _candidate(tmp_path, 1)
    commit_checkpoint(older, config, "run-one", "attempt-001", 1)
    newer, _ = _candidate(tmp_path, 2)
    commit_checkpoint(newer, config, "run-one", "attempt-001", 2)
    (newer / "COMMITTED").write_bytes(b"\xff")
    assert latest_valid_checkpoint(tmp_path, config, "run-one").path == older


def test_commit_requires_dcp_output_from_each_rank(tmp_path: Path) -> None:
    path, config = _candidate(tmp_path, 1)
    (path / "dcp" / "__1_0.distcp").unlink()
    (path / "dcp" / "__0_1.distcp").write_bytes(b"extra rank zero file")
    with pytest.raises(CheckpointInvalid, match="DCP rank"):
        commit_checkpoint(path, config, "run-one", "attempt-001", 1)


def test_rng_state_round_trip() -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    random.seed(38)
    np.random.seed(38)
    torch.manual_seed(38)
    state = capture_rank_state(config, "run-one", "attempt-001", 0, 1, {})
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng(state)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected


def test_checkpoint_rejects_different_rank_consumed_boundaries(tmp_path):
    path, config = _candidate(tmp_path, 1)
    state_path = path / 'rank-1.json'
    state = json.loads(state_path.read_text())
    state.update(consumed_batches=2, next_data_step=2)
    write_json_atomic(state_path, state)
    with pytest.raises(CheckpointInvalid, match='consumed'):
        commit_checkpoint(path, config, 'run-one', 'attempt-001', 1)


def test_checkpoint_rejects_different_source_identity(tmp_path):
    path, config = _candidate(tmp_path, 1)
    commit_checkpoint(path, config, 'run-one', 'attempt-001', 1)
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_sha256'] = '0' * 64
    write_json_atomic(manifest_path, manifest)
    (path / 'COMMITTED').write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + '\n')
    with pytest.raises(CheckpointInvalid, match='source_sha256'):
        validate_checkpoint(path, config, 'run-one')


def test_checkpoint_selection_and_new_candidate_refuse_linked_root(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint, config = _candidate(outside, 1)
    commit_checkpoint(checkpoint, config, "run-one", "attempt-001", 1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoints").symlink_to(outside / "checkpoints", target_is_directory=True)
    with pytest.raises(CheckpointInvalid, match="symbolic link"):
        latest_valid_checkpoint(run_dir, config, "run-one")
    with pytest.raises(CheckpointInvalid, match="symbolic link"):
        candidate_path(run_dir, "attempt-002", 2)
    linked_path = run_dir / "checkpoints" / checkpoint.name
    with pytest.raises(CheckpointInvalid, match="symbolic link"):
        validate_checkpoint(linked_path, config, "run-one")
    with pytest.raises(CheckpointInvalid, match="symbolic link"):
        commit_checkpoint(linked_path, config, "run-one", "attempt-001", 1)


def test_commit_reads_payload_once_and_later_validation_rechecks_it(tmp_path, monkeypatch):
    from trainguard import checkpoint as checkpoint_module

    path, config = _candidate(tmp_path, 1)
    original = checkpoint_module._sha256
    payload_reads = []

    def measured(entry):
        if entry.suffix == ".distcp":
            payload_reads.append(entry)
        return original(entry)

    monkeypatch.setattr(checkpoint_module, "_sha256", measured)
    committed = commit_checkpoint(path, config, "run-one", "attempt-001", 1)
    assert committed.global_step == 1
    assert len(payload_reads) == config.run.world_size
    payload_reads.clear()
    assert validate_checkpoint(path, config, "run-one").global_step == 1
    assert len(payload_reads) == config.run.world_size
    (path / "dcp" / "__0_0.distcp").write_bytes(b"corrupt")
    with pytest.raises(CheckpointInvalid, match="hash"):
        validate_checkpoint(path, config, "run-one")


def test_payload_change_during_publication_cannot_report_success(tmp_path, monkeypatch):
    from trainguard import checkpoint as checkpoint_module

    path, config = _candidate(tmp_path, 1)
    original = checkpoint_module._write_marker_atomic

    def mutate_after_marker(marker_path, value):
        original(marker_path, value)
        payload = path / "dcp" / "__0_0.distcp"
        payload.write_bytes(b"change")

    monkeypatch.setattr(checkpoint_module, "_write_marker_atomic", mutate_after_marker)
    with pytest.raises(CheckpointInvalid, match="changed during publication"):
        commit_checkpoint(path, config, "run-one", "attempt-001", 1)
    with pytest.raises(CheckpointInvalid, match="hash"):
        validate_checkpoint(path, config, "run-one")


def _refresh_metadata_hash(path: Path) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = path / "dcp/.metadata"
    for entry in manifest["files"]:
        if entry["path"] == "dcp/.metadata":
            entry["size"] = metadata.stat().st_size
            entry["sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    write_json_atomic(manifest_path, manifest)
    (path / "COMMITTED").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n")


@pytest.mark.parametrize("damage", ["unreadable", "missing_shard", "short_shard"])
def test_self_consistent_but_unloadable_metadata_falls_back(tmp_path: Path, damage: str) -> None:
    older, config = _candidate(tmp_path, 1)
    commit_checkpoint(older, config, "run-one", "attempt-001", 1)
    newer, _ = _candidate(tmp_path, 2)
    commit_checkpoint(newer, config, "run-one", "attempt-001", 2)
    metadata_path = newer / "dcp/.metadata"
    if damage == "unreadable":
        metadata_path.write_bytes(b"not a DCP metadata pickle")
    else:
        metadata = pickle.loads(metadata_path.read_bytes())
        location = next(iter(metadata.storage_data.values()))
        if damage == "missing_shard":
            location.relative_path = "missing.distcp"
        else:
            location.length = 10_000
        metadata_path.write_bytes(pickle.dumps(metadata))
    _refresh_metadata_hash(newer)
    with pytest.raises(CheckpointInvalid, match="DCP metadata"):
        validate_checkpoint(newer, config, "run-one")
    assert latest_valid_checkpoint(tmp_path, config, "run-one").path == older


def test_self_consistent_but_undecodable_payload_falls_back_and_preserves_retention(
    tmp_path: Path,
) -> None:
    from trainguard.lifecycle import prune_checkpoints

    base = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    raw = base.model_dump()
    raw["checkpoint"]["keep_last_k"] = 2
    config = type(base).model_validate(raw)
    older, _ = _candidate(tmp_path, 1, config)
    commit_checkpoint(older, config, "run-one", "attempt-001", 1)
    middle, _ = _candidate(tmp_path, 2, config)
    commit_checkpoint(middle, config, "run-one", "attempt-001", 2)
    newest, _ = _candidate(tmp_path, 3, config)
    commit_checkpoint(newest, config, "run-one", "attempt-001", 3)
    payload = newest / "dcp/__0_0.distcp"
    changed = bytearray(payload.read_bytes())
    changed[:4] = b"bad!"
    payload.write_bytes(changed)
    manifest_path = newest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["files"] if item["path"] == "dcp/__0_0.distcp")
    entry["sha256"] = hashlib.sha256(changed).hexdigest()
    write_json_atomic(manifest_path, manifest)
    (newest / "COMMITTED").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n")

    assert validate_checkpoint(newest, config, "run-one").global_step == 3
    with pytest.raises(CheckpointInvalid, match="cannot be decoded"):
        validate_checkpoint(newest, config, "run-one", decode_payload=True)
    assert latest_valid_checkpoint(tmp_path, config, "run-one").path == middle
    prune_checkpoints(tmp_path, config, "run-one")
    assert older.exists() and middle.exists()
