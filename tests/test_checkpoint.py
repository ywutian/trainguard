import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

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


def _candidate(root: Path, step: int) -> tuple[Path, object]:
    config = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    path = candidate_path(root, "attempt-001", step)
    (path / "dcp").mkdir(parents=True)
    (path / "dcp" / ".metadata").write_bytes(b"metadata")
    for rank in range(config.run.world_size):
        (path / "dcp" / f"__{rank}_0.distcp").write_bytes(f"rank {rank}".encode())
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
