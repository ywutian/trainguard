"""Carry real DCP checkpoint bytes through the remote publication model."""

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

from trainguard.checkpoint import (
    candidate_path,
    capture_rank_state,
    commit_checkpoint,
    validate_checkpoint,
)
from trainguard.config import ProjectConfig, load_config
from trainguard.events import write_json_atomic
from trainguard.remote_protocol import (
    EpochAuthority,
    InMemoryIsolationOracle,
    InMemoryObjectStore,
    RemoteCheckpointProtocol,
)


def _single_rank_config() -> ProjectConfig:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["run"]["world_size"] = 1
    return ProjectConfig.model_validate(raw)


def _checkpoint(root: Path, config: ProjectConfig, step: int) -> Path:
    path = candidate_path(root, "attempt-001", step)
    dcp.save(
        {"model": {"weight": torch.full((4,), float(step))}},
        checkpoint_id=path / "dcp",
    )
    write_json_atomic(
        path / "rank-0.json",
        capture_rank_state(config, "run", "attempt-001", 0, step, {}),
    )
    commit_checkpoint(path, config, "run", "attempt-001", step)
    return path


def test_real_dcp_bytes_publish_fallback_and_restore_through_remote_model(tmp_path: Path) -> None:
    config = _single_rank_config()
    store = InMemoryObjectStore()
    authority = EpochAuthority(InMemoryIsolationOracle())
    controller = authority.start("run", "controller-a")
    worker = authority.worker(controller, "worker-a")
    protocol = RemoteCheckpointProtocol(store, authority, "run", config.fingerprint())
    generations = {}

    for step in (1, 2):
        candidate = _checkpoint(tmp_path / "source", config, step)
        generation = f"generation-{step}"
        payloads = {
            file.relative_to(candidate).as_posix(): file.read_bytes()
            for file in candidate.rglob("*")
            if file.is_file()
        }
        expected = {
            path: protocol.write_payload(worker, generation, path, data)
            for path, data in payloads.items()
        }
        protocol.seal(controller, generation, step, expected)
        assert protocol.publish(controller, generation).global_step == step
        generations[generation] = payloads

    assert protocol.select_latest().chosen.global_step == 2
    newest_payload = next(path for path in generations["generation-2"] if path.endswith(".distcp"))
    key = protocol.payload_key("generation-2", newest_payload)
    original = store.get(key)
    store.put(key, b"broken payload", if_match=original.etag)

    selection = protocol.select_latest()
    assert selection.chosen is not None and selection.chosen.global_step == 1
    assert [candidate.generation_id for candidate in selection.rejected] == ["generation-2"]

    restored_path = candidate_path(tmp_path / "download", "attempt-001", 1)
    for path in generations[selection.chosen.generation_id]:
        stored = store.get(protocol.payload_key(selection.chosen.generation_id, path))
        assert stored is not None
        destination = restored_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(stored.data)

    assert validate_checkpoint(restored_path, config, "run", decode_payload=True).global_step == 1
    restored = {"model": {"weight": torch.zeros(4)}}
    dcp.load(restored, checkpoint_id=restored_path / "dcp")
    torch.testing.assert_close(restored["model"]["weight"], torch.ones(4))
