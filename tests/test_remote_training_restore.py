"""Restore a real two-rank training run from bytes carried by the remote model."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from trainguard import controller
from trainguard.checkpoint import candidate_path, ordered_candidates, validate_checkpoint
from trainguard.config import load_config
from trainguard.remote_protocol import (
    EpochAuthority,
    InMemoryIsolationOracle,
    InMemoryObjectStore,
    RemoteCheckpointProtocol,
)
from trainguard.validation import validate_runs


def _configuration(tmp_path: Path, *, recover: bool) -> Path:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["model"]["dropout"] = 0.2
    raw["checkpoint"].update(mode="sync" if recover else "none", interval_steps=1)
    raw["recovery"].update(max_restarts=2, progress_timeout_seconds=20)
    if recover:
        raw["fault"].update(kind="worker_exit", step=3, rank=0)
    path = tmp_path / ("recovery-config.json" if recover else "reference-config.json")
    path.write_text(json.dumps(raw))
    return path


def test_two_rank_training_recovers_from_remote_model_after_local_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, reference_ok = controller.run(
        _configuration(tmp_path, recover=False), tmp_path / "reference-runs"
    )
    assert reference_ok, (reference / "launcher.log").read_text()

    original_launch = controller._launch_attempt

    class StopAfterFault(BaseException):
        pass

    def stop_after_fault(*args, **kwargs):
        result = original_launch(*args, **kwargs)
        if args[3] == "attempt-001":
            assert not result.succeeded
            raise StopAfterFault
        return result

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_launch_attempt", stop_after_fault)
        with pytest.raises(StopAfterFault):
            controller.run(
                _configuration(tmp_path, recover=True), tmp_path / "recovery-runs"
            )

    recovered = next((tmp_path / "recovery-runs").iterdir())
    config = load_config(recovered / "config.json")
    run_id = json.loads((recovered / "run.json").read_text())["run_id"]
    candidates = ordered_candidates(recovered)
    assert [item.name for item in candidates] == [
        "step-000002-attempt-001", "step-000001-attempt-001"
    ]

    store = InMemoryObjectStore()
    authority = EpochAuthority(InMemoryIsolationOracle())
    owner = authority.start(run_id, "controller-a")
    workers = [authority.worker(owner, f"worker-{rank}") for rank in range(2)]
    protocol = RemoteCheckpointProtocol(store, authority, run_id, config.fingerprint())
    paths_by_generation = {}
    for checkpoint in reversed(candidates):
        step = int(checkpoint.name.split("-")[1])
        generation = f"generation-{step}"
        paths = {}
        for entry in checkpoint.rglob("*"):
            if not entry.is_file():
                continue
            relative = entry.relative_to(checkpoint).as_posix()
            rank = 1 if relative == "rank-1.json" or relative.startswith("dcp/__1_") else 0
            paths[relative] = protocol.write_payload(
                workers[rank], generation, relative, entry.read_bytes()
            )
        protocol.seal(owner, generation, step, paths)
        protocol.publish(owner, generation)
        paths_by_generation[generation] = tuple(paths)

    newest_key = protocol.payload_key("generation-2", "rank-1.json")
    newest = store.get(newest_key)
    assert newest is not None
    store.put(newest_key, b"damaged rank state", if_match=newest.etag)
    selection = protocol.select_latest()
    assert selection.chosen is not None and selection.chosen.global_step == 1
    assert [item.generation_id for item in selection.rejected] == ["generation-2"]

    shutil.rmtree(recovered / "checkpoints")
    downloaded = candidate_path(recovered, "attempt-001", 1)
    for relative in paths_by_generation[selection.chosen.generation_id]:
        stored = store.get(protocol.payload_key(selection.chosen.generation_id, relative))
        assert stored is not None
        destination = downloaded / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(stored.data)
    assert validate_checkpoint(downloaded, config, run_id, decode_payload=True).global_step == 1

    assert controller.resume(recovered), (recovered / "launcher.log").read_text()
    comparison = validate_runs(reference, recovered)
    assert comparison["passed"], comparison
    with sqlite3.connect(recovered / "run.sqlite3") as database:
        selected = database.execute(
            "SELECT checkpoint_path, resume_step FROM recoveries WHERE to_attempt='attempt-002'"
        ).fetchone()
    assert selected == (str(downloaded), 1)
    for rank in range(2):
        events = [
            json.loads(line)
            for line in (recovered / f"attempts/attempt-002/rank-{rank}.jsonl").read_text().splitlines()
        ]
        loaded = [item for item in events if item["event_type"] == "state_loaded"]
        assert len(loaded) == 1 and loaded[0]["global_step"] == 1
