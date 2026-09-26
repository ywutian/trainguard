import json
from pathlib import Path

import pytest

from trainguard import controller, events
from trainguard.checkpoint import candidate_path, capture_rank_state, commit_checkpoint
from trainguard.config import ProjectConfig, load_config
from trainguard.run_store import RunStore


def candidate(root, config, step):
    path = candidate_path(root, "attempt-001", step)
    (path / "dcp").mkdir(parents=True)
    (path / "dcp/.metadata").write_bytes(b"metadata")
    for rank in range(config.run.world_size):
        (path / f"dcp/__{rank}_0.distcp").write_bytes(b"data")
        events.write_json_atomic(
            path / f"rank-{rank}.json",
            capture_rank_state(config, "run", "attempt-001", rank, step, {}),
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

    def track(path, *args):
        calls.append(path)
        return original(path, *args)

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
