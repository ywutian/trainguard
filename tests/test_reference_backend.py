"""Two-rank CPU checkpoint publication and HEAD-only recovery experiments."""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import sqlite3
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch.distributed.checkpoint import FileSystemReader

from trainguard import controller, reference_backend
from trainguard.checkpoint import CheckpointInvalid, ordered_candidates, validate_checkpoint
from trainguard.config import load_config
from trainguard.events import write_json_atomic
from trainguard.local_reference_store import LocalReferenceObjectStore
from trainguard.reference_backend import LocalReferenceAuthority, LocalReferenceSession
from trainguard.remote_protocol import (
    FencedOut,
    InvalidRemoteCheckpoint,
    RemoteCheckpointProtocol,
    ResponseLost,
)
from trainguard.support import SupportBundleError, build_support_bundle
from trainguard.validation import validate_runs


def _config(tmp_path: Path, *, recover: bool) -> Path:
    raw = load_config(Path(__file__).parents[1] / "configs/cpu_demo.yaml").model_dump()
    raw["model"]["dropout"] = 0.2
    raw["checkpoint"].update(mode="sync" if recover else "none", interval_steps=1)
    raw["recovery"].update(max_restarts=3, progress_timeout_seconds=30)
    if recover:
        raw["fault"].update(kind="worker_exit", step=3, rank=0)
    path = tmp_path / ("recovered.json" if recover else "reference.json")
    path.write_text(json.dumps(raw))
    return path


def _sync_without_fault(tmp_path: Path) -> Path:
    path = _config(tmp_path, recover=True)
    raw = json.loads(path.read_text())
    raw["fault"].update(kind="none", step=None)
    path.write_text(json.dumps(raw))
    return path


def _stop_after_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    original = controller._launch_attempt

    class StopAfterFault(BaseException):
        pass

    def stop(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[3] == "attempt-001":
            assert not result.succeeded
            raise StopAfterFault
        return result

    monkeypatch.setattr(controller, "_launch_attempt", stop)
    return StopAfterFault


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _damage_concrete_key(checkpoint: Path) -> None:
    metadata_path = checkpoint / "dcp/.metadata"
    metadata = FileSystemReader(checkpoint / "dcp").read_metadata()
    name = next(key for key in metadata.state_dict_metadata if key.startswith("model."))
    changed = f"model.unsupported_{name.removeprefix('model.')}"
    metadata.state_dict_metadata[changed] = metadata.state_dict_metadata.pop(name)
    metadata.planner_data[changed] = metadata.planner_data.pop(name)
    metadata.storage_data = {
        replace(index, fqn=changed) if index.fqn == name else index: location
        for index, location in metadata.storage_data.items()
    }
    metadata_path.write_bytes(pickle.dumps(metadata))
    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["files"] if item["path"] == "dcp/.metadata")
    entry.update(size=metadata_path.stat().st_size, sha256=_digest(metadata_path))
    write_json_atomic(manifest_path, manifest)
    (checkpoint / "COMMITTED").write_text(_digest(manifest_path) + "\n")


def test_reference_backend_requires_explicit_bounded_experiment(tmp_path: Path) -> None:
    config = _config(tmp_path, recover=True)
    with pytest.raises(controller.ExperimentNotAuthorizedError):
        controller.run(
            config, tmp_path / "unauthorized",
            reference_store_path=tmp_path / "objects.sqlite3",
        )
    raw = json.loads(config.read_text())
    raw["checkpoint"]["mode"] = "async"
    config.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="two-rank CPU sync experiment"):
        controller.run(
            config, tmp_path / "unsupported", allow_experiment=True,
            reference_store_path=tmp_path / "objects.sqlite3",
        )
    assert not (tmp_path / "unauthorized").exists()
    assert not (tmp_path / "unsupported").exists()


@pytest.mark.parametrize("damage", ["missing-head", "corrupt-head", "missing-database"])
def test_completed_reference_claim_requires_existing_final_head(
    tmp_path: Path, damage: str
) -> None:
    database = tmp_path / "objects.sqlite3"
    run_dir, ok = controller.run(
        _sync_without_fault(tmp_path), tmp_path / "recovered",
        allow_experiment=True, reference_store_path=database,
    )
    assert ok, (run_dir / "launcher.log").read_text()
    saved = json.loads((run_dir / "run.json").read_text())
    store = LocalReferenceObjectStore(database)
    head_key = f"runs/{saved['run_id']}/HEAD"
    before = store.get(f"runs/{saved['run_id']}/AUTHORITY")
    assert before is not None
    assert validate_runs(run_dir, run_dir)["passed"]
    assert build_support_bundle(run_dir)["status"] == "SUCCEEDED"
    assert LocalReferenceObjectStore(database).get(
        f"runs/{saved['run_id']}/AUTHORITY"
    ) == before
    if damage == "missing-database":
        database.unlink()
    else:
        current = store.get(head_key)
        assert current is not None
        if damage == "missing-head":
            assert store.delete(head_key, if_match=current.etag)
        else:
            store.put(head_key, b"{", if_match=current.etag)
    comparison = validate_runs(run_dir, run_dir)
    assert comparison["passed"] is False
    assert any("reference final publication" in item for item in comparison["differences"])
    with pytest.raises(SupportBundleError, match="reference final publication"):
        build_support_bundle(run_dir)
    assert not controller.resume(run_dir)
    assert json.loads((run_dir / "run.json").read_text())["status"] == "FAILED"
    with sqlite3.connect(run_dir / "run.sqlite3") as index:
        assert index.execute("SELECT status FROM runs").fetchone() == ("FAILED",)
    if damage == "missing-database":
        assert not database.exists()


def test_configured_reference_path_alias_is_canonical_before_saving(tmp_path: Path) -> None:
    config_path = _sync_without_fault(tmp_path)
    raw = json.loads(config_path.read_text())
    raw["checkpoint"]["reference_store_path"] = str(tmp_path / "private" / ".." / "objects.sqlite3")
    config_path.write_text(json.dumps(raw))
    run_dir, ok = controller.run(
        config_path, tmp_path / "runs", allow_experiment=True
    )
    assert ok, (run_dir / "launcher.log").read_text()
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["local_reference_store"] == str((tmp_path / "objects.sqlite3").resolve())
    assert saved["config"]["checkpoint"]["reference_store_path"] == saved[
        "local_reference_store"
    ]
    assert validate_runs(run_dir, run_dir)["passed"]
    assert build_support_bundle(run_dir)["status"] == "SUCCEEDED"


def test_local_reference_database_requires_private_files_and_directory(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        LocalReferenceObjectStore(public / "objects.sqlite3")
    assert not (public / "objects.sqlite3").exists()
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    database = private / "objects.sqlite3"
    store = LocalReferenceObjectStore(database)
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    database.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        LocalReferenceObjectStore(database, read_only=True)
    database.chmod(0o600)
    sidecar = Path(str(database) + "-wal")
    sidecar.write_bytes(b"unsafe")
    sidecar.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        LocalReferenceObjectStore(database, read_only=True)
    sidecar.unlink()
    assert store.get("missing") is None
    missing = private / "missing.sqlite3"
    with pytest.raises(ValueError, match="missing"):
        LocalReferenceObjectStore(missing, read_only=True)
    assert not missing.exists()


def test_upload_size_preflight_runs_before_payload_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "payload").write_bytes(b"oversized")
    monkeypatch.setattr(reference_backend, "MAX_CHECKPOINT_BYTES", 1)
    monkeypatch.setattr(
        reference_backend, "validate_checkpoint",
        lambda *args, **kwargs: pytest.fail("oversized DCP was decoded"),
    )
    with pytest.raises(CheckpointInvalid, match="experiment limit"):
        LocalReferenceSession.publish_local(object.__new__(LocalReferenceSession), checkpoint)


def test_reference_head_size_is_checked_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _sync_without_fault(tmp_path)
    config = load_config(config_path)
    session = LocalReferenceSession(
        tmp_path / "run", tmp_path / "objects.sqlite3", config, "run-one", resume=False
    )
    session.store.put(session.protocol.head_key, b"{" + b"x" * 100, if_none_match=True)
    monkeypatch.setattr(reference_backend, "MAX_HEAD_BYTES", 10)
    original = LocalReferenceObjectStore.get

    def reject_head_read(self, key):
        if key == session.protocol.head_key:
            pytest.fail("oversized HEAD was read")
        return original(self, key)

    monkeypatch.setattr(LocalReferenceObjectStore, "get", reject_head_read)
    with pytest.raises(InvalidRemoteCheckpoint, match="exceeds experiment limit"):
        session.published_candidates()


def test_old_local_configuration_without_reference_field_remains_valid(
    tmp_path: Path,
) -> None:
    run_dir, ok = controller.run(_sync_without_fault(tmp_path), tmp_path / "local")
    assert ok, (run_dir / "launcher.log").read_text()
    configuration_path = run_dir / "config.json"
    configuration = json.loads(configuration_path.read_text())
    configuration["checkpoint"].pop("reference_store_path")
    write_json_atomic(configuration_path, configuration)
    status_path = run_dir / "run.json"
    status = json.loads(status_path.read_text())
    status["config"]["checkpoint"].pop("reference_store_path")
    write_json_atomic(status_path, status)
    assert validate_runs(run_dir, run_dir)["passed"]
    assert build_support_bundle(run_dir)["status"] == "SUCCEEDED"
    assert controller.resume(run_dir)
    assert json.loads(status_path.read_text())["status"] == "SUCCEEDED"


def test_reference_initialization_failure_is_saved_as_failed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "objects.sqlite3"
    database.write_bytes(b"not a database")
    database.chmod(0o600)
    run_dir, ok = controller.run(
        _sync_without_fault(tmp_path), tmp_path / "runs",
        allow_experiment=True, reference_store_path=database,
    )
    assert not ok
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["status"] == "FAILED"
    assert "reference database initialization failed" in saved["reason"]
    with sqlite3.connect(run_dir / "run.sqlite3") as index:
        assert index.execute("SELECT status FROM runs").fetchone() == ("FAILED",)


def test_final_commit_published_after_launcher_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_module = controller.subprocess
    original_publish = controller._publish_reference_commits
    observed_exit = False
    publication_after_exit = False

    class ObservedLauncher:
        def __init__(self, *args, **kwargs):
            self.process = original_module.Popen(*args, **kwargs)

        def poll(self):
            nonlocal observed_exit
            result = self.process.poll()
            observed_exit |= result is not None
            return result

        def __getattr__(self, name):
            return getattr(self.process, name)

    def defer_publication(*args, **kwargs):
        nonlocal publication_after_exit
        if not observed_exit:
            return None
        publication_after_exit = True
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        controller, "subprocess",
        SimpleNamespace(
            Popen=ObservedLauncher, run=original_module.run,
            STDOUT=original_module.STDOUT, TimeoutExpired=original_module.TimeoutExpired,
        ),
    )
    monkeypatch.setattr(controller, "_publish_reference_commits", defer_publication)
    run_dir, ok = controller.run(
        _sync_without_fault(tmp_path), tmp_path / "runs",
        allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
    )
    assert observed_exit and publication_after_exit and ok
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["post_run_audit"]["final_checkpoint"]["global_step"] == 4
    assert validate_runs(run_dir, run_dir)["passed"]


def test_head_restores_real_two_rank_training_after_local_checkpoints_are_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, ok = controller.run(_config(tmp_path, recover=False), tmp_path / "reference")
    assert ok, (reference / "launcher.log").read_text()
    database = tmp_path / "objects.sqlite3"
    with monkeypatch.context() as patch:
        stopped = _stop_after_first_attempt(patch)
        with pytest.raises(stopped):
            controller.run(
                _config(tmp_path, recover=True), tmp_path / "recovered",
                allow_experiment=True, reference_store_path=database,
            )
    recovered = next((tmp_path / "recovered").iterdir())
    assert len(ordered_candidates(recovered)) == 2
    run_id = json.loads((recovered / "run.json").read_text())["run_id"]
    authority = LocalReferenceAuthority(
        LocalReferenceObjectStore(database), run_id,
        load_config(recovered / "config.json").fingerprint(),
        recovered,
    )
    protocol = RemoteCheckpointProtocol(
        authority.store, authority, run_id, authority.identity, history_limit=8
    )
    assert [item.global_step for item in protocol.published_candidates()] == [2, 1]
    shutil.rmtree(recovered / "checkpoints")
    assert controller.resume(recovered), (recovered / "launcher.log").read_text()
    comparison = validate_runs(reference, recovered)
    assert comparison["passed"], comparison
    with sqlite3.connect(recovered / "run.sqlite3") as connection:
        selected = connection.execute(
            "SELECT checkpoint_path, resume_step FROM recoveries WHERE to_attempt='attempt-002'"
        ).fetchone()
    assert selected is not None and "reference-cache" in selected[0] and selected[1] == 2
    saved = json.loads((recovered / "run.json").read_text())
    assert saved["config"]["checkpoint"]["reference_store_path"] == str(database)
    saved.pop("local_reference_store")
    write_json_atomic(recovered / "run.json", saved)
    with pytest.raises(ValueError, match="database identity differs"):
        controller.resume(recovered)


def test_local_committed_without_head_is_not_recovery_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(controller, "_publish_reference_commits", lambda *args, **kwargs: None)
        stopped = _stop_after_first_attempt(patch)
        with pytest.raises(stopped):
            controller.run(
                _config(tmp_path, recover=True), tmp_path / "recovered",
                allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
            )
    recovered = next((tmp_path / "recovered").iterdir())
    assert len(ordered_candidates(recovered)) == 2
    assert not controller.resume(recovered)
    assert "no valid checkpoint" in json.loads((recovered / "run.json").read_text())["reason"]


@pytest.mark.parametrize("intrusion", ["symlink", "public", "oversize"])
def test_reference_cache_and_download_limit_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, intrusion: str
) -> None:
    with monkeypatch.context() as patch:
        stopped = _stop_after_first_attempt(patch)
        with pytest.raises(stopped):
            controller.run(
                _config(tmp_path, recover=True), tmp_path / "recovered",
                allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
            )
    recovered = next((tmp_path / "recovered").iterdir())
    cache = recovered / "reference-cache"
    outside = tmp_path / "outside"
    outside.mkdir()
    if intrusion == "symlink":
        cache.symlink_to(outside, target_is_directory=True)
    elif intrusion == "public":
        cache.mkdir()
        cache.chmod(0o755)
    else:
        monkeypatch.setattr(reference_backend, "MAX_CHECKPOINT_BYTES", 100)
        original_get = LocalReferenceObjectStore.get

        def refuse_payload_read(self, key):
            if "/payload/" in key:
                raise AssertionError("oversized payload was read into memory")
            return original_get(self, key)

        monkeypatch.setattr(LocalReferenceObjectStore, "get", refuse_payload_read)
    shutil.rmtree(recovered / "checkpoints")
    assert not controller.resume(recovered)
    assert "no valid checkpoint" in json.loads((recovered / "run.json").read_text())["reason"]
    assert list(outside.iterdir()) == []


def test_old_local_identity_cannot_upload_or_publish_after_epoch_claim(tmp_path: Path) -> None:
    store = LocalReferenceObjectStore(tmp_path / "objects.sqlite3")
    first_authority = LocalReferenceAuthority(store, "run-one", "identity", tmp_path / "run")
    first = first_authority.claim("controller-a", resume=False)
    old_worker = first_authority.worker(first, "rank-zero")
    protocol = RemoteCheckpointProtocol(store, first_authority, "run-one", "identity")
    digest = protocol.write_payload(old_worker, "generation-one", "state.bin", b"state")
    protocol.seal(first, "generation-one", 1, {"state.bin": digest})
    protocol.publish(first, "generation-one")
    second_authority = LocalReferenceAuthority(
        LocalReferenceObjectStore(tmp_path / "objects.sqlite3"), "run-one", "identity",
        tmp_path / "run",
    )
    second = second_authority.claim("controller-b", resume=True)
    successor = RemoteCheckpointProtocol(
        second_authority.store, second_authority, "run-one", "identity"
    )
    successor.synchronize_head_epoch(second)
    with pytest.raises(FencedOut):
        protocol.write_payload(old_worker, "generation-two", "state.bin", b"old")
    with pytest.raises(FencedOut):
        protocol.publish(first, "generation-one")
    assert successor.published_candidates()[0].generation_id == "generation-one"
    with pytest.raises(FencedOut, match="identity"):
        LocalReferenceAuthority(
            LocalReferenceObjectStore(tmp_path / "objects.sqlite3"),
            "run-one", "identity", tmp_path / "copied-run",
        ).claim("controller-c", resume=True)


def test_head_lost_ack_is_confirmed_by_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = LocalReferenceObjectStore.put
    lost = False

    def lose_once(self, key, data, *, if_none_match=False, if_match=None):
        nonlocal lost
        result = original(self, key, data, if_none_match=if_none_match, if_match=if_match)
        if key.endswith("/HEAD") and not lost:
            lost = True
            raise ResponseLost("ack was lost after commit")
        return result

    monkeypatch.setattr(LocalReferenceObjectStore, "put", lose_once)
    recovered, ok = controller.run(
        _config(tmp_path, recover=True), tmp_path / "recovered",
        allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
    )
    assert lost and ok, (recovered / "launcher.log").read_text()
    assert json.loads((recovered / "run.json").read_text())["post_run_audit"]["status"] == "PASSED"


def test_unapplied_unknown_head_response_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = LocalReferenceObjectStore.put
    lost = False

    def lose_before_write(self, key, data, *, if_none_match=False, if_match=None):
        nonlocal lost
        if key.endswith("/HEAD") and not lost:
            lost = True
            raise ResponseLost("head outcome is unknown")
        return original(self, key, data, if_none_match=if_none_match, if_match=if_match)

    monkeypatch.setattr(LocalReferenceObjectStore, "put", lose_before_write)
    recovered, ok = controller.run(
        _config(tmp_path, recover=True), tmp_path / "recovered",
        allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
    )
    assert lost and not ok
    saved = json.loads((recovered / "run.json").read_text())
    assert saved["status"] == "FAILED" and "ResponseLost" in saved["reason"]
    assert LocalReferenceObjectStore(tmp_path / "objects.sqlite3").get(
        f"runs/{saved['run_id']}/HEAD"
    ) is None


def test_final_worker_commit_event_cannot_replace_missing_head_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = LocalReferenceSession.publish_local
    stopped = False

    def fail_final(self, path, *, recovery=False):
        nonlocal stopped
        if path.name.startswith("step-000004-") and not recovery and not stopped:
            stopped = True
            raise ResponseLost("final HEAD write was not confirmed")
        return original(self, path, recovery=recovery)

    with monkeypatch.context() as patch:
        patch.setattr(LocalReferenceSession, "publish_local", fail_final)
        recovered, ok = controller.run(
            _sync_without_fault(tmp_path), tmp_path / "recovered",
            allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
        )
    assert stopped and not ok
    saved = json.loads((recovered / "run.json").read_text())
    assert saved["status"] == "FAILED"
    assert saved.get("post_run_audit", {}).get("status") != "PASSED"
    rank_zero_events = [
        json.loads(line) for line in
        (recovered / "attempts/attempt-001/rank-0.jsonl").read_text().splitlines()
    ]
    assert any(
        event["event_type"] == "checkpoint_committed" and event["global_step"] == 4
        for event in rank_zero_events
    )
    config = load_config(recovered / "config.json")
    store = LocalReferenceObjectStore(tmp_path / "objects.sqlite3")
    authority = LocalReferenceAuthority(store, saved["run_id"], config.fingerprint(), recovered)
    protocol = RemoteCheckpointProtocol(
        store, authority, saved["run_id"], config.fingerprint(), history_limit=8
    )
    assert protocol.published_candidates()[0].global_step == 3
    shutil.rmtree(recovered / "checkpoints")
    assert controller.resume(recovered), (recovered / "launcher.log").read_text()
    with sqlite3.connect(recovered / "run.sqlite3") as connection:
        selected = connection.execute(
            "SELECT resume_step FROM recoveries WHERE to_attempt='attempt-002'"
        ).fetchone()
    assert selected == (3,)


def test_final_head_ack_lost_after_commit_is_confirmed_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = LocalReferenceObjectStore.put
    lost = False

    def lose_final_ack(self, key, data, *, if_none_match=False, if_match=None):
        nonlocal lost
        result = original(self, key, data, if_none_match=if_none_match, if_match=if_match)
        if key.endswith("/HEAD") and not lost:
            head = json.loads(data)
            if head["commits"] and head["commits"][0]["global_step"] == 4:
                lost = True
                raise ResponseLost("final HEAD ack was lost after commit")
        return result

    monkeypatch.setattr(LocalReferenceObjectStore, "put", lose_final_ack)
    recovered, ok = controller.run(
        _sync_without_fault(tmp_path), tmp_path / "recovered",
        allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
    )
    assert lost and ok, (recovered / "launcher.log").read_text()
    saved = json.loads((recovered / "run.json").read_text())
    assert saved["status"] == "SUCCEEDED"
    assert saved["post_run_audit"]["final_checkpoint"]["global_step"] == 4


@pytest.mark.parametrize("damage_all", [False, True])
def test_byte_damaged_head_candidates_fall_back_or_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage_all: bool
) -> None:
    database = tmp_path / "objects.sqlite3"
    with monkeypatch.context() as patch:
        stopped = _stop_after_first_attempt(patch)
        with pytest.raises(stopped):
            controller.run(
                _config(tmp_path, recover=True), tmp_path / "recovered",
                allow_experiment=True, reference_store_path=database,
            )
    recovered = next((tmp_path / "recovered").iterdir())
    saved = json.loads((recovered / "run.json").read_text())
    config = load_config(recovered / "config.json")
    store = LocalReferenceObjectStore(database)
    authority = LocalReferenceAuthority(store, saved["run_id"], config.fingerprint(), recovered)
    protocol = RemoteCheckpointProtocol(
        store, authority, saved["run_id"], config.fingerprint(), history_limit=8
    )
    candidates = protocol.published_candidates()
    assert [item.global_step for item in candidates] == [2, 1]
    for candidate in candidates if damage_all else candidates[:1]:
        key = protocol.payload_key(candidate.generation_id, "rank-1.json")
        current = store.get(key)
        assert current is not None
        store.put(key, b"damaged", if_match=current.etag)
    shutil.rmtree(recovered / "checkpoints")
    succeeded = controller.resume(recovered)
    assert succeeded is not damage_all
    if damage_all:
        assert "no valid checkpoint" in json.loads((recovered / "run.json").read_text())["reason"]
    else:
        with sqlite3.connect(recovered / "run.sqlite3") as connection:
            selected = connection.execute(
                "SELECT resume_step FROM recoveries WHERE to_attempt='attempt-002'"
            ).fetchone()
        assert selected == (1,)


def test_self_consistent_bad_head_candidate_falls_back_to_older_dcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, ok = controller.run(_config(tmp_path, recover=False), tmp_path / "reference")
    assert ok
    original = controller._publish_reference_commits
    damaged = False

    def damage_before_publish(reference_backend, run_dir, config, run_id, attempt_id,
                              published, milestone, *, recovery):
        nonlocal damaged
        candidate = run_dir / "checkpoints" / f"step-000002-{attempt_id}"
        if candidate.joinpath("COMMITTED").is_file() and not damaged:
            _damage_concrete_key(candidate)
            assert validate_checkpoint(
                candidate, config, run_id, decode_payload=True, require_trainable_state=True
            )
            damaged = True
        return original(
            reference_backend, run_dir, config, run_id, attempt_id, published, milestone,
            recovery=recovery,
        )

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_publish_reference_commits", damage_before_publish)
        stopped = _stop_after_first_attempt(patch)
        with pytest.raises(stopped):
            controller.run(
                _config(tmp_path, recover=True), tmp_path / "recovered",
                allow_experiment=True, reference_store_path=tmp_path / "objects.sqlite3",
            )
    assert damaged
    recovered = next((tmp_path / "recovered").iterdir())
    shutil.rmtree(recovered / "checkpoints")
    assert controller.resume(recovered), (recovered / "launcher.log").read_text()
    with sqlite3.connect(recovered / "run.sqlite3") as connection:
        attempts = connection.execute(
            "SELECT attempt_id, status, resume_step FROM attempts ORDER BY number"
        ).fetchall()
    assert len(attempts) == 3
    assert attempts[1][1:] == ("FAILED", 2)
    assert attempts[2][1:] == ("SUCCEEDED", 1)
    assert validate_runs(reference, recovered)["passed"]
