"""Same-host persistence tests for the local protocol reference, not production storage."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from trainguard.local_reference_store import LocalReferenceObjectStore
from trainguard.remote_protocol import (
    EpochAuthority,
    InMemoryIsolationOracle,
    PreconditionFailed,
    RemoteCheckpointProtocol,
    ResponseLost,
)


def _compete_for_head(
    database_path: str,
    etag: str,
    payload: bytes,
    barrier: multiprocessing.synchronize.Barrier,
    results: multiprocessing.queues.Queue,
) -> None:
    store = LocalReferenceObjectStore(Path(database_path))
    barrier.wait(timeout=20)
    try:
        store.put("runs/local/HEAD", payload, if_match=etag)
    except PreconditionFailed:
        results.put("conflict")
    else:
        results.put("committed")


def test_local_reference_revisions_do_not_reuse_after_a_b_a_or_delete(tmp_path: Path) -> None:
    path = tmp_path / "objects.sqlite3"
    store = LocalReferenceObjectStore(path)
    with pytest.raises(ValueError, match="exactly one"):
        store.put("key", b"A")
    first = store.put("key", b"A", if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put("key", b"other", if_none_match=True)
    second = store.put("key", b"B", if_match=first.etag)
    third = store.put("key", b"A", if_match=second.etag)
    assert third.data == first.data
    assert len({first.etag, second.etag, third.etag}) == 3
    with pytest.raises(PreconditionFailed):
        store.put("key", b"stale overwrite", if_match=first.etag)
    assert [int(item.etag.removeprefix("revision-")) for item in (first, second, third)] == [
        1, 2, 3,
    ]
    with pytest.raises(PreconditionFailed):
        store.delete("key", if_match=first.etag)
    assert store.delete("key", if_match=third.etag)
    assert store.get("key") is None
    assert not store.delete("key", if_match=third.etag)

    reopened = LocalReferenceObjectStore(path)
    fourth = reopened.put("key", b"A", if_none_match=True)
    assert fourth.etag == "revision-4"
    reopened.put("key%extra", b"percent", if_none_match=True)
    assert reopened.list("key%") == ("key%extra",)
    assert reopened.list("key") == ("key", "key%extra")
    assert LocalReferenceObjectStore(path).get("key") == fourth


def test_local_reference_two_processes_have_one_head_cas_winner(tmp_path: Path) -> None:
    path = tmp_path / "objects.sqlite3"
    first = LocalReferenceObjectStore(path).put("runs/local/HEAD", b"initial", if_none_match=True)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(target=_compete_for_head, args=(str(path), first.etag, payload,
                                                        barrier, results))
        for payload in (b"controller-A", b"controller-B")
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=30)
        assert all(process.exitcode == 0 for process in processes)
        assert sorted(results.get(timeout=5) for _ in processes) == ["committed", "conflict"]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        results.close()
    stored = LocalReferenceObjectStore(path).get("runs/local/HEAD")
    assert stored is not None
    assert stored.data in {b"controller-A", b"controller-B"}
    assert stored.etag == "revision-2"


@pytest.mark.parametrize("loss_before_commit", [False, True])
def test_local_reference_lost_response_readback_model(
    tmp_path: Path, loss_before_commit: bool
) -> None:
    path = tmp_path / "objects.sqlite3"
    backend = LocalReferenceObjectStore(path)

    class LosingHeadResponse:
        def __init__(self) -> None:
            self.lose_once = True

        def get(self, key: str):
            return backend.get(key)

        def list(self, prefix: str):
            return backend.list(prefix)

        def delete(self, key: str, *, if_match: str | None = None):
            return backend.delete(key, if_match=if_match)

        def put(self, key: str, data: bytes, *, if_none_match=False, if_match=None):
            if key.endswith("/HEAD") and self.lose_once:
                self.lose_once = False
                if loss_before_commit:
                    raise ResponseLost("request outcome is unknown")
                backend.put(key, data, if_none_match=if_none_match, if_match=if_match)
                raise ResponseLost("response was lost after commit")
            return backend.put(key, data, if_none_match=if_none_match, if_match=if_match)

    authority = EpochAuthority(InMemoryIsolationOracle())
    controller = authority.start("local-run", "controller-a")
    worker = authority.worker(controller, "worker-a")
    protocol = RemoteCheckpointProtocol(
        LosingHeadResponse(), authority, "local-run", "local-reference-fixture"
    )
    digest = protocol.write_payload(worker, "generation-a", "state.bin", b"state")
    protocol.seal(controller, "generation-a", 1, {"state.bin": digest})
    if loss_before_commit:
        with pytest.raises(ResponseLost):
            protocol.publish(controller, "generation-a")
        assert LocalReferenceObjectStore(path).get(protocol.head_key) is None
    published = protocol.publish(controller, "generation-a")
    reopened_protocol = RemoteCheckpointProtocol(
        LocalReferenceObjectStore(path), authority, "local-run", "local-reference-fixture"
    )
    assert reopened_protocol.select_latest().chosen == published
