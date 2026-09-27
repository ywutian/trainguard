"""S3/DynamoDB adapter contract checks against in-process service doubles."""

from __future__ import annotations

import threading

import pytest
from aws_fakes import FakeClientError, FakeDynamoDB, FakeS3, FakeTransportError

from trainguard.aws_store import AwsObjectStore, parse_location
from trainguard.remote_protocol import (
    EpochAuthority,
    FencedOut,
    GenerationConflict,
    InMemoryIsolationOracle,
    PreconditionFailed,
    RemoteCheckpointProtocol,
    RemoteProtocolError,
    ResponseLost,
)

LOCATION = "aws://trainguard-test/team/ckpt?region=us-east-1&table=trainguard-heads"


def _store(**kwargs) -> tuple[AwsObjectStore, FakeS3, FakeDynamoDB]:
    s3, dynamodb = FakeS3(), FakeDynamoDB()
    store = AwsObjectStore(
        s3, dynamodb, parse_location(LOCATION),
        transport_errors=(FakeTransportError,), **kwargs,
    )
    return store, s3, dynamodb


def _protocol(store):
    oracle = InMemoryIsolationOracle()
    authority = EpochAuthority(oracle)
    controller = authority.start("run-one", "controller-a")
    worker = authority.worker(controller, "worker-a")
    protocol = RemoteCheckpointProtocol(store, authority, "run-one", "identity-v1")
    return protocol, oracle, authority, controller, worker


def _publish(protocol, controller, worker, generation, step):
    expected = {
        path: protocol.write_payload(worker, generation, path, f"{path}-{step}".encode())
        for path in ("dcp/.metadata", "dcp/__0_0.distcp", "rank-0.json")
    }
    protocol.seal(controller, generation, step, expected)
    return protocol.publish(controller, generation)


def test_location_is_canonical_and_explicit() -> None:
    location = parse_location(LOCATION)
    assert (location.bucket, location.prefix, location.table, location.region) == (
        "trainguard-test", "team/ckpt", "trainguard-heads", "us-east-1",
    )
    for bad in (
        "aws://trainguard-test/team/ckpt?table=trainguard-heads&region=us-east-1",
        "aws://trainguard-test/team/ckpt?region=us-east-1",
        "aws://trainguard-test/?region=us-east-1&table=trainguard-heads",
        "aws://Bad_Bucket/ckpt?region=us-east-1&table=trainguard-heads",
        "aws://trainguard-test/../x?region=us-east-1&table=trainguard-heads",
        "s3://trainguard-test/ckpt?region=us-east-1&table=trainguard-heads",
    ):
        with pytest.raises(ValueError):
            parse_location(bad)


def test_mutable_records_use_revisions_that_survive_a_b_a() -> None:
    store, _, dynamodb = _store()
    key = "runs/run-one/HEAD"
    first = store.put(key, b"A", if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put(key, b"other", if_none_match=True)
    second = store.put(key, b"B", if_match=first.etag)
    third = store.put(key, b"A", if_match=second.etag)
    assert [first.etag, second.etag, third.etag] == ["revision-1", "revision-2", "revision-3"]
    with pytest.raises(PreconditionFailed):
        store.put(key, b"stale", if_match=first.etag)
    with pytest.raises(PreconditionFailed):
        store.put(key, b"forged", if_match='"not-a-revision"')
    assert store.get(key) == third
    assert store.object_size(key) == 1
    with pytest.raises(ValueError, match="never deleted"):
        store.delete(key)
    assert set(dynamodb.items) == {"team/ckpt/runs/run-one/HEAD"}
    with pytest.raises(ValueError, match="DynamoDB item budget"):
        store.put("runs/run-one/AUTHORITY", b"x" * (256 * 1024 + 1), if_none_match=True)


def test_generation_objects_are_create_only_checksummed_and_listed_by_page() -> None:
    store, s3, _ = _store()
    root = "runs/run-one/generations/g1/payload/"
    for name in ("a", "b", "c", "d", "e"):
        store.put(root + name, name.encode(), if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put(root + "a", b"changed", if_none_match=True)
    with pytest.raises(ValueError, match="create-only"):
        store.put(root + "a", b"changed", if_match=FakeS3.etag(b"a"))
    assert store.get(root + "a").data == b"a"
    assert store.get(root + "missing") is None
    assert store.object_size(root + "missing") is None
    assert store.list(root) == tuple(root + name for name in "abcde")
    assert set(s3.objects) == {f"team/ckpt/{root}{name}" for name in "abcde"}
    assert s3.unchecked_puts == 0
    with pytest.raises(ValueError, match="generation prefixes"):
        store.list("runs/run-one/")
    assert store.delete(root + "e") and not store.delete(root + "e")


def test_read_only_store_rejects_every_write() -> None:
    store, _, _ = _store(read_only=True)
    with pytest.raises(ValueError, match="read only"):
        store.put("runs/run-one/HEAD", b"A", if_none_match=True)
    with pytest.raises(ValueError, match="read only"):
        store.delete("runs/run-one/generations/g/payload/a")


def test_protocol_selects_newest_and_falls_back_on_damaged_s3_payload() -> None:
    store, s3, _ = _store()
    protocol, _, _, controller, worker = _protocol(store)
    older = _publish(protocol, controller, worker, "generation-a", 10)
    newest = _publish(protocol, controller, worker, "generation-b", 20)
    assert protocol.select_latest().chosen == newest
    s3.objects["team/ckpt/" + protocol.payload_key("generation-b", "rank-0.json")] = b"bitrot"
    selection = protocol.select_latest()
    assert selection.chosen == older
    assert [item.generation_id for item in selection.rejected] == ["generation-b"]
    with pytest.raises(GenerationConflict):
        protocol.write_payload(worker, "generation-a", "late", b"late")


def test_dropped_connections_after_apply_are_reconciled_by_readback() -> None:
    store, s3, dynamodb = _store()
    protocol, _, _, controller, worker = _protocol(store)
    s3.inject("put", "team/ckpt/" + protocol.payload_key("generation-a", "dcp/.metadata"), "after")
    s3.inject("put", "team/ckpt/" + protocol.manifest_key("generation-a"), "after")
    dynamodb.inject("put", "team/ckpt/" + protocol.head_key, "after")
    published = _publish(protocol, controller, worker, "generation-a", 10)
    assert protocol.select_latest().chosen == published


def test_unapplied_or_throttled_head_write_is_not_claimed() -> None:
    store, _, dynamodb = _store()
    protocol, _, _, controller, worker = _protocol(store)
    _publish(protocol, controller, worker, "generation-a", 10)
    expected = {
        "rank-0.json": protocol.write_payload(worker, "generation-b", "rank-0.json", b"b"),
    }
    protocol.seal(controller, "generation-b", 20, expected)
    for fault in ("before", "throttle"):
        dynamodb.inject("put", "team/ckpt/" + protocol.head_key, fault)
        with pytest.raises(ResponseLost):
            protocol.publish(controller, "generation-b")
        assert protocol.select_latest().chosen.global_step == 10
    assert protocol.publish(controller, "generation-b").global_step == 20


def test_s3_conflicts_retry_then_fail_closed() -> None:
    store, s3, _ = _store()
    key = "runs/run-one/generations/g1/payload/a"
    s3.inject("put", "team/ckpt/" + key, "conflict", "conflict")
    assert store.put(key, b"a", if_none_match=True).data == b"a"
    other = "runs/run-one/generations/g1/payload/b"
    s3.inject("put", "team/ckpt/" + other, *["conflict"] * 4)
    with pytest.raises(RemoteProtocolError, match="kept conflicting"):
        store.put(other, b"b", if_none_match=True)
    assert store.get(other) is None


def test_takeover_barrier_fences_old_controller_over_aws() -> None:
    store, _, _ = _store()
    protocol, oracle, authority, old_controller, old_worker = _protocol(store)
    _publish(protocol, old_controller, old_worker, "generation-a", 10)
    oracle.confirm("run-one", old_controller.epoch)
    new_controller = protocol.takeover("controller-b")
    with pytest.raises(FencedOut):
        protocol.write_payload(old_worker, "generation-stale", "weights", b"old")
    with pytest.raises(FencedOut):
        protocol.publish(old_controller, "generation-a")
    new_worker = authority.worker(new_controller, "worker-b")
    assert _publish(protocol, new_controller, new_worker, "generation-b", 20).epoch == 2


def test_concurrent_head_writers_have_exactly_one_winner() -> None:
    store, _, _ = _store()
    first = store.put("runs/run-one/HEAD", b"initial", if_none_match=True)
    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def race(index: int) -> None:
        barrier.wait(timeout=10)
        try:
            store.put("runs/run-one/HEAD", f"writer-{index}".encode(), if_match=first.etag)
        except PreconditionFailed:
            outcomes.append("conflict")
        else:
            outcomes.append("committed")

    threads = [threading.Thread(target=race, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert sorted(outcomes) == ["committed"] + ["conflict"] * 7
    assert store.get("runs/run-one/HEAD").etag == "revision-2"


def test_denied_or_unexpected_service_errors_fail_closed_as_protocol_errors() -> None:
    store, s3, dynamodb = _store()

    def denied(**kwargs):
        raise FakeClientError("AccessDenied", 403)

    s3.get_object = denied
    s3.put_object = denied
    dynamodb.get_item = denied
    with pytest.raises(RemoteProtocolError, match="read failed: AccessDenied"):
        store.get("runs/run-one/generations/g/payload/a")
    with pytest.raises(RemoteProtocolError, match="write failed: AccessDenied"):
        store.put("runs/run-one/generations/g/payload/a", b"a", if_none_match=True)
    with pytest.raises(RemoteProtocolError, match="read failed: AccessDenied"):
        store.get("runs/run-one/HEAD")
