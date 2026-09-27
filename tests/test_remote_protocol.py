import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from trainguard.remote_protocol import (
    EpochAuthority,
    FencedOut,
    GenerationConflict,
    InMemoryIsolationOracle,
    InMemoryObjectStore,
    InvalidRemoteCheckpoint,
    IsolationRequired,
    PreconditionFailed,
    PublishConflict,
    RemoteCheckpointProtocol,
    ResponseLost,
)


def _setup(store=None, *, history_limit=3):
    store = store or InMemoryObjectStore()
    oracle = InMemoryIsolationOracle()
    authority = EpochAuthority(oracle)
    controller = authority.start("run-one", "controller-a")
    worker = authority.worker(controller, "worker-a")
    protocol = RemoteCheckpointProtocol(
        store, authority, "run-one", "source-data-config-v1", history_limit=history_limit
    )
    return store, oracle, authority, protocol, controller, worker


def _stage(protocol, controller, worker, generation, step, payloads=None):
    payloads = payloads or {
        "dcp/.metadata": f"metadata-{step}".encode(),
        "dcp/__0_0.distcp": f"rank-{step}".encode(),
        "rank-0.json": f"state-{step}".encode(),
    }
    expected = {
        path: protocol.write_payload(worker, generation, path, data)
        for path, data in payloads.items()
    }
    manifest_hash = protocol.seal(controller, generation, step, expected)
    return expected, manifest_hash


def _publish(protocol, controller, worker, generation, step):
    _stage(protocol, controller, worker, generation, step)
    return protocol.publish(controller, generation)


def test_store_conditions_use_unique_revision_tokens():
    store = InMemoryObjectStore()
    first = store.put("key", b"same", if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put("key", b"other", if_none_match=True)
    with pytest.raises(PreconditionFailed):
        store.put("key", b"other", if_match="stale")
    second = store.put("key", b"other", if_match=first.etag)
    third = store.put("key", b"same", if_match=second.etag)
    assert third.etag not in {first.etag, second.etag}
    assert store.list("ke") == ("key",)
    with pytest.raises(PreconditionFailed):
        store.delete("key", if_match=first.etag)
    assert store.delete("key", if_match=third.etag)


def test_partial_or_unsealed_generation_is_never_selected():
    store, _, _, protocol, controller, worker = _setup()
    expected = {
        "dcp/.metadata": protocol.write_payload(worker, "generation-a", "dcp/.metadata", b"a")
    }
    assert protocol.select_latest().chosen is None
    expected["dcp/__0_0.distcp"] = expected["dcp/.metadata"]
    with pytest.raises(InvalidRemoteCheckpoint, match="missing"):
        protocol.seal(controller, "generation-a", 10, expected)
    assert store.get(protocol.manifest_key("generation-a")) is None
    assert protocol.select_latest().chosen is None
    expected["dcp/__0_0.distcp"] = protocol.write_payload(
        worker, "generation-a", "dcp/__0_0.distcp", b"a"
    )
    protocol.seal(controller, "generation-a", 10, expected)
    assert protocol.select_latest().chosen is None
    assert protocol.publish(controller, "generation-a").global_step == 10
    assert protocol.select_latest().chosen.generation_id == "generation-a"


def test_multiple_workers_can_stage_out_of_order_but_all_expected_parts_are_required():
    _, _, authority, protocol, controller, worker_a = _setup()
    worker_b = authority.worker(controller, "worker-b")
    second = protocol.write_payload(worker_b, "generation-a", "dcp/__1_0.distcp", b"rank-b")
    first = protocol.write_payload(worker_a, "generation-a", "dcp/__0_0.distcp", b"rank-a")
    metadata = protocol.write_payload(worker_a, "generation-a", "dcp/.metadata", b"metadata")
    with pytest.raises(InvalidRemoteCheckpoint, match="payload set"):
        protocol.seal(
            controller,
            "generation-a",
            10,
            {"dcp/__0_0.distcp": first, "dcp/.metadata": metadata},
        )
    protocol.seal(
        controller,
        "generation-a",
        10,
        {
            "dcp/__0_0.distcp": first,
            "dcp/__1_0.distcp": second,
            "dcp/.metadata": metadata,
        },
    )
    assert protocol.publish(controller, "generation-a").global_step == 10


def test_immutable_generation_rejects_collision_and_extra_payload():
    _, _, _, protocol, controller, worker = _setup()
    digest = protocol.write_payload(worker, "generation-a", "weights", b"first")
    assert protocol.write_payload(worker, "generation-a", "weights", b"first") == digest
    with pytest.raises(GenerationConflict, match="differs"):
        protocol.write_payload(worker, "generation-a", "weights", b"second")
    protocol.write_payload(worker, "generation-a", "unexpected", b"extra")
    with pytest.raises(InvalidRemoteCheckpoint, match="payload set"):
        protocol.seal(controller, "generation-a", 10, {"weights": digest})
    with pytest.raises(ValueError, match="safe relative"):
        protocol.write_payload(worker, "generation-b", "../escape", b"bad")


def test_corrupt_or_missing_latest_payload_falls_back_with_reason():
    store, _, _, protocol, controller, worker = _setup()
    _publish(protocol, controller, worker, "generation-a", 10)
    _publish(protocol, controller, worker, "generation-b", 20)
    payload_key = protocol.payload_key("generation-b", "dcp/__0_0.distcp")
    original = store.get(payload_key)
    store.put(payload_key, b"silent-corruption", if_match=original.etag)
    selection = protocol.select_latest()
    assert selection.chosen.generation_id == "generation-a"
    assert selection.rejected[0].generation_id == "generation-b"
    assert "hash or size" in selection.rejected[0].reason
    store.delete(payload_key)
    assert protocol.select_latest().chosen.generation_id == "generation-a"


def test_recovery_can_republish_a_step_after_rollback_without_losing_head_order():
    _, _, _, protocol, controller, worker = _setup()
    _publish(protocol, controller, worker, "generation-a", 1)
    _publish(protocol, controller, worker, "generation-b", 2)
    _stage(protocol, controller, worker, "generation-c", 2)
    with pytest.raises(PublishConflict, match="monotonically"):
        protocol.publish(controller, "generation-c")
    recovered = protocol.publish(controller, "generation-c", allow_recovery=True)
    assert protocol.select_latest().chosen == recovered
    assert [item.generation_id for item in protocol.published_candidates()] == [
        "generation-c", "generation-b", "generation-a"
    ]
    assert protocol.publish(controller, "generation-c", allow_recovery=True) == recovered
    _publish(protocol, controller, worker, "generation-d", 3)
    assert protocol.select_latest().chosen.generation_id == "generation-d"


def test_manifest_mutation_and_all_invalid_candidates_fail_closed():
    store, _, _, protocol, controller, worker = _setup()
    _publish(protocol, controller, worker, "generation-a", 10)
    _publish(protocol, controller, worker, "generation-b", 20)
    manifest_key = protocol.manifest_key("generation-b")
    original = store.get(manifest_key)
    store.put(manifest_key, b"{}", if_match=original.etag)
    assert protocol.select_latest().chosen.generation_id == "generation-a"
    store.delete(protocol.manifest_key("generation-a"))
    selection = protocol.select_latest()
    assert selection.chosen is None
    assert [row.generation_id for row in selection.rejected] == ["generation-b", "generation-a"]
    head = store.get(protocol.head_key)
    store.put(protocol.head_key, b"broken", if_match=head.etag)
    with pytest.raises(InvalidRemoteCheckpoint, match="head cannot"):
        protocol.select_latest()


def test_damage_between_seal_and_publish_is_not_visible():
    store, _, _, protocol, controller, worker = _setup()
    _publish(protocol, controller, worker, "generation-a", 10)
    _stage(protocol, controller, worker, "generation-b", 20)
    payload_key = protocol.payload_key("generation-b", "rank-0.json")
    original = store.get(payload_key)
    store.put(payload_key, b"bad", if_match=original.etag)
    with pytest.raises(InvalidRemoteCheckpoint, match="payload hash"):
        protocol.publish(controller, "generation-b")
    assert protocol.select_latest().chosen.generation_id == "generation-a"


class LostResponseStore(InMemoryObjectStore):
    def __init__(self):
        super().__init__()
        self.lose_after_put: set[str] = set()
        self.lose_before_put: set[str] = set()

    def put(self, key, data, *, if_none_match=False, if_match=None):
        if key in self.lose_before_put:
            self.lose_before_put.remove(key)
            raise ResponseLost("request outcome is unknown")
        result = super().put(key, data, if_none_match=if_none_match, if_match=if_match)
        if key in self.lose_after_put:
            self.lose_after_put.remove(key)
            raise ResponseLost("response was lost after write")
        return result


def test_lost_write_responses_are_reconciled_by_readback():
    store = LostResponseStore()
    _, _, _, protocol, controller, worker = _setup(store)
    store.lose_after_put = {
        protocol.payload_key("generation-a", "dcp/.metadata"),
        protocol.manifest_key("generation-a"),
        protocol.head_key,
    }
    published = _publish(protocol, controller, worker, "generation-a", 10)
    assert published.generation_id == "generation-a"
    assert protocol.select_latest().chosen == published
    assert protocol.publish(controller, "generation-a") == published
    with pytest.raises(GenerationConflict, match="sealed"):
        protocol.write_payload(worker, "generation-a", "late", b"late")


def test_unapplied_lost_response_does_not_claim_publication():
    store = LostResponseStore()
    _, _, _, protocol, controller, worker = _setup(store)
    _stage(protocol, controller, worker, "generation-a", 10)
    store.lose_before_put.add(protocol.head_key)
    with pytest.raises(ResponseLost):
        protocol.publish(controller, "generation-a")
    assert protocol.select_latest().chosen is None
    assert protocol.publish(controller, "generation-a").global_step == 10


def test_lost_takeover_barrier_response_is_reconciled():
    store = LostResponseStore()
    _, oracle, authority, protocol, old_controller, old_worker = _setup(store)
    _publish(protocol, old_controller, old_worker, "generation-a", 10)
    oracle.confirm("run-one", old_controller.epoch)
    store.lose_after_put.add(protocol.head_key)
    new_controller = protocol.takeover("controller-b")
    with pytest.raises(FencedOut):
        protocol.write_payload(old_worker, "generation-stale", "weights", b"old")
    new_worker = authority.worker(new_controller, "worker-b")
    assert _publish(protocol, new_controller, new_worker, "generation-b", 20).epoch == 2


def test_lower_step_cannot_replace_newer_published_checkpoint():
    _, _, _, protocol, controller, worker = _setup()
    _stage(protocol, controller, worker, "generation-old", 10)
    _publish(protocol, controller, worker, "generation-new", 20)
    with pytest.raises(PublishConflict, match="monotonically"):
        protocol.publish(controller, "generation-old")
    assert protocol.select_latest().chosen.generation_id == "generation-new"


def test_competing_publications_have_one_winner_and_never_regress():
    class RacingStore(InMemoryObjectStore):
        def __init__(self):
            super().__init__()
            self.race = threading.Barrier(2)

        def put(self, key, data, *, if_none_match=False, if_match=None):
            if key.endswith("/HEAD") and if_none_match:
                self.race.wait(timeout=5)
            return super().put(key, data, if_none_match=if_none_match, if_match=if_match)

    store = RacingStore()
    _, _, _, protocol, controller, worker = _setup(store)
    _stage(protocol, controller, worker, "generation-a", 10)
    _stage(protocol, controller, worker, "generation-b", 20)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda gen: _publish_existing(protocol, controller, gen),
                ["generation-a", "generation-b"],
            )
        )
    assert sum(result == "published" for result in outcomes) == 1
    assert sum(result == "conflict" for result in outcomes) == 1
    winner = protocol.select_latest().chosen
    assert winner is not None
    if winner.global_step == 20:
        with pytest.raises(PublishConflict, match="monotonically"):
            protocol.publish(controller, "generation-a")


def _publish_existing(protocol, controller, generation):
    try:
        protocol.publish(controller, generation)
        return "published"
    except PublishConflict:
        return "conflict"


def test_takeover_needs_isolation_and_fences_old_actors():
    _, oracle, authority, protocol, old_controller, old_worker = _setup()
    _publish(protocol, old_controller, old_worker, "generation-a", 10)
    with pytest.raises(IsolationRequired):
        protocol.takeover("controller-b")
    oracle.confirm("run-one", old_controller.epoch)
    new_controller = protocol.takeover("controller-b")
    assert new_controller.epoch == old_controller.epoch + 1
    with pytest.raises(FencedOut):
        protocol.write_payload(old_worker, "generation-old", "weights", b"stale")
    with pytest.raises(FencedOut):
        protocol.publish(old_controller, "generation-a")
    new_worker = authority.worker(new_controller, "worker-b")
    _publish(protocol, new_controller, new_worker, "generation-b", 20)
    assert protocol.select_latest().chosen.generation_id == "generation-b"


def test_in_flight_old_head_write_loses_to_epoch_barrier():
    class DelayedOldPublishStore(InMemoryObjectStore):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.delay_old = False

        def put(self, key, data, *, if_none_match=False, if_match=None):
            if (
                self.delay_old
                and key.endswith("/HEAD")
                and if_match is not None
                and json.loads(data)["epoch"] == 1
            ):
                self.entered.set()
                assert self.release.wait(5)
            return super().put(key, data, if_none_match=if_none_match, if_match=if_match)

    store = DelayedOldPublishStore()
    _, oracle, authority, protocol, old_controller, old_worker = _setup(store)
    _publish(protocol, old_controller, old_worker, "generation-a", 10)
    _stage(protocol, old_controller, old_worker, "generation-stale", 20)
    store.delay_old = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(protocol.publish, old_controller, "generation-stale")
        assert store.entered.wait(5)
        oracle.confirm("run-one", old_controller.epoch)
        new_controller = protocol.takeover("controller-b")
        store.release.set()
        with pytest.raises(FencedOut, match="epoch changed"):
            pending.result(timeout=5)
    assert protocol.select_latest().chosen.generation_id == "generation-a"
    new_worker = authority.worker(new_controller, "worker-b")
    _publish(protocol, new_controller, new_worker, "generation-b", 30)
    assert protocol.select_latest().chosen.generation_id == "generation-b"


def test_in_flight_old_payload_is_rejected_and_remains_unpublished():
    class DelayedPayloadStore(InMemoryObjectStore):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def put(self, key, data, *, if_none_match=False, if_match=None):
            if key.endswith("generation-stale/payload/weights"):
                self.entered.set()
                assert self.release.wait(5)
            return super().put(key, data, if_none_match=if_none_match, if_match=if_match)

    store = DelayedPayloadStore()
    _, oracle, _, protocol, old_controller, old_worker = _setup(store)
    _publish(protocol, old_controller, old_worker, "generation-a", 10)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            protocol.write_payload, old_worker, "generation-stale", "weights", b"orphan"
        )
        assert store.entered.wait(5)
        oracle.confirm("run-one", old_controller.epoch)
        protocol.takeover("controller-b")
        store.release.set()
        with pytest.raises(FencedOut):
            pending.result(timeout=5)
    assert store.get(protocol.payload_key("generation-stale", "weights")) is not None
    assert protocol.select_latest().chosen.generation_id == "generation-a"


def test_foreign_identity_rejects_all_candidates():
    store, _, authority, protocol, controller, worker = _setup()
    _publish(protocol, controller, worker, "generation-a", 10)
    foreign = RemoteCheckpointProtocol(store, authority, "run-one", "different-identity")
    selection = foreign.select_latest()
    assert selection.chosen is None
    assert "identity" in selection.rejected[0].reason
