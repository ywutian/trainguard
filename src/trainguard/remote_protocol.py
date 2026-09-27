"""Executable object-store and fencing contract for checkpoint recovery.

This module is an isolated protocol model. It does not connect the training loop to a
remote service. A deployment must provide atomic conditional writes, strongly
consistent reads and listings, an independent isolation authority, and a tested
adapter for its storage service before relying on these guarantees. Delayed writes
from an old epoch may leave orphan objects; readers follow only the fenced head.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol


class RemoteProtocolError(RuntimeError):
    """A remote checkpoint operation could not complete safely."""


class PreconditionFailed(RemoteProtocolError):
    """A conditional object write or deletion did not match its precondition."""


class ResponseLost(RemoteProtocolError):
    """The caller cannot tell from the response whether a write took effect."""


class GenerationConflict(RemoteProtocolError):
    """An immutable generation key already contains different bytes."""


class PublishConflict(RemoteProtocolError):
    """The head advanced or the proposed step is no longer publishable."""


class FencedOut(RemoteProtocolError):
    """An actor's epoch is no longer permitted to write or publish."""


class IsolationRequired(RemoteProtocolError):
    """The previous epoch has not been independently confirmed isolated."""


class InvalidRemoteCheckpoint(RemoteProtocolError):
    """A remote checkpoint or selection head failed validation."""


@dataclass(frozen=True)
class StoredObject:
    data: bytes
    etag: str


class ObjectStore(Protocol):
    """Conditional, atomic object operations with read-after-write consistency.

    ``etag`` must identify a particular revision for compare-and-swap; it must
    not be reused when an object changes and later returns to the same bytes.
    Exactly one precondition is required for put. The model never overwrites
    generation objects; only the run head is updated with ``if_match``.
    """

    def get(self, key: str) -> StoredObject | None: ...

    def list(self, prefix: str) -> tuple[str, ...]: ...

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> StoredObject: ...

    def delete(self, key: str, *, if_match: str | None = None) -> bool: ...


class InMemoryObjectStore:
    """Strongly consistent store used for deterministic protocol experiments."""

    def __init__(self) -> None:
        self._objects: dict[str, StoredObject] = {}
        self._revision = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> StoredObject | None:
        with self._lock:
            return self._objects.get(key)

    def list(self, prefix: str) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(key for key in self._objects if key.startswith(prefix)))

    def put(
        self,
        key: str,
        data: bytes,
        *,
        if_none_match: bool = False,
        if_match: str | None = None,
    ) -> StoredObject:
        if if_none_match == (if_match is not None):
            raise ValueError("put requires exactly one conditional precondition")
        with self._lock:
            current = self._objects.get(key)
            if if_none_match and current is not None:
                raise PreconditionFailed(f"object already exists: {key}")
            if if_match is not None and (current is None or current.etag != if_match):
                raise PreconditionFailed(f"object revision changed: {key}")
            self._revision += 1
            stored = StoredObject(bytes(data), f"revision-{self._revision}")
            self._objects[key] = stored
            return stored

    def delete(self, key: str, *, if_match: str | None = None) -> bool:
        with self._lock:
            current = self._objects.get(key)
            if current is None:
                return False
            if if_match is not None and current.etag != if_match:
                raise PreconditionFailed(f"object revision changed: {key}")
            del self._objects[key]
            return True


class IsolationOracle(Protocol):
    """Independent evidence that every actor from an epoch has been isolated."""

    def is_isolated(self, run_id: str, epoch: int) -> bool: ...


class InMemoryIsolationOracle:
    """Manually confirmed isolation facts for protocol tests."""

    def __init__(self) -> None:
        self._confirmed: set[tuple[str, int]] = set()
        self._lock = threading.RLock()

    def confirm(self, run_id: str, epoch: int) -> None:
        with self._lock:
            self._confirmed.add((run_id, epoch))

    def is_isolated(self, run_id: str, epoch: int) -> bool:
        with self._lock:
            return (run_id, epoch) in self._confirmed


@dataclass(frozen=True)
class FencingToken:
    run_id: str
    epoch: int
    controller_id: str
    actor_id: str
    role: Literal["controller", "worker"]
    lease_id: str


class EpochAuthority:
    """Independent epoch authority; takeover is activated after a head CAS barrier.

    Isolation is supplied by an external oracle. This in-memory authority models
    the decisions but is not itself a durable multi-node coordinator.
    """

    def __init__(self, isolation: IsolationOracle) -> None:
        self._isolation = isolation
        self._active: dict[str, FencingToken] = {}
        self._pending: dict[str, FencingToken] = {}
        self._lock = threading.RLock()

    def start(self, run_id: str, controller_id: str) -> FencingToken:
        _name(run_id)
        _name(controller_id)
        with self._lock:
            if run_id in self._active:
                raise PublishConflict("run already has an active controller")
            token = FencingToken(
                run_id, 1, controller_id, controller_id, "controller", secrets.token_hex(16)
            )
            self._active[run_id] = token
            return token

    def worker(self, controller: FencingToken, worker_id: str) -> FencingToken:
        _name(worker_id)
        self.require(controller, role="controller")
        return FencingToken(
            controller.run_id,
            controller.epoch,
            controller.controller_id,
            worker_id,
            "worker",
            controller.lease_id,
        )

    def require(self, token: FencingToken, *, role: Literal["controller", "worker"]) -> None:
        with self._lock:
            active = self._active.get(token.run_id)
            if (
                active is None
                or token.role != role
                or token.epoch != active.epoch
                or token.controller_id != active.controller_id
                or token.lease_id != active.lease_id
                or (role == "controller" and token.actor_id != active.actor_id)
            ):
                raise FencedOut("actor has no active authority for this epoch")

    def prepare_takeover(self, run_id: str, controller_id: str) -> FencingToken:
        _name(controller_id)
        with self._lock:
            active = self._active.get(run_id)
            if active is None:
                raise FencedOut("run has no active epoch")
            if not self._isolation.is_isolated(run_id, active.epoch):
                raise IsolationRequired("previous epoch is not confirmed isolated")
            pending = self._pending.get(run_id)
            if pending is not None:
                if pending.controller_id != controller_id:
                    raise PublishConflict("another takeover is already pending")
                return pending
            if controller_id == active.controller_id:
                raise PublishConflict("takeover requires a new controller identity")
            pending = FencingToken(
                run_id,
                active.epoch + 1,
                controller_id,
                controller_id,
                "controller",
                secrets.token_hex(16),
            )
            self._pending[run_id] = pending
            return pending

    def activate_takeover(self, token: FencingToken) -> None:
        with self._lock:
            if self._pending.get(token.run_id) != token:
                raise FencedOut("takeover token is not pending")
            self._active[token.run_id] = token
            del self._pending[token.run_id]


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _name(value: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError("identifier must be a safe, nonempty path component")
    return value


def _path(value: str) -> str:
    if (
        not isinstance(value, str)
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("payload path must be a safe relative POSIX path")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class PayloadDigest:
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.size) is not int
            or self.size < 0
            or not isinstance(self.sha256, str)
            or not _SHA.fullmatch(self.sha256)
        ):
            raise ValueError("invalid payload digest")

    @classmethod
    def of(cls, data: bytes) -> PayloadDigest:
        return cls(len(data), _sha256(data))


@dataclass(frozen=True)
class PublishedCheckpoint:
    generation_id: str
    global_step: int
    epoch: int
    manifest_sha256: str


@dataclass(frozen=True)
class RejectedCandidate:
    generation_id: str
    reason: str


@dataclass(frozen=True)
class Selection:
    chosen: PublishedCheckpoint | None
    rejected: tuple[RejectedCandidate, ...]


class RemoteCheckpointProtocol:
    """Model immutable generations and one conditional publication head per run."""

    def __init__(
        self,
        store: ObjectStore,
        authority: EpochAuthority,
        run_id: str,
        identity: str,
        *,
        history_limit: int = 3,
    ) -> None:
        self.store = store
        self.authority = authority
        self.run_id = _name(run_id)
        if not isinstance(identity, str) or not identity:
            raise ValueError("identity is required")
        self.identity = identity
        if type(history_limit) is not int or history_limit < 2:
            raise ValueError("history_limit must retain at least two candidates")
        self.history_limit = history_limit
        self._root = f"runs/{run_id}"
        self._head_key = f"{self._root}/HEAD"

    def _generation_root(self, generation_id: str) -> str:
        return f"{self._root}/generations/{_name(generation_id)}"

    def payload_key(self, generation_id: str, path: str) -> str:
        return f"{self._generation_root(generation_id)}/payload/{_path(path)}"

    def manifest_key(self, generation_id: str) -> str:
        return f"{self._generation_root(generation_id)}/manifest.json"

    @property
    def head_key(self) -> str:
        return self._head_key

    def _immutable_put(self, key: str, data: bytes) -> StoredObject:
        try:
            return self.store.put(key, data, if_none_match=True)
        except (PreconditionFailed, ResponseLost) as exc:
            observed = self.store.get(key)
            if observed is not None and observed.data == data:
                return observed
            if isinstance(exc, ResponseLost) and observed is None:
                raise
            raise GenerationConflict(f"immutable object differs: {key}") from exc

    def write_payload(
        self, token: FencingToken, generation_id: str, path: str, data: bytes
    ) -> PayloadDigest:
        self.authority.require(token, role="worker")
        if token.run_id != self.run_id:
            raise FencedOut("token belongs to another run")
        if self.store.get(self.manifest_key(generation_id)) is not None:
            raise GenerationConflict("generation is already sealed")
        self._immutable_put(self.payload_key(generation_id, path), data)
        self.authority.require(token, role="worker")
        return PayloadDigest.of(data)

    def seal(
        self,
        token: FencingToken,
        generation_id: str,
        global_step: int,
        expected: Mapping[str, PayloadDigest],
    ) -> str:
        self.authority.require(token, role="controller")
        if token.run_id != self.run_id:
            raise FencedOut("token belongs to another run")
        if type(global_step) is not int or global_step < 0 or not expected:
            raise ValueError("seal requires a nonnegative step and payloads")
        prefix = f"{self._generation_root(generation_id)}/payload/"
        entries = []
        for path, digest in sorted(expected.items()):
            _path(path)
            if not isinstance(digest, PayloadDigest):
                raise TypeError("expected payload records must be PayloadDigest values")
            object_key = self.payload_key(generation_id, path)
            stored = self.store.get(object_key)
            if (
                stored is None
                or len(stored.data) != digest.size
                or _sha256(stored.data) != digest.sha256
            ):
                raise InvalidRemoteCheckpoint(f"missing or damaged payload: {path}")
            entries.append({"path": path, "size": digest.size, "sha256": digest.sha256})
        if set(self.store.list(prefix)) != {self.payload_key(generation_id, p) for p in expected}:
            raise InvalidRemoteCheckpoint("generation payload set differs from expected")
        self.authority.require(token, role="controller")
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "identity": self.identity,
            "generation_id": generation_id,
            "controller_id": token.controller_id,
            "epoch": token.epoch,
            "global_step": global_step,
            "payloads": entries,
        }
        data = _encode(manifest)
        self._immutable_put(self.manifest_key(generation_id), data)
        self.authority.require(token, role="controller")
        return _sha256(data)

    def _read_manifest(
        self, generation_id: str, expected_hash: str | None = None
    ) -> tuple[dict, str]:
        stored = self.store.get(self.manifest_key(generation_id))
        if stored is None:
            raise InvalidRemoteCheckpoint("manifest is missing or its digest differs")
        manifest_hash = _sha256(stored.data)
        if expected_hash is not None and manifest_hash != expected_hash:
            raise InvalidRemoteCheckpoint("manifest is missing or its digest differs")
        try:
            manifest = json.loads(stored.data)
        except (UnicodeDecodeError, ValueError) as exc:
            raise InvalidRemoteCheckpoint("manifest cannot be decoded") from exc
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "run_id",
            "identity",
            "generation_id",
            "controller_id",
            "epoch",
            "global_step",
            "payloads",
        }:
            raise InvalidRemoteCheckpoint("manifest structure differs")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 1
            or manifest["run_id"] != self.run_id
            or manifest["identity"] != self.identity
            or manifest["generation_id"] != generation_id
            or type(manifest["epoch"]) is not int
            or manifest["epoch"] < 1
            or type(manifest["global_step"]) is not int
            or manifest["global_step"] < 0
        ):
            raise InvalidRemoteCheckpoint("manifest identity or progress differs")
        try:
            _name(manifest["controller_id"])
        except ValueError as exc:
            raise InvalidRemoteCheckpoint("manifest controller is invalid") from exc
        rows = manifest["payloads"]
        if not isinstance(rows, list) or not rows:
            raise InvalidRemoteCheckpoint("manifest has no payloads")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"path", "size", "sha256"}:
                raise InvalidRemoteCheckpoint("payload record structure differs")
            try:
                path = _path(row["path"])
                digest = PayloadDigest(row["size"], row["sha256"])
            except (ValueError, TypeError) as exc:
                raise InvalidRemoteCheckpoint("payload record is invalid") from exc
            if path in seen:
                raise InvalidRemoteCheckpoint("manifest repeats a payload path")
            seen.add(path)
            stored_payload = self.store.get(self.payload_key(generation_id, path))
            if (
                stored_payload is None
                or len(stored_payload.data) != digest.size
                or _sha256(stored_payload.data) != digest.sha256
            ):
                raise InvalidRemoteCheckpoint(f"payload hash or size differs: {path}")
        return manifest, manifest_hash

    def _head(self) -> tuple[dict, StoredObject] | None:
        stored = self.store.get(self._head_key)
        if stored is None:
            return None
        try:
            head = json.loads(stored.data)
        except (UnicodeDecodeError, ValueError) as exc:
            raise InvalidRemoteCheckpoint("run head cannot be decoded") from exc
        if (
            not isinstance(head, dict)
            or set(head) != {"schema_version", "epoch", "commits"}
            or type(head["schema_version"]) is not int
            or head["schema_version"] != 1
            or type(head["epoch"]) is not int
            or head["epoch"] < 1
            or not isinstance(head["commits"], list)
        ):
            raise InvalidRemoteCheckpoint("run head structure differs")
        previous_step = None
        seen = set()
        for row in head["commits"]:
            if not isinstance(row, dict) or set(row) != {
                "generation_id",
                "global_step",
                "epoch",
                "manifest_sha256",
            }:
                raise InvalidRemoteCheckpoint("head candidate structure differs")
            try:
                _name(row["generation_id"])
            except ValueError as exc:
                raise InvalidRemoteCheckpoint("head generation is invalid") from exc
            if (
                type(row["global_step"]) is not int
                or row["global_step"] < 0
                or type(row["epoch"]) is not int
                or row["epoch"] < 1
                or row["epoch"] > head["epoch"]
                or not isinstance(row["manifest_sha256"], str)
                or not _SHA.fullmatch(row["manifest_sha256"])
                or row["generation_id"] in seen
                or (previous_step is not None and row["global_step"] >= previous_step)
            ):
                raise InvalidRemoteCheckpoint("head candidate order or identity differs")
            seen.add(row["generation_id"])
            previous_step = row["global_step"]
        return head, stored

    def publish(self, token: FencingToken, generation_id: str) -> PublishedCheckpoint:
        self.authority.require(token, role="controller")
        if token.run_id != self.run_id:
            raise FencedOut("token belongs to another run")
        manifest, manifest_hash = self._read_manifest(generation_id)
        if manifest["epoch"] != token.epoch or manifest["controller_id"] != token.controller_id:
            raise FencedOut("generation was sealed by a different epoch or controller")
        proposed = {
            "generation_id": generation_id,
            "global_step": manifest["global_step"],
            "epoch": token.epoch,
            "manifest_sha256": manifest_hash,
        }
        current = self._head()
        if current is None:
            if token.epoch != 1:
                raise FencedOut("takeover epoch has no head barrier")
            previous: list[dict] = []
            old_etag = None
        else:
            head, stored = current
            if head["epoch"] != token.epoch:
                raise FencedOut("run head belongs to another epoch")
            previous = head["commits"]
            old_etag = stored.etag
            if proposed in previous:
                return PublishedCheckpoint(
                    generation_id, manifest["global_step"], token.epoch, manifest_hash
                )
            if previous and manifest["global_step"] <= previous[0]["global_step"]:
                raise PublishConflict("checkpoint step must advance monotonically")
        data = _encode(
            {
                "schema_version": 1,
                "epoch": token.epoch,
                "commits": [proposed, *previous][: self.history_limit],
            }
        )
        self.authority.require(token, role="controller")
        try:
            if old_etag is None:
                self.store.put(self._head_key, data, if_none_match=True)
            else:
                self.store.put(self._head_key, data, if_match=old_etag)
        except (PreconditionFailed, ResponseLost) as exc:
            observed = self._head()
            if observed is not None and proposed in observed[0]["commits"]:
                return PublishedCheckpoint(
                    generation_id, manifest["global_step"], token.epoch, manifest_hash
                )
            if observed is not None and observed[0]["epoch"] != token.epoch:
                raise FencedOut("epoch changed during publication") from exc
            if isinstance(exc, ResponseLost) and (
                (observed is None and old_etag is None)
                or (observed is not None and observed[1].etag == old_etag)
            ):
                raise
            raise PublishConflict("run head changed during publication") from exc
        return PublishedCheckpoint(
            generation_id, manifest["global_step"], token.epoch, manifest_hash
        )

    def takeover(self, controller_id: str) -> FencingToken:
        """Confirm isolation, then CAS the epoch barrier before activating its token."""
        token = self.authority.prepare_takeover(self.run_id, controller_id)
        for _ in range(8):
            current = self._head()
            if current is None:
                old_etag = None
                commits: list[dict] = []
                old_epoch = token.epoch - 1
            else:
                head, stored = current
                old_etag = stored.etag
                commits = head["commits"]
                old_epoch = head["epoch"]
            if old_epoch == token.epoch:
                self.authority.activate_takeover(token)
                return token
            if old_epoch != token.epoch - 1:
                raise FencedOut("head and authority epochs disagree")
            data = _encode({"schema_version": 1, "epoch": token.epoch, "commits": commits})
            try:
                if old_etag is None:
                    self.store.put(self._head_key, data, if_none_match=True)
                else:
                    self.store.put(self._head_key, data, if_match=old_etag)
            except (PreconditionFailed, ResponseLost):
                observed = self._head()
                if observed is not None and observed[0]["epoch"] == token.epoch:
                    self.authority.activate_takeover(token)
                    return token
                continue
            self.authority.activate_takeover(token)
            return token
        raise PublishConflict("epoch barrier could not be published")

    def select_latest(self) -> Selection:
        """Validate each published candidate in order and fall back on corruption."""
        current = self._head()
        if current is None:
            return Selection(None, ())
        rejected = []
        for row in current[0]["commits"]:
            generation_id = row["generation_id"]
            try:
                manifest, _ = self._read_manifest(generation_id, row["manifest_sha256"])
                if (
                    manifest["global_step"] != row["global_step"]
                    or manifest["epoch"] != row["epoch"]
                ):
                    raise InvalidRemoteCheckpoint("manifest and head progress differ")
            except InvalidRemoteCheckpoint as exc:
                rejected.append(RejectedCandidate(generation_id, str(exc)))
                continue
            return Selection(
                PublishedCheckpoint(
                    generation_id,
                    row["global_step"],
                    row["epoch"],
                    row["manifest_sha256"],
                ),
                tuple(rejected),
            )
        return Selection(None, tuple(rejected))
