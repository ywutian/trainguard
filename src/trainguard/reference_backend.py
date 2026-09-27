"""Checkpoint experiment that publishes local DCP commits through a selection head.

The store is either the same-host SQLite reference or the S3/DynamoDB adapter.
This is a bounded CPU experiment: neither backend here proves cross-host
isolation or power-loss durability. Only the controller handles store tokens;
training workers write the ordinary local DCP transaction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path

from trainguard import aws_store
from trainguard.aws_store import is_aws_location, parse_location
from trainguard.checkpoint import CheckpointInvalid, CheckpointRecord, validate_checkpoint
from trainguard.config import ProjectConfig
from trainguard.events import sync_directory
from trainguard.local_reference_store import LocalReferenceObjectStore
from trainguard.remote_protocol import (
    FencedOut,
    FencingToken,
    InvalidRemoteCheckpoint,
    PayloadDigest,
    PreconditionFailed,
    PublishedCheckpoint,
    RemoteCheckpointProtocol,
    RemoteProtocolError,
    ResponseLost,
)

MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_HEAD_BYTES = 64 * 1024
_ATTEMPT = re.compile(r"attempt-[0-9]{3,}\Z")


def backend_label(location: str) -> str:
    return (
        "aws_reference_experiment" if is_aws_location(location)
        else "same_host_reference_experiment"
    )


def canonical_reference(location: str | Path) -> str:
    """Resolve a local path; keep an AWS location exactly as configured."""
    if is_aws_location(location):
        return str(location)
    return str(Path(location).resolve())


def open_reference_store(location: str | Path, *, read_only: bool = False):
    if is_aws_location(location):
        return aws_store.connect(location, read_only=read_only)
    return LocalReferenceObjectStore(Path(location), read_only=read_only)


def check_reference_mode(
    config: ProjectConfig, run_dir: Path, database_path: str | Path,
) -> str | Path:
    """Reject configurations outside the deliberately narrow reference experiment."""
    if (
        config.run.profile != "experiment"
        or config.run.world_size != 2
        or config.run.backend != "gloo"
        or config.run.device != "cpu"
        or config.run.strategy != "ddp"
        or config.checkpoint.mode != "sync"
        or config.checkpoint.keep_last_k is not None
    ):
        raise ValueError(
            "local reference checkpoints require experiment, two CPU/Gloo DDP ranks, "
            "synchronous saves and no local retention policy"
        )
    if is_aws_location(database_path):
        return parse_location(database_path).uri
    path = Path(database_path).absolute()
    if path.is_symlink() or path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("reference database must be a regular path outside the run directory")
    if path.exists() and not path.is_file():
        raise ValueError("reference database path is not a regular file")
    if path.parent.exists():
        parent = path.parent.stat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or parent.st_mode & 0o077
        ):
            raise ValueError("reference database directory must be private")
    return path.resolve()


def _check_head_size(store, protocol: RemoteCheckpointProtocol) -> None:
    size = store.object_size(protocol.head_key)
    if size is not None and size > MAX_HEAD_BYTES:
        raise InvalidRemoteCheckpoint("reference HEAD exceeds experiment limit")


def _read_bounded_payloads(
    store,
    protocol: RemoteCheckpointProtocol,
    candidate: PublishedCheckpoint,
) -> dict[str, bytes]:
    _check_head_size(store, protocol)
    manifest_key = protocol.manifest_key(candidate.generation_id)
    manifest_size = store.object_size(manifest_key)
    if manifest_size is None or manifest_size > MAX_MANIFEST_BYTES:
        raise InvalidRemoteCheckpoint("reference manifest is missing or too large")
    stored_manifest = store.get(manifest_key)
    if stored_manifest is None:
        raise InvalidRemoteCheckpoint("reference manifest disappeared")
    try:
        outer_manifest = json.loads(stored_manifest.data)
        rows = outer_manifest["payloads"]
        if not isinstance(rows, list):
            raise TypeError("invalid payload list")
        declared_total = manifest_size
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("path"), str)
                or type(row.get("size")) is not int
                or row["size"] < 0
            ):
                raise TypeError("invalid payload size")
            declared_total += row["size"]
            if declared_total > MAX_CHECKPOINT_BYTES:
                raise InvalidRemoteCheckpoint("reference checkpoint exceeds experiment limit")
            key = protocol.payload_key(candidate.generation_id, row["path"])
            actual_size = store.object_size(key)
            if actual_size is None or actual_size > row["size"]:
                raise InvalidRemoteCheckpoint("reference payload is missing or oversized")
    except (KeyError, TypeError, UnicodeError, ValueError) as exc:
        raise InvalidRemoteCheckpoint("reference manifest size map is invalid") from exc
    return protocol.read_published_payloads(candidate)


def _local_size_preflight(path: Path) -> None:
    """Bound the tree before DCP metadata can cause a large payload decode."""
    if not path.is_dir() or path.is_symlink():
        raise CheckpointInvalid("checkpoint directory is missing or linked")
    total = 0
    for entry in path.rglob("*"):
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not (
            stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
        ):
            raise CheckpointInvalid("checkpoint tree contains an unsafe entry")
        if stat.S_ISREG(metadata.st_mode):
            total += metadata.st_size
            if total > MAX_CHECKPOINT_BYTES:
                raise CheckpointInvalid("local reference checkpoint exceeds experiment limit")


class LocalReferenceAuthority:
    """Persist the active token so a later same-host controller fences old tokens.

    Claiming a new epoch requires the caller to hold the run's exclusive lock and
    to have checked that no recorded worker remains. SQLite CAS handles stale
    claims; the run lock supplies local process isolation.
    """

    def __init__(self, store, run_id: str, identity: str, run_dir: Path) -> None:
        self.store = store
        self.run_id = run_id
        self.identity = identity
        self.key = f"runs/{run_id}/AUTHORITY"
        self.location = hashlib.sha256(
            (str(run_dir.resolve()) + "\x00" + store.location).encode()
        ).hexdigest()

    def _read(self) -> tuple[dict, str] | None:
        stored = self.store.get(self.key)
        if stored is None:
            return None
        try:
            record = json.loads(stored.data)
        except (UnicodeError, ValueError) as exc:
            raise FencedOut("stored authority cannot be decoded") from exc
        if (
            not isinstance(record, dict)
            or set(record) != {
                "run_id", "identity", "location", "epoch", "controller_id", "lease_id"
            }
            or record["run_id"] != self.run_id
            or record["identity"] != self.identity
            or record["location"] != self.location
            or type(record["epoch"]) is not int
            or record["epoch"] < 1
            or not isinstance(record["controller_id"], str)
            or not isinstance(record["lease_id"], str)
            or not record["controller_id"]
            or not record["lease_id"]
        ):
            raise FencedOut("stored authority identity is invalid")
        return record, stored.etag

    def claim(self, controller_id: str, *, resume: bool) -> FencingToken:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", controller_id) is None:
            raise ValueError("controller identity is invalid")
        current = self._read()
        if current is not None and not resume:
            raise FencedOut("fresh run already has an authority")
        epoch = 1 if current is None else current[0]["epoch"] + 1
        lease_id = secrets.token_hex(16)
        record = {
            "run_id": self.run_id,
            "identity": self.identity,
            "location": self.location,
            "epoch": epoch,
            "controller_id": controller_id,
            "lease_id": lease_id,
        }
        data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        try:
            if current is None:
                self.store.put(self.key, data, if_none_match=True)
            else:
                self.store.put(self.key, data, if_match=current[1])
        except (PreconditionFailed, ResponseLost) as exc:
            observed = self._read()
            if observed is None or observed[0] != record:
                raise FencedOut("authority claim outcome is unconfirmed") from exc
        return FencingToken(
            self.run_id, epoch, controller_id, controller_id, "controller", lease_id
        )

    def saved_epoch(self) -> int:
        current = self._read()
        if current is None:
            raise FencedOut("stored authority is missing")
        return current[0]["epoch"]

    def require(self, token: FencingToken, *, role: str) -> None:
        current = self._read()
        if (
            current is None
            or token.run_id != self.run_id
            or token.role != role
            or token.epoch != current[0]["epoch"]
            or token.controller_id != current[0]["controller_id"]
            or token.lease_id != current[0]["lease_id"]
            or (role == "controller" and token.actor_id != token.controller_id)
        ):
            raise FencedOut("actor has no active local reference authority")

    def worker(self, controller: FencingToken, worker_id: str) -> FencingToken:
        self.require(controller, role="controller")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", worker_id) is None:
            raise ValueError("worker identity is invalid")
        return FencingToken(
            controller.run_id, controller.epoch, controller.controller_id,
            worker_id, "worker", controller.lease_id,
        )


class LocalReferenceSession:
    """Controller-only adapter between committed local DCP trees and HEAD."""

    def __init__(
        self, run_dir: Path, location: str | Path, config: ProjectConfig, run_id: str,
        *, resume: bool,
    ) -> None:
        self.run_dir = run_dir
        self.location = check_reference_mode(config, run_dir, location)
        self.config = config
        self.run_id = run_id
        self.store = open_reference_store(self.location)
        self.authority = LocalReferenceAuthority(
            self.store, run_id, config.fingerprint(), run_dir
        )
        self.token = self.authority.claim(secrets.token_hex(12), resume=resume)
        self.protocol = RemoteCheckpointProtocol(
            self.store, self.authority, run_id, config.fingerprint(), history_limit=8
        )
        _check_head_size(self.store, self.protocol)
        self.protocol.synchronize_head_epoch(self.token)

    @property
    def cache_root(self) -> Path:
        return self.run_dir / "reference-cache"

    def _generation_id(self, record: CheckpointRecord) -> str:
        return (
            f"e{self.token.epoch}-s{record.global_step}-{record.attempt_id}-"
            f"{record.manifest_sha256[:16]}"
        )

    def publish_local(
        self, path: Path, *, recovery: bool = False
    ) -> PublishedCheckpoint:
        """Publish only after exact local bytes pass application validation."""
        _local_size_preflight(path)
        record = validate_checkpoint(
            path, self.config, self.run_id,
            decode_payload=True, require_trainable_state=True,
        )
        if path.parent != self.run_dir / "checkpoints":
            raise ValueError("only this run's local checkpoint can be published")
        generation_id = self._generation_id(record)
        files = [path / "manifest.json", path / "COMMITTED"]
        files.extend(path / item["path"] for item in record.manifest["files"])
        payloads: dict[str, bytes] = {}
        total = 0
        for entry in files:
            if not entry.is_file() or entry.is_symlink():
                raise CheckpointInvalid("checkpoint payload changed before upload")
            data = entry.read_bytes()
            total += len(data)
            if total > MAX_CHECKPOINT_BYTES:
                raise CheckpointInvalid("local reference checkpoint exceeds experiment limit")
            payloads[entry.relative_to(path).as_posix()] = data
        # Recheck after snapshotting; these are the bytes the store will receive.
        manifest_data = payloads["manifest.json"]
        if hashlib.sha256(manifest_data).hexdigest() != record.manifest_sha256:
            raise CheckpointInvalid("checkpoint manifest changed during upload")
        if payloads["COMMITTED"] != (record.manifest_sha256 + "\n").encode():
            raise CheckpointInvalid("checkpoint marker changed during upload")
        for item in record.manifest["files"]:
            data = payloads[item["path"]]
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise CheckpointInvalid("checkpoint payload changed during upload")
        expected: dict[str, PayloadDigest] = {}
        workers = (
            self.authority.worker(self.token, "upload-rank-0"),
            self.authority.worker(self.token, "upload-rank-1"),
        )
        for relative, data in sorted(payloads.items()):
            rank = 1 if relative == "rank-1.json" or relative.startswith("dcp/__1_") else 0
            expected[relative] = self.protocol.write_payload(
                workers[rank], generation_id, relative, data
            )
        self.protocol.seal(self.token, generation_id, record.global_step, expected)
        _check_head_size(self.store, self.protocol)
        published = self.protocol.publish(
            self.token, generation_id, allow_recovery=recovery
        )
        if published not in self.published_candidates():
            raise InvalidRemoteCheckpoint("published checkpoint is absent from HEAD")
        return published

    def published_candidates(self) -> tuple[PublishedCheckpoint, ...]:
        _check_head_size(self.store, self.protocol)
        candidates = self.protocol.published_candidates()
        if len(candidates) > self.protocol.history_limit:
            raise InvalidRemoteCheckpoint("reference HEAD has too many candidates")
        return candidates

    def materialize(self, candidate: PublishedCheckpoint) -> CheckpointRecord:
        payloads = _read_bounded_payloads(self.store, self.protocol, candidate)
        try:
            manifest = json.loads(payloads["manifest.json"])
            attempt_id = manifest["attempt_id"]
        except (KeyError, TypeError, UnicodeError, ValueError) as exc:
            raise InvalidRemoteCheckpoint("application manifest cannot be decoded") from exc
        if not isinstance(attempt_id, str) or _ATTEMPT.fullmatch(attempt_id) is None:
            raise InvalidRemoteCheckpoint("application attempt identity is invalid")
        checkpoint_name = f"step-{candidate.global_step:06d}-{attempt_id}"
        root = self.cache_root
        if root.is_symlink():
            raise InvalidRemoteCheckpoint("reference cache root is a symbolic link")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.stat().st_uid != os.getuid() or root.stat().st_mode & 0o077:
            raise InvalidRemoteCheckpoint("reference cache root is not private")
        temporary = Path(tempfile.mkdtemp(prefix=".stage-", dir=root))
        destination = root / candidate.generation_id
        try:
            staged = temporary / checkpoint_name
            for relative, data in payloads.items():
                target = staged / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            record = validate_checkpoint(
                staged, self.config, self.run_id,
                decode_payload=True, require_trainable_state=True,
            )
            if record.global_step != candidate.global_step:
                raise InvalidRemoteCheckpoint("application and HEAD steps differ")
            if destination.is_symlink():
                raise InvalidRemoteCheckpoint("reference cache generation is a symbolic link")
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(temporary, destination)
            sync_directory(root)
            return CheckpointRecord(
                destination / checkpoint_name, record.global_step, record.attempt_id,
                record.manifest, record.manifest_sha256,
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def reference_final_errors(run_dir: Path, status: dict, config: ProjectConfig) -> list[str]:
    """Check a completed run against existing HEAD bytes without claiming an epoch."""
    configured = config.checkpoint.reference_store_path
    if configured is None:
        return []
    try:
        if status.get("local_reference_store") != configured:
            raise InvalidRemoteCheckpoint("saved reference database identity differs")
        run_id = status["run_id"]
        if not isinstance(run_id, str):
            raise InvalidRemoteCheckpoint("run identity is invalid")
        location = check_reference_mode(config, run_dir, configured)
        store = open_reference_store(location, read_only=True)
        authority = LocalReferenceAuthority(store, run_id, config.fingerprint(), run_dir)
        protocol = RemoteCheckpointProtocol(
            store, authority, run_id, config.fingerprint(), history_limit=8
        )
        _check_head_size(store, protocol)
        candidates = protocol.published_candidates()
        if not candidates or len(candidates) > protocol.history_limit:
            raise InvalidRemoteCheckpoint("reference HEAD has no bounded final candidate")
        if protocol.head_epoch() != authority.saved_epoch():
            raise InvalidRemoteCheckpoint("reference HEAD and authority epochs differ")
        candidate = candidates[0]
        audit = status.get("post_run_audit")
        final = audit.get("final_checkpoint") if isinstance(audit, dict) else None
        if (
            not isinstance(audit, dict)
            or audit.get("status") != "PASSED"
            or audit.get("checkpoint_backend") != backend_label(configured)
            or not isinstance(final, dict)
            or set(final) != {
                "attempt_id", "global_step", "manifest_sha256", "generation_id",
                "protocol_manifest_sha256",
            }
            or final.get("attempt_id") != status.get("attempt_id")
            or final.get("global_step") != config.training.total_steps
            or final.get("global_step") != candidate.global_step
            or final.get("generation_id") != candidate.generation_id
            or final.get("protocol_manifest_sha256") != candidate.manifest_sha256
            or not isinstance(final.get("manifest_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", final["manifest_sha256"]) is None
        ):
            raise InvalidRemoteCheckpoint("final audit differs from reference HEAD")
        payloads = _read_bounded_payloads(store, protocol, candidate)
        manifest_bytes = payloads.get("manifest.json")
        if (
            manifest_bytes is None
            or hashlib.sha256(manifest_bytes).hexdigest() != final["manifest_sha256"]
            or payloads.get("COMMITTED") != (final["manifest_sha256"] + "\n").encode()
        ):
            raise InvalidRemoteCheckpoint("final application manifest differs")
        manifest = json.loads(manifest_bytes)
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_id") != run_id
            or manifest.get("attempt_id") != final["attempt_id"]
            or manifest.get("global_step") != final["global_step"]
            or manifest.get("config_fingerprint") != config.fingerprint()
            or manifest.get("data_fingerprint") != config.data_fingerprint()
            or manifest.get("world_size") != config.run.world_size
            or not isinstance(manifest.get("files"), list)
        ):
            raise InvalidRemoteCheckpoint("final application identity differs")
        listed = {"manifest.json", "COMMITTED"}
        for item in manifest["files"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"path", "size", "sha256"}
                or not isinstance(item["path"], str)
                or item["path"] in listed
                or type(item["size"]) is not int
                or item["size"] < 0
                or not isinstance(item["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            ):
                raise InvalidRemoteCheckpoint("final application file list is invalid")
            listed.add(item["path"])
            data = payloads.get(item["path"])
            if (
                data is None or len(data) != item["size"]
                or hashlib.sha256(data).hexdigest() != item["sha256"]
            ):
                raise InvalidRemoteCheckpoint("final application file differs")
        if set(payloads) != listed:
            raise InvalidRemoteCheckpoint("final application payload set differs")
    except (KeyError, OSError, ValueError, TypeError, sqlite3.Error, RemoteProtocolError) as exc:
        return [f"reference final publication is invalid: {exc}"]
    return []
