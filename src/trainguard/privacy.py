"""Customer-keyed sample evidence for guarded runs."""

from __future__ import annotations

import hmac
import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

SAMPLE_KEY_FILE_ENV = "TRAINGUARD_SAMPLE_HMAC_KEY_FILE"
_SAMPLE_EVENTS = {"batch_consumed", "step_completed", "update_skipped"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _mac(key: bytes, domain: bytes, value: Any) -> str:
    return hmac.new(key, domain + _canonical(value), sha256).hexdigest()


def sample_key_id(key: bytes) -> str:
    return hmac.new(key, b"trainguard-sample-key-id-v1", sha256).hexdigest()


def load_sample_key(run_dir: Path | None = None, expected_id: str | None = None) -> bytes:
    """Read a customer-held key outside the run; never persist its path or bytes."""
    value = os.environ.get(SAMPLE_KEY_FILE_ENV)
    if not value:
        raise ValueError("guarded sample commitment key file is required")
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("guarded sample commitment key path must be absolute and not a link")
    try:
        resolved = path.resolve(strict=True)
        if run_dir is not None and resolved.is_relative_to(run_dir.resolve()):
            raise ValueError("guarded sample commitment key must be outside the run directory")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as stream:
            details = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_uid != os.getuid()
                or details.st_mode & 0o077
            ):
                raise ValueError("guarded sample commitment key file must be private")
            key = stream.read(4097)
    except OSError as exc:
        raise ValueError("guarded sample commitment key file is unreadable") from exc
    if not 32 <= len(key) <= 4096:
        raise ValueError("guarded sample commitment key must contain 32 to 4096 bytes")
    if expected_id is not None and not hmac.compare_digest(sample_key_id(key), expected_id):
        raise ValueError("guarded sample commitment key differs from the run identity")
    return key


def key_for_run(run_dir: Path) -> bytes:
    try:
        status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("guarded run identity is unreadable") from exc
    expected_id = status.get("sample_key_id") if isinstance(status, dict) else None
    if (
        not isinstance(expected_id, str)
        or len(expected_id) != 64
        or any(character not in "0123456789abcdef" for character in expected_id)
    ):
        raise ValueError("guarded sample commitment key identity is missing")
    return load_sample_key(run_dir, expected_id)


def protect_sample_event(fields: dict[str, Any], sample_ids: list[int], key: bytes) -> dict[str, Any]:
    if fields.get("event_type") not in _SAMPLE_EVENTS:
        raise ValueError("sample commitment event type is unsupported")
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or any(type(item) is not int or item < 0 for item in sample_ids)
    ):
        raise ValueError("sample IDs must be nonnegative integers")
    result = dict(fields)
    if "sample_ids" in result:
        raise ValueError("raw sample IDs cannot be written to guarded events")
    stable_context = {
        field: result.get(field)
        for field in ("rank", "event_type", "global_step", "consumed_batches")
    }
    result["sample_count"] = len(sample_ids)
    result["sample_commitment"] = _mac(
        key, b"trainguard-ordered-samples-v1\0", [stable_context, sample_ids]
    )
    result["sample_evidence_mac"] = _mac(
        key, b"trainguard-sample-event-v1\0", result
    )
    return result


def verified_sample_event(event: dict[str, Any], key: bytes) -> tuple[int, str]:
    count = event.get("sample_count")
    commitment = event.get("sample_commitment")
    supplied_mac = event.get("sample_evidence_mac")
    if (
        event.get("event_type") not in _SAMPLE_EVENTS
        or "sample_ids" in event
        or type(count) is not int
        or count < 1
        or not isinstance(commitment, str)
        or len(commitment) != 64
        or any(character not in "0123456789abcdef" for character in commitment)
        or not isinstance(supplied_mac, str)
        or len(supplied_mac) != 64
    ):
        raise ValueError("guarded sample commitment is incomplete or exposes raw IDs")
    signed = {field: value for field, value in event.items() if field not in {"time", "sample_evidence_mac"}}
    try:
        expected = _mac(key, b"trainguard-sample-event-v1\0", signed)
    except (TypeError, ValueError) as exc:
        raise ValueError("guarded sample evidence is invalid") from exc
    if not hmac.compare_digest(expected, supplied_mac):
        raise ValueError("guarded sample evidence MAC differs")
    return count, commitment
