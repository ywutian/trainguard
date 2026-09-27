"""Per-process event records and atomic single-writer summaries."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def append_event(path: Path, *, max_bytes: int | None = None, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": utc_now(), **fields}
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "ab") as stream:
        os.fchmod(stream.fileno(), 0o600)
        if max_bytes is not None and os.fstat(stream.fileno()).st_size + len(payload) > max_bytes:
            raise OSError("event log byte budget is exhausted")
        stream.write(payload)


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_event_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
