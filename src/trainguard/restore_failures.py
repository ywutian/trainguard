"""Durable evidence for checkpoints that failed during worker restore."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path


def _run_attempts(run_dir: Path, run_id: str) -> list[sqlite3.Row]:
    database_path = run_dir / "run.sqlite3"
    if not database_path.exists() and not database_path.is_symlink():
        return []
    if database_path.is_symlink() or not database_path.is_file():
        raise ValueError("run database is not a regular file")
    with closing(sqlite3.connect(
        database_path.absolute().as_uri() + "?mode=ro", uri=True
    )) as database:
        database.row_factory = sqlite3.Row
        return list(database.execute(
            "SELECT attempt_id, resume_checkpoint FROM attempts WHERE run_id=? ORDER BY number",
            (run_id,),
        ))


def failed_restore_candidates(
    run_dir: Path, run_id: str, world_size: int, attempts: list | None = None
) -> dict[Path, set[str]]:
    """Map resumed checkpoint paths to manifest digests rejected by a worker."""
    if attempts is None:
        attempts = _run_attempts(run_dir, run_id)
    failed: dict[Path, set[str]] = {}
    for attempt in attempts:
        checkpoint = attempt["resume_checkpoint"]
        if checkpoint is None:
            continue
        attempt_id = attempt["attempt_id"]
        if not isinstance(attempt_id, str) or re.fullmatch(r"attempt-\d{3,}", attempt_id) is None:
            raise ValueError("restore attempt identity is invalid")
        if not isinstance(checkpoint, str):
            raise TypeError("restore checkpoint path is invalid")
        for rank in range(world_size):
            path = run_dir / "attempts" / attempt_id / f"rank-{rank}-restore-failure.json"
            if not path.exists() and not path.is_symlink():
                continue
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"restore failure evidence is not a regular file: {path}")
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeError) as exc:
                raise ValueError(f"restore failure evidence is unreadable: {path}") from exc
            if (
                not isinstance(record, dict)
                or set(record) != {
                    "schema_version", "run_id", "attempt_id", "rank", "checkpoint_path",
                    "manifest_sha256", "error_type",
                }
                or type(record["schema_version"]) is not int
                or record["schema_version"] != 1
                or record["run_id"] != run_id
                or record["attempt_id"] != attempt_id
                or type(record["rank"]) is not int
                or record["rank"] != rank
                or record["checkpoint_path"] != checkpoint
                or not isinstance(record["manifest_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", record["manifest_sha256"]) is None
                or not isinstance(record["error_type"], str)
                or not record["error_type"]
            ):
                raise ValueError(f"restore failure evidence identity is invalid: {path}")
            failed.setdefault(Path(checkpoint), set()).add(record["manifest_sha256"])
    return failed
