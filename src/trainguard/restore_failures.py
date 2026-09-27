"""Durable evidence for checkpoints that failed or stalled during restore."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from trainguard.events import write_json_atomic


@dataclass
class RestoreFailures:
    explicit: dict[Path, set[str]]
    incomplete: dict[Path, set[str]]


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_record(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(record, dict):
        raise TypeError(f"{label} is not a mapping: {path}")
    return record


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


def _progress(
    run_dir: Path, run_id: str, attempt_id: str, checkpoint: str, world_size: int,
    expected_digest: str | None = None,
) -> tuple[str, list[int]] | None:
    digest = expected_digest
    incomplete = []
    missing = False
    for rank in range(world_size):
        path = run_dir / "attempts" / attempt_id / f"rank-{rank}-restore-progress.json"
        if not path.exists() and not path.is_symlink():
            missing = True
            continue
        record = _read_record(path, "restore progress evidence")
        if (
            set(record) != {
                "schema_version", "run_id", "attempt_id", "rank", "checkpoint_path",
                "manifest_sha256", "phase",
            }
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["run_id"] != run_id
            or record["attempt_id"] != attempt_id
            or type(record["rank"]) is not int
            or record["rank"] != rank
            or record["checkpoint_path"] != checkpoint
            or not _valid_digest(record["manifest_sha256"])
            or not isinstance(record["phase"], str)
            or record["phase"] not in {"restore_started", "payload_loaded", "state_loaded"}
            or (digest is not None and record["manifest_sha256"] != digest)
        ):
            raise ValueError(f"restore progress evidence identity is invalid: {path}")
        digest = record["manifest_sha256"]
        if record["phase"] == "restore_started":
            incomplete.append(rank)
    if missing or digest is None:
        return None
    return digest, incomplete


def record_restore_progress(
    run_dir: Path, run_id: str, attempt_id: str, rank: int,
    checkpoint: Path, manifest_sha256: str, phase: str,
) -> None:
    if not _valid_digest(manifest_sha256) or phase not in {
        "restore_started", "payload_loaded", "state_loaded"
    }:
        raise ValueError("restore progress identity or phase is invalid")
    write_json_atomic(
        run_dir / "attempts" / attempt_id / f"rank-{rank}-restore-progress.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "rank": rank,
            "checkpoint_path": str(checkpoint),
            "manifest_sha256": manifest_sha256,
            "phase": phase,
        },
    )


def record_group_ended(
    run_dir: Path, run_id: str, attempt_id: str, reason: str, observation: str
) -> None:
    if (
        not isinstance(reason, str)
        or not reason
        or not isinstance(observation, str)
        or observation not in {"controller_cleanup", "no_owned_workers"}
    ):
        raise ValueError("worker group observation is invalid")
    path = run_dir / "attempts" / attempt_id / "worker-group-ended.json"
    if path.exists() or path.is_symlink():
        _group_ended(run_dir, run_id, attempt_id)
        return
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "reason": reason,
            "observation": observation,
        },
    )


def _group_ended(run_dir: Path, run_id: str, attempt_id: str) -> str | None:
    path = run_dir / "attempts" / attempt_id / "worker-group-ended.json"
    if not path.exists() and not path.is_symlink():
        return None
    record = _read_record(path, "worker cleanup evidence")
    if (
        set(record) != {"schema_version", "run_id", "attempt_id", "reason", "observation"}
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["run_id"] != run_id
        or record["attempt_id"] != attempt_id
        or not isinstance(record["reason"], str)
        or not record["reason"]
        or not isinstance(record["observation"], str)
        or record["observation"] not in {"controller_cleanup", "no_owned_workers"}
    ):
        raise ValueError(f"worker cleanup evidence identity is invalid: {path}")
    return record["reason"]


def _eligible_end(reason: str | None) -> bool:
    return reason is not None and reason != "interrupted by user" and not reason.startswith(
        "controller "
    )


def record_restore_incomplete(
    run_dir: Path, run_id: str, attempt_id: str, checkpoint: Path,
    world_size: int, expected_digest: str | None = None,
) -> bool:
    """Persist a conservative verdict after a cleaned-up attempt ends inside restore."""
    reason = _group_ended(run_dir, run_id, attempt_id)
    if not _eligible_end(reason):
        return False
    progress = _progress(
        run_dir, run_id, attempt_id, str(checkpoint), world_size, expected_digest
    )
    if progress is None:
        return False
    digest, incomplete = progress
    if not incomplete:
        return False
    path = run_dir / "attempts" / attempt_id / "restore-incomplete.json"
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "checkpoint_path": str(checkpoint),
        "manifest_sha256": digest,
        "incomplete_ranks": incomplete,
    }
    if path.exists() or path.is_symlink():
        if _read_record(path, "restore incomplete verdict") != record:
            raise ValueError(f"restore incomplete verdict identity is invalid: {path}")
    else:
        write_json_atomic(path, record)
    return True


def failed_restore_candidates(
    run_dir: Path, run_id: str, world_size: int, attempts: list | None = None
) -> RestoreFailures:
    """Read strict, manifest-bound explicit and inferred failure evidence."""
    if attempts is None:
        attempts = _run_attempts(run_dir, run_id)
    failures = RestoreFailures({}, {})
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
            record = _read_record(path, "restore failure evidence")
            if (
                set(record) != {
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
                or not _valid_digest(record["manifest_sha256"])
                or not isinstance(record["error_type"], str)
                or not record["error_type"]
            ):
                raise ValueError(f"restore failure evidence identity is invalid: {path}")
            failures.explicit.setdefault(Path(checkpoint), set()).add(record["manifest_sha256"])
        path = run_dir / "attempts" / attempt_id / "restore-incomplete.json"
        if not path.exists() and not path.is_symlink():
            continue
        record = _read_record(path, "restore incomplete verdict")
        if (
            set(record) != {
                "schema_version", "run_id", "attempt_id", "checkpoint_path",
                "manifest_sha256", "incomplete_ranks",
            }
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["run_id"] != run_id
            or record["attempt_id"] != attempt_id
            or record["checkpoint_path"] != checkpoint
            or not _valid_digest(record["manifest_sha256"])
            or not isinstance(record["incomplete_ranks"], list)
            or not record["incomplete_ranks"]
            or any(type(rank) is not int for rank in record["incomplete_ranks"])
        ):
            raise ValueError(f"restore incomplete verdict identity is invalid: {path}")
        progress = _progress(
            run_dir, run_id, attempt_id, checkpoint, world_size,
            record["manifest_sha256"],
        )
        if (
            not _eligible_end(_group_ended(run_dir, run_id, attempt_id))
            or progress is None
            or progress[1] != record["incomplete_ranks"]
        ):
            raise ValueError(f"restore incomplete verdict lacks matching evidence: {path}")
        failures.incomplete.setdefault(Path(checkpoint), set()).add(record["manifest_sha256"])
    return failures
