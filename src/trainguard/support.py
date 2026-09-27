"""Small, read-only diagnostic export with an explicit field allowlist."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from trainguard.config import load_config
from trainguard.events import write_json_atomic
from trainguard.records import summary_errors


class SupportBundleError(ValueError):
    """A run cannot be summarized safely from the available evidence."""


def _mapping(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SupportBundleError("run metadata is missing or unreadable") from exc
    if not isinstance(value, dict):
        raise SupportBundleError("run metadata has an invalid shape")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def build_support_bundle(run_dir: Path) -> dict:
    """Read only selected scalar facts; never copy logs, paths, samples or reasons."""
    status = _mapping(run_dir / "run.json")
    run_id = status.get("run_id")
    fingerprint = status.get("config_fingerprint")
    environment = status.get("environment")
    if (
        not isinstance(run_id, str)
        or not run_id
        or status.get("status") not in {"RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED"}
        or not _hex_digest(fingerprint)
        or not isinstance(environment, dict)
        or not _hex_digest(environment.get("source_sha256"))
    ):
        raise SupportBundleError("run identity is incomplete or invalid")

    database_path = run_dir / "run.sqlite3"
    if not database_path.is_file() or database_path.is_symlink():
        raise SupportBundleError("run index is missing or unsafe")
    try:
        with sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            identity = db.execute(
                "SELECT status, config_fingerprint FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if identity is None or identity != (status["status"], fingerprint):
                raise SupportBundleError("run index status or identity differs from metadata")
            attempts = db.execute(
                "SELECT attempt_id, number, status, resume_step, exit_code FROM attempts "
                "WHERE run_id=? ORDER BY number", (run_id,)
            ).fetchall()
            checkpoint_counts = db.execute(
                "SELECT status, COUNT(*) FROM checkpoints WHERE run_id=? GROUP BY status",
                (run_id,),
            ).fetchall()
            recoveries = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(recomputed_steps), 0) "
                "FROM recoveries WHERE run_id=?", (run_id,)
            ).fetchone()
    except sqlite3.Error as exc:
        raise SupportBundleError("run index is unreadable") from exc

    attempt_records = []
    for attempt_id, number, attempt_status, resume_step, exit_code in attempts:
        if (
            not isinstance(attempt_id, str)
            or not attempt_id
            or type(number) is not int
            or number < 1
            or attempt_status not in {"RUNNING", "SUCCEEDED", "FAILED", "INTERRUPTED"}
            or type(resume_step) is not int
            or resume_step < 0
            or (exit_code is not None and type(exit_code) is not int)
        ):
            raise SupportBundleError("attempt index contains invalid values")
        attempt_records.append(
            {"number": number, "status": attempt_status, "resume_step": resume_step,
             "exit_code": exit_code}
        )
    if any(kind not in {"VALID", "INVALID"} for kind, _ in checkpoint_counts):
        raise SupportBundleError("checkpoint index contains invalid values")
    if recoveries is None or any(type(value) is not int or value < 0 for value in recoveries):
        raise SupportBundleError("recovery index contains invalid values")

    completed_step = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = _mapping(summary_path)
        completed_step = summary.get("global_step")
        if (
            status["status"] != "SUCCEEDED"
            or not attempts
            or attempts[-1][2] != "SUCCEEDED"
            or summary.get("run_id") != run_id
            or summary.get("config_fingerprint") != fingerprint
            or summary.get("attempt_id") != attempts[-1][0]
            or status.get("attempt_id") != attempts[-1][0]
            or type(completed_step) is not int
            or completed_step < 0
        ):
            raise SupportBundleError("completion summary identity or status differs from run index")
        try:
            config = load_config(run_dir / "config.json")
        except (OSError, TypeError, ValueError) as exc:
            raise SupportBundleError("saved run configuration is unreadable") from exc
        if config.fingerprint() != fingerprint or summary_errors(
            summary, config, run_id, attempts[-1][0]
        ):
            raise SupportBundleError("completion summary differs from saved run configuration")
    elif status["status"] == "SUCCEEDED":
        raise SupportBundleError("successful run has no completion summary")

    return {
        "schema_version": 1,
        "data_policy": "allowlisted-summary; raw evidence stays in the run directory",
        "run_id_sha256": _digest(run_id),
        "status": status["status"],
        "config_fingerprint": fingerprint,
        "source_sha256": environment["source_sha256"],
        "completed_step": completed_step,
        "attempts": attempt_records,
        "checkpoints": {kind.lower(): count for kind, count in checkpoint_counts},
        "recovery_count": recoveries[0],
        "recomputed_steps": recoveries[1],
    }


def export_support_bundle(run_dir: Path, output: Path) -> None:
    if output.resolve().is_relative_to(run_dir.resolve()):
        raise SupportBundleError("diagnostic output must be outside the run directory")
    report = build_support_bundle(run_dir)
    write_json_atomic(output, report)
