"""Cross-check completed run claims against the saved configuration and index."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from trainguard.config import ProjectConfig

RUNTIME_IDENTITY_FIELDS = (
    "source_sha256", "python", "torch", "versions", "installed_distributions",
    "platform", "device", "world_size", "storage", "storage_device",
    "cuda_available", "cuda_version", "cuda_device_count",
)


def measurement_sha256(value: Any) -> str | None:
    """Return a digest only for the controller's complete, finite timing record."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "method", "elapsed_seconds", "load_average_before", "load_average_after"
    } or value["method"] != "controller_monotonic":
        raise ValueError("original measurement has an invalid shape")
    duration = value["elapsed_seconds"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("original measurement duration is invalid")
    for field in ("load_average_before", "load_average_after"):
        loads = value[field]
        if not isinstance(loads, list) or len(loads) != 3 or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in loads
        ):
            raise ValueError(f"original measurement {field} is invalid")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def saved_completed_metadata_errors(status: Any, config: ProjectConfig) -> list[str]:
    """Check saved facts before any completed run may be certified again."""
    if not isinstance(status, dict):
        return ["run metadata is not a mapping"]
    errors: list[str] = []
    if type(status.get("run_schema_version")) is not int or status["run_schema_version"] != 2:
        errors.append("run schema is unsupported or missing")
    run_id = status.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        errors.append("run identity is invalid")
    if not isinstance(status.get("started_at"), str) or not status["started_at"]:
        errors.append("run start time is invalid")
    if _valid_time(status.get("started_at")) is None or _valid_time(
        status.get("finished_at")
    ) is None:
        errors.append("completed run timing is invalid")
    if not isinstance(status.get("attempt_id"), str) or re.fullmatch(
        r"attempt-[0-9]{3,}", status["attempt_id"]
    ) is None:
        errors.append("final attempt identity is invalid")
    if not config.matches_saved_config(status.get("config")) or status.get(
        "config_fingerprint"
    ) != config.fingerprint():
        errors.append("run configuration differs from saved configuration")
    if "execution_started_monotonic" in status or "execution_load_before" in status:
        errors.append("completed run retains an unfinished timing window")
    try:
        measurement_sha256(status.get("measurement"))
    except ValueError as exc:
        errors.append(str(exc))

    environment = status.get("environment")
    if not isinstance(environment, dict):
        return [*errors, "run environment is not a mapping"]
    if type(environment.get("world_size")) is not int or (
        environment["world_size"] != config.run.world_size
    ):
        errors.append("run environment world size differs from configuration")
    if environment.get("device") != config.run.device:
        errors.append("run environment device differs from configuration")
    if environment.get("storage") != "local filesystem":
        errors.append("run environment storage kind is invalid")
    if type(environment.get("storage_device")) is not int:
        errors.append("run environment storage device is invalid")
    if type(environment.get("cuda_available")) is not bool or type(
        environment.get("cuda_device_count")
    ) is not int or environment["cuda_device_count"] < 0:
        errors.append("run environment CUDA inventory is invalid")
    if environment.get("cuda_version") is not None and not isinstance(
        environment.get("cuda_version"), str
    ):
        errors.append("run environment CUDA version is invalid")
    for field in ("python", "torch", "platform"):
        if not isinstance(environment.get(field), str) or not environment[field]:
            errors.append(f"run environment {field} is invalid")
    digest = environment.get("source_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        errors.append("run environment source identity is invalid")
    if not isinstance(environment.get("versions"), dict) or not isinstance(
        environment.get("installed_distributions"), list
    ):
        errors.append("run environment dependency identity is invalid")
    return errors


def _valid_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def indexed_schema_version(run_dir: Path, run_id: Any) -> int | None:
    """Read the independent run schema marker, including after JSON version loss."""
    path = run_dir / "run.sqlite3"
    if not isinstance(run_id, str) or not path.is_file() or path.is_symlink():
        return None
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            row = database.execute(
                "SELECT evidence_schema_version FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row is not None and type(row[0]) is int else None


def completed_index_errors(run_dir: Path, status: dict, config: ProjectConfig) -> list[str]:
    """Read the index without creating a replacement for missing evidence."""
    path = run_dir / "run.sqlite3"
    if not path.is_file() or path.is_symlink():
        return ["run index is missing or unsafe"]
    run_id = status.get("run_id")
    if not isinstance(run_id, str):
        return ["run identity is invalid"]
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            database.execute("PRAGMA query_only=ON")
            run = database.execute(
                "SELECT status, config_fingerprint, started_at, evidence_schema_version, "
                "measurement_sha256 "
                "FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            attempts = database.execute(
                "SELECT attempt_id, number, status, exit_code, started_at, finished_at "
                "FROM attempts WHERE run_id=? ORDER BY number", (run_id,)
            ).fetchall()
    except sqlite3.Error as exc:
        return [f"run index is unreadable: {exc}"]
    errors: list[str] = []
    if run is None:
        return ["run index has no matching run"]
    if type(run[3]) is not int or run[3] != 2:
        errors.append("run index evidence schema differs from metadata")
    if run[0] != "SUCCEEDED" or run[1] != config.fingerprint() or run[2] != status.get(
        "started_at"
    ):
        errors.append("run index status or identity differs from metadata")
    if not attempts:
        errors.append("run index has no completed attempt")
    else:
        attempt_id, number, attempt_status, exit_code, started_at, finished_at = attempts[-1]
        if (
            not isinstance(attempt_id, str)
            or attempt_id != status.get("attempt_id")
        ):
            errors.append("final attempt index has an invalid identity")
        if (
            type(number) is not int
            or number != len(attempts)
            or attempt_status != "SUCCEEDED"
            or type(exit_code) is not int
            or exit_code != 0
        ):
            errors.append("final attempt index status differs from metadata")
        started, finished = _valid_time(started_at), _valid_time(finished_at)
        run_started = _valid_time(status.get("started_at"))
        run_finished = _valid_time(status.get("finished_at"))
        if (
            started is None or finished is None or run_started is None or run_finished is None
            or not run_started <= started <= finished <= run_finished
        ):
            errors.append("final attempt timing is invalid")
    try:
        expected_measurement = measurement_sha256(status.get("measurement"))
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if run[4] != expected_measurement:
            errors.append("original measurement differs from run index")
    if config.checkpoint.reference_store_path is not None:
        from trainguard.reference_backend import reference_final_errors

        errors.extend(reference_final_errors(run_dir, status, config))
    return errors


def trusted_measurement(run_dir: Path) -> dict | None:
    """Return original timing only when a separate index record still agrees."""
    try:
        status = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(status, dict) or status.get("status") != "SUCCEEDED" or not isinstance(
        status.get("run_id"), str
    ):
        return None
    value = status.get("measurement")
    try:
        digest = measurement_sha256(value)
    except ValueError:
        return None
    if digest is None:
        return None
    path = run_dir / "run.sqlite3"
    if not path.is_file() or path.is_symlink():
        return None
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            row = database.execute(
                "SELECT status, evidence_schema_version, measurement_sha256 "
                "FROM runs WHERE run_id=?",
                (status["run_id"],),
            ).fetchone()
    except sqlite3.Error:
        return None
    return value if row == ("SUCCEEDED", 2, digest) else None
