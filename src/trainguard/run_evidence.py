"""Cross-check completed run claims against the saved configuration and index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from trainguard.checkpoint import (
    CheckpointInvalid,
    candidate_path,
    ordered_candidates,
    validate_checkpoint,
)
from trainguard.config import ProjectConfig
from trainguard.privacy import key_for_run
from trainguard.restore_failures import failed_restore_candidates

RUNTIME_IDENTITY_FIELDS = (
    "source_sha256", "python", "torch", "versions", "installed_distributions",
    "environment_options", "startup_identity_sha256",
    "platform", "device", "world_size", "storage", "storage_device",
    "cuda_available", "cuda_version", "cuda_device_count",
)

MAX_CHECKPOINT_AUDIT_ENTRIES = 100_000
DEFAULT_EXPERIMENT_CHECKPOINT_AUDIT_BYTES = 512 * 1024 * 1024


def runtime_identity_sha256(value: Any) -> str:
    """Digest only stable runtime identity fields from a saved environment snapshot."""
    if not isinstance(value, dict) or any(
        field not in value for field in RUNTIME_IDENTITY_FIELDS
    ):
        raise ValueError("run environment identity is incomplete")
    selected = {field: value[field] for field in RUNTIME_IDENTITY_FIELDS}
    try:
        payload = json.dumps(
            selected, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("run environment identity cannot be encoded") from exc
    return hashlib.sha256(payload).hexdigest()


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
    if status.get("local_reference_store") != config.checkpoint.reference_store_path:
        errors.append("saved local reference database identity differs")
    requires_experiment = (
        config.fault.kind != "none"
        or config.recovery.omit_state != "none"
        or config.checkpoint.reference_store_path is not None
    )
    if requires_experiment and status.get("experiment_authorized") is not True:
        errors.append("saved experiment authorization is missing")
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
    startup = environment.get("startup_identity_sha256")
    if not isinstance(startup, str) or re.fullmatch(r"[0-9a-f]{64}", startup) is None:
        errors.append("run environment startup identity is invalid")
    if not isinstance(environment.get("versions"), dict) or not isinstance(
        environment.get("installed_distributions"), list
    ):
        errors.append("run environment dependency identity is invalid")
    options = environment.get("environment_options")
    if not isinstance(options, dict) or set(options) != {
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG"
    } or any(value is not None and not isinstance(value, str) for value in options.values()):
        errors.append("run environment options identity is invalid")
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


def _bounded_checkpoint_tree_bytes(path: Path, maximum: int) -> int:
    """Count logical bytes before reading checkpoint payloads during an audit."""
    total = 0
    entries = 1
    pending = [path]
    while pending:
        current = pending.pop()
        details = current.lstat()
        if stat.S_ISREG(details.st_mode):
            total += details.st_size
            if total > maximum:
                raise ValueError("checkpoint audit byte budget is exceeded")
        elif stat.S_ISDIR(details.st_mode):
            with os.scandir(current) as children:
                for child in children:
                    entries += 1
                    if entries > MAX_CHECKPOINT_AUDIT_ENTRIES:
                        raise ValueError("checkpoint audit entry limit is exceeded")
                    pending.append(Path(child.path))
        else:
            raise ValueError("checkpoint tree contains a link or unsupported entry")
    return total


def _completed_checkpoint_errors(
    run_dir: Path, status: dict, config: ProjectConfig
) -> list[str]:
    audit = status.get("post_run_audit")
    if not isinstance(audit, dict):
        return ["completed run has no post-run audit"]
    audit_time = _valid_time(audit.get("checked_at"))
    started = _valid_time(status.get("started_at"))
    finished = _valid_time(status.get("finished_at"))
    if (
        audit.get("attempt_id") != status.get("attempt_id")
        or audit.get("checkpoint_mode") != config.checkpoint.mode
        or audit_time is None
        or started is None
        or finished is None
        or not started <= audit_time <= finished
    ):
        return ["post-run audit identity or timing is invalid"]
    if config.checkpoint.mode == "none":
        if audit.get("status") != "NOT_APPLICABLE" or "final_checkpoint" in audit:
            return ["checkpoint-free post-run audit status is invalid"]
        return []
    if audit.get("status") != "PASSED":
        return ["post-run checkpoint audit did not pass"]
    if config.checkpoint.keep_last_k is not None and (
        audit.get("retention_enabled") is not True
        or audit.get("budget_satisfied") is not True
        or audit.get("pending_deletions") != 0
    ):
        return ["post-run checkpoint retention audit did not pass"]
    final = audit.get("final_checkpoint")
    if (
        not isinstance(final, dict)
        or final.get("attempt_id") != status.get("attempt_id")
        or type(final.get("global_step")) is not int
        or final["global_step"] != config.training.total_steps
        or not isinstance(final.get("manifest_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", final["manifest_sha256"]) is None
    ):
        return ["post-run final checkpoint identity is invalid"]
    if config.run.profile == "guarded":
        try:
            key_for_run(run_dir)
        except ValueError as exc:
            return [f"guarded checkpoint evidence cannot be verified: {exc}"]
    if config.checkpoint.reference_store_path is not None:
        return []
    run_id = status.get("run_id")
    if not isinstance(run_id, str):
        return ["run identity is invalid"]
    try:
        if config.run.profile == "guarded":
            retained_budget = config.checkpoint.max_retained_bytes
            candidate_budget = config.checkpoint.max_checkpoint_bytes
            free_floor = config.checkpoint.min_free_bytes
            if retained_budget is None or candidate_budget is None or free_floor is None:
                return ["guarded checkpoint capacity limits are missing"]
            _bounded_checkpoint_tree_bytes(run_dir / "checkpoints", retained_budget)
            if shutil.disk_usage(run_dir).free < free_floor:
                return ["guarded checkpoint free-space floor is unsatisfied"]
            failures = failed_restore_candidates(run_dir, run_id, config.run.world_size)
            valid = []
            for path in ordered_candidates(run_dir):
                _bounded_checkpoint_tree_bytes(path, candidate_budget)
                try:
                    record = validate_checkpoint(
                        path, config, run_id, decode_payload=True,
                        require_trainable_state=True,
                    )
                except (CheckpointInvalid, OSError):
                    continue
                if record.manifest_sha256 in failures.explicit.get(path, set()) or (
                    record.manifest_sha256 in failures.incomplete.get(path, set())
                ):
                    continue
                valid.append(record)
                if len(valid) == 2:
                    break
            if len(valid) < 2:
                return ["fewer than two usable checkpoint candidates remain"]
            selected = valid[0]
        else:
            path = candidate_path(
                run_dir, final["attempt_id"], final["global_step"]
            )
            _bounded_checkpoint_tree_bytes(
                path,
                config.checkpoint.max_checkpoint_bytes
                or DEFAULT_EXPERIMENT_CHECKPOINT_AUDIT_BYTES,
            )
            selected = validate_checkpoint(
                path, config, run_id, decode_payload=True,
                require_trainable_state=True,
                expected_manifest_sha256=final["manifest_sha256"],
            )
    except (CheckpointInvalid, OSError, ValueError, TypeError, sqlite3.Error) as exc:
        return [f"completed checkpoint evidence is invalid: {exc}"]
    if (
        selected.attempt_id != final["attempt_id"]
        or selected.global_step != final["global_step"]
        or selected.manifest_sha256 != final["manifest_sha256"]
    ):
        return ["current checkpoint selection differs from post-run final checkpoint"]
    return []


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
            columns = {row[1] for row in database.execute("PRAGMA table_info(runs)")}
            if "environment_sha256" not in columns:
                return ["run index evidence schema predates runtime identity binding"]
            run = database.execute(
                "SELECT status, config_fingerprint, started_at, evidence_schema_version, "
                "measurement_sha256, environment_sha256 "
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
    if type(run[3]) is not int or run[3] != 3:
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
    try:
        expected_environment = runtime_identity_sha256(status.get("environment"))
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if run[5] != expected_environment:
            errors.append("run environment identity differs from run index")
    if not errors:
        errors.extend(_completed_checkpoint_errors(run_dir, status, config))
    if not errors and config.checkpoint.reference_store_path is not None:
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
    try:
        from trainguard.config import load_config

        config = load_config(run_dir / "config.json")
    except (OSError, ValueError, TypeError):
        return None
    if saved_completed_metadata_errors(status, config) or completed_index_errors(
        run_dir, status, config
    ):
        return None
    value = status.get("measurement")
    return value if measurement_sha256(value) is not None else None
