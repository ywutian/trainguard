"""Controller-owned, retryable retention of committed checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from trainguard.checkpoint import (
    CheckpointInvalid,
    _check_checkpoint_root,
    ordered_candidates,
    validate_checkpoint,
)
from trainguard.config import ProjectConfig
from trainguard.events import sync_directory, write_json_atomic
from trainguard.restore_failures import failed_restore_candidates


def _tree_bytes(path: Path) -> int:
    """Count actual local files without following links into another namespace."""
    if path.is_symlink():
        raise ValueError("checkpoint tree contains a symbolic link")
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise ValueError("checkpoint tree contains an unsupported entry")
    total = 0
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise ValueError("checkpoint tree contains a symbolic link")
        if entry.is_file():
            total += entry.stat().st_size
        elif not entry.is_dir():
            raise ValueError("checkpoint tree contains an unsupported entry")
    return total


def prune_checkpoints(
    run_dir: Path, config: ProjectConfig, run_id: str, protected: set[Path] | None = None
) -> dict:
    keep = config.checkpoint.keep_last_k
    if keep is None:
        return {"enabled": False}
    _check_checkpoint_root(run_dir)
    root = (run_dir / "checkpoints").resolve()
    failed_restores = failed_restore_candidates(run_dir, run_id, config.run.world_size)
    protected_paths = {path.resolve() for path in (protected or set())}
    valid = []
    failed = []
    for path in ordered_candidates(run_dir):
        try:
            record = validate_checkpoint(
                path, config, run_id, decode_payload=True, require_trainable_state=True
            )
        except (CheckpointInvalid, OSError):
            continue
        if record.manifest_sha256 in failed_restores.explicit.get(path, set()):
            failed.append(record)
            continue
        if record.manifest_sha256 in failed_restores.incomplete.get(path, set()):
            protected_paths.add(path.resolve())
            continue
        valid.append(record)
    retained = valid[:keep]
    budget = config.checkpoint.max_retained_bytes

    def size(record):
        return sum(item["size"] for item in record.manifest["files"])

    while (
        budget is not None and len(retained) > 2 and sum(size(item) for item in retained) > budget
    ):
        retained.pop()
    protected_paths.update(item.path.resolve() for item in retained)
    journal_path = run_dir / "retention.json"
    previous = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    if previous and (
        not isinstance(previous, dict)
        or previous.get("run_id") != run_id
        or not isinstance(previous.get("pending"), list)
    ):
        raise ValueError("retention intent identity or schema is invalid")
    if previous.get("pending") and len(valid) < 2:
        previous["reason"] = "deletion paused until two valid fallback checkpoints exist"
        previous["valid_retained_count"] = len(valid)
        previous["budget_satisfied"] = False
        write_json_atomic(journal_path, previous)
        return previous
    targets = {
        str(item.path.resolve()) for item in valid if item.path.resolve() not in protected_paths
    }
    # Retry partially deleted directories from a persisted intent only after two fallbacks exist.
    if len(valid) >= 2:
        targets.update(str(item.path.resolve()) for item in failed)
        targets.update(previous.get("pending", []))
    safe = []
    for target in sorted(targets):
        path = Path(target)
        if (
            path.parent != root
            or path.is_symlink()
            or not re.fullmatch(r"step-\d+-attempt-\d+", path.name)
        ):
            raise ValueError("retention intent points outside an owned checkpoint directory")
        if path.resolve() not in protected_paths:
            safe.append(target)
    retained_bytes = sum(size(item) for item in valid if item.path.resolve() in protected_paths)
    valid_retained_count = sum(item.path.resolve() in protected_paths for item in valid)
    payload_budget_satisfied = budget is None or retained_bytes <= budget
    journal = {
        "enabled": True,
        "run_id": run_id,
        "pending": safe,
        "retained_bytes": retained_bytes,
        "valid_retained_count": valid_retained_count,
        "payload_budget_satisfied": payload_budget_satisfied,
        "budget_satisfied": payload_budget_satisfied,
    }
    write_json_atomic(journal_path, journal)
    for target in safe.copy():
        path = Path(target)
        if path.exists():
            shutil.rmtree(path)
            sync_directory(root)
        journal["pending"].remove(target)
        write_json_atomic(journal_path, journal)
    if config.run.profile == "guarded":
        valid_paths = {item.path.resolve() for item in valid if item.path.exists()}
        children = list(root.iterdir()) if root.is_dir() else []
        sizes = {child: _tree_bytes(child) for child in children}
        checkpoint_file_bytes = sum(sizes.values())
        unverified = {path: amount for path, amount in sizes.items()
                      if path.resolve() not in valid_paths}
        free_bytes = shutil.disk_usage(run_dir).free
        checkpoint_file_budget_satisfied = (
            budget is None or checkpoint_file_bytes <= budget
        )
        free_floor_satisfied = free_bytes >= config.checkpoint.min_free_bytes
        journal.update(
            checkpoint_file_bytes=checkpoint_file_bytes,
            unverified_candidate_count=len(unverified),
            unverified_candidate_bytes=sum(unverified.values()),
            free_bytes=free_bytes,
            checkpoint_file_budget_satisfied=checkpoint_file_budget_satisfied,
            free_floor_satisfied=free_floor_satisfied,
            budget_satisfied=(
                payload_budget_satisfied and checkpoint_file_budget_satisfied
                and free_floor_satisfied
            ),
        )
        write_json_atomic(journal_path, journal)
    return journal
