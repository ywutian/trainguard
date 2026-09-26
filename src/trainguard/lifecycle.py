"""Controller-owned, retryable retention of committed checkpoints."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from trainguard.checkpoint import CheckpointInvalid, ordered_candidates, validate_checkpoint
from trainguard.config import ProjectConfig
from trainguard.events import sync_directory, write_json_atomic


def prune_checkpoints(
    run_dir: Path, config: ProjectConfig, run_id: str, protected: set[Path] | None = None
) -> dict:
    keep = config.checkpoint.keep_last_k
    if keep is None:
        return {"enabled": False}
    root = (run_dir / "checkpoints").resolve()
    protected_paths = {path.resolve() for path in (protected or set())}
    valid = []
    for path in ordered_candidates(run_dir):
        try:
            valid.append(validate_checkpoint(path, config, run_id))
        except (CheckpointInvalid, OSError):
            continue
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
        write_json_atomic(journal_path, previous)
        return previous
    targets = {
        str(item.path.resolve()) for item in valid if item.path.resolve() not in protected_paths
    }
    # Retry partially deleted directories from a persisted intent only after two fallbacks exist.
    if len(valid) >= 2:
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
    journal = {
        "enabled": True,
        "run_id": run_id,
        "pending": safe,
        "retained_bytes": retained_bytes,
        "budget_satisfied": budget is None or retained_bytes <= budget,
    }
    write_json_atomic(journal_path, journal)
    for target in safe.copy():
        path = Path(target)
        if path.exists():
            shutil.rmtree(path)
            sync_directory(root)
        journal["pending"].remove(target)
        write_json_atomic(journal_path, journal)
    return journal
