"""Conservative local capacity gates for guarded checkpoint publication."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
import torch.distributed as dist

from trainguard.config import ProjectConfig


def event_log_limit(config: ProjectConfig) -> int | None:
    return config.checkpoint.max_event_log_bytes if config.run.profile == "guarded" else None


def require_save_capacity(
    config: ProjectConfig, run_dir: Path, attempt_id: str, rank: int,
    process_group: dist.ProcessGroup,
) -> None:
    """Make every rank reject before creating a candidate if any rank lacks reserve."""
    if config.run.profile != "guarded":
        return
    checkpoint = config.checkpoint
    assert checkpoint.max_checkpoint_bytes is not None
    assert checkpoint.min_free_bytes is not None
    assert checkpoint.max_event_log_bytes is not None
    local_error = None
    try:
        free_bytes = shutil.disk_usage(run_dir).free
        # Reserve one whole candidate and a full log per rank and controller.
        required = (
            checkpoint.max_checkpoint_bytes
            + checkpoint.min_free_bytes
            + (config.run.world_size + 1) * checkpoint.max_event_log_bytes
        )
        if free_bytes < required:
            local_error = "checkpoint capacity reserve is unavailable"
        event_path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
        if event_path.exists() and event_path.stat().st_size >= checkpoint.max_event_log_bytes:
            local_error = "rank event log byte budget is exhausted"
    except OSError as exc:
        local_error = f"checkpoint capacity inspection failed: {type(exc).__name__}"
    failed = torch.tensor([int(local_error is not None)], dtype=torch.int64)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM, group=process_group)
    if failed.item():
        raise RuntimeError(local_error or "another rank rejected checkpoint capacity")
