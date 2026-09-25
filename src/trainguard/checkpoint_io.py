"""Distributed save and load operations at a completed optimizer update."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from trainguard.checkpoint import (
    candidate_path,
    capture_rank_state,
    commit_checkpoint,
    restore_rng,
)
from trainguard.config import ProjectConfig
from trainguard.events import append_event, write_json_atomic


@dataclass
class PendingSave:
    path: Path
    step: int
    started: float
    staging_seconds: float
    future: Any | None


def start_save(
    config: ProjectConfig,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    rank: int,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    process_group: dist.ProcessGroup,
) -> PendingSave:
    path = candidate_path(run_dir, attempt_id, step)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.monotonic()
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state_ready = time.monotonic()
    state = {"model": model_state, "optimizer": optimizer_state}
    if config.checkpoint.mode == "async":
        future = dcp.async_save(
            state, checkpoint_id=path / "dcp", process_group=process_group
        )
        staging_seconds = time.monotonic() - started
    else:
        dcp.save(state, checkpoint_id=path / "dcp", process_group=process_group)
        future = None
        staging_seconds = state_ready - started
    local = capture_rank_state(config, run_id, attempt_id, rank, step, scheduler.state_dict())
    write_json_atomic(path / f"rank-{rank}.json", local)
    return PendingSave(path, step, started, staging_seconds, future)


def finish_save(
    pending: PendingSave,
    config: ProjectConfig,
    run_id: str,
    attempt_id: str,
    rank: int,
    event_path: Path,
) -> None:
    if pending.future is not None:
        pending.future.result()
    dist.barrier()
    writing_seconds = time.monotonic() - pending.started - pending.staging_seconds
    if rank == 0:
        commit_started = time.monotonic()
        commit_checkpoint(pending.path, config, run_id, attempt_id, pending.step)
        commit_seconds = time.monotonic() - commit_started
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type="checkpoint_committed",
            global_step=pending.step,
            checkpoint_path=str(pending.path),
            staging_seconds=pending.staging_seconds,
            writing_seconds=writing_seconds,
            checksum_commit_seconds=commit_seconds,
        )
    dist.barrier()


def load_training_state(
    path: Path,
    rank: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    process_group: dist.ProcessGroup,
    omit_state: str = "none",
) -> tuple[int, int]:
    # Allocate AdamW's per-parameter slots before DCP loads in place.
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=path / "dcp", process_group=process_group)
    if omit_state == "optimizer":
        initial_optimizer = optimizer.state_dict()
    set_state_dict(
        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"]
    )
    if omit_state == "optimizer":
        optimizer.load_state_dict(initial_optimizer)
    local = json.loads((path / f"rank-{rank}.json").read_text(encoding="utf-8"))
    scheduler.load_state_dict(local["scheduler"])
    if omit_state != "rng":
        restore_rng(local)
    cursor = local["next_data_step"] if omit_state != "cursor" else 0
    return local["global_step"], cursor
