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
from trainguard.training_state import TrainingState


@dataclass
class PendingSave:
    path: Path
    step: int
    started: float
    staging_seconds: float
    future: Any | None
    preparation_seconds: float = 0.0
    upload_started: float = 0.0
    upload_finished: float | None = None
    deadline: float | None = None

    def mark_uploaded(self, future: Any = None) -> None:
        self.upload_finished = time.monotonic()


def save_ready(pending: PendingSave, process_group: dist.ProcessGroup) -> bool:
    done = pending.future is None or pending.future.done()
    failed = done and pending.future is not None and pending.future.exception() is not None
    expired = pending.deadline is not None and (
        (not done and time.monotonic() > pending.deadline)
        or (pending.upload_finished is not None and pending.upload_finished > pending.deadline)
    )
    status = torch.tensor([int(not done), int(failed), int(expired)], dtype=torch.int64)
    dist.all_reduce(status, op=dist.ReduceOp.SUM, group=process_group)
    if status[2].item():
        raise TimeoutError("checkpoint upload exceeded its deadline on at least one rank")
    if status[1].item():
        raise RuntimeError("checkpoint upload failed on at least one rank")
    return status[0].item() == 0


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
    training_state: TrainingState | None = None,
) -> PendingSave:
    path = candidate_path(run_dir, attempt_id, step)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.monotonic()
    model_state, optimizer_state = get_state_dict(model, optimizer)
    state_ready = time.monotonic()
    state = {"model": model_state, "optimizer": optimizer_state}
    upload_started = time.monotonic()
    if config.checkpoint.mode == "async":
        response = dcp.async_save(state, checkpoint_id=path / "dcp", process_group=process_group)
        if hasattr(response, "staging_completion"):
            response.staging_completion.result(timeout=config.checkpoint.save_timeout_seconds)
            future = response.upload_completion
        else:
            future = response
        staging_seconds = time.monotonic() - upload_started
        pending = PendingSave(
            path, step, started, staging_seconds, future, state_ready - started, time.monotonic()
        )
        future.add_done_callback(pending.mark_uploaded)
    else:
        dcp.save(state, checkpoint_id=path / "dcp", process_group=process_group)
        pending = PendingSave(
            path, step, started, 0.0, None, state_ready - started, upload_started, time.monotonic()
        )
    pending.deadline = started + config.checkpoint.save_timeout_seconds
    local = capture_rank_state(config, run_id, attempt_id, rank, step, scheduler.state_dict())
    if training_state is not None:
        local.update(training_state.snapshot())
        local["next_data_step"] = training_state.consumed_batches
    write_json_atomic(path / f"rank-{rank}.json", local)
    return pending


def finish_save(
    pending: PendingSave,
    config: ProjectConfig,
    run_id: str,
    attempt_id: str,
    rank: int,
    event_path: Path,
    current_step: int | None = None,
    control_group: dist.ProcessGroup | None = None,
) -> None:
    waited = time.monotonic()
    group = control_group
    if pending.future is not None:
        if pending.deadline is None:
            pending.deadline = waited + config.checkpoint.save_timeout_seconds
        while not save_ready(pending, group):
            time.sleep(0.001)
        pending.future.result()
    dist.barrier(group=group)
    main_wait = time.monotonic() - waited
    upload_finished = pending.upload_finished or time.monotonic()
    metrics = torch.tensor(
        [
            pending.preparation_seconds,
            pending.staging_seconds,
            max(0.0, upload_finished - pending.upload_started),
            main_wait,
        ],
        dtype=torch.float64,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX, group=group)
    # Completion times stay local; durations can safely be reduced across hosts.
    commit_started = time.monotonic()
    commit_metrics = torch.zeros(3, dtype=torch.float64)
    commit_error = None
    commit_cause: Exception | None = None
    if rank == 0:
        try:
            record = commit_checkpoint(pending.path, config, run_id, attempt_id, pending.step)
            commit_metrics[0] = time.monotonic() - commit_started
            commit_metrics[1] = sum(item["size"] for item in record.manifest["files"])
        except Exception as exc:  # noqa: BLE001 - notify peers before leaving the collective
            commit_metrics[2] = 1
            commit_error = f"{type(exc).__name__}: {exc}"
            commit_cause = exc
    dist.broadcast(commit_metrics, src=0, group=group)
    if commit_metrics[2].item():
        failure = [commit_error]
        dist.broadcast_object_list(failure, src=0, group=group)
        if commit_cause is not None:
            raise RuntimeError(f"checkpoint commit failed: {failure[0]}") from commit_cause
        raise RuntimeError(f"checkpoint commit failed: {failure[0]}")
    eligibility = torch.tensor([max(0.0, time.monotonic() - upload_finished)], dtype=torch.float64)
    dist.all_reduce(eligibility, op=dist.ReduceOp.MAX, group=group)
    append_event(
        event_path,
        run_id=run_id,
        attempt_id=attempt_id,
        rank=rank,
        event_type="checkpoint_phase_completed",
        global_step=pending.step,
        local_upload_seconds=max(0.0, upload_finished - pending.upload_started),
        local_staging_seconds=pending.staging_seconds,
        local_main_thread_wait_seconds=main_wait,
    )
    if rank == 0:
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type="checkpoint_committed",
            global_step=pending.step,
            checkpoint_path=str(pending.path),
            committed_at_step=current_step or pending.step,
            recoverable_step_lag=(current_step or pending.step) - pending.step,
            preparation_seconds=metrics[0].item(),
            staging_seconds=metrics[1].item(),
            upload_seconds=metrics[2].item(),
            main_thread_wait_seconds=metrics[3].item(),
            writing_seconds=max(0.0, time.monotonic() - pending.started - pending.staging_seconds),
            checksum_commit_seconds=commit_metrics[0].item(),
            eligibility_lag_seconds=eligibility.item(),
            checkpoint_bytes=int(commit_metrics[1].item()),
        )
    dist.barrier(group=group)


def load_training_state(
    path: Path,
    rank: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    process_group: dist.ProcessGroup,
    omit_state: str = "none",
    training_state: TrainingState | None = None,
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
    if training_state is not None:
        training_state.restore(local)
    cursor = local["next_data_step"] if omit_state != "cursor" else 0
    return local["global_step"], cursor
