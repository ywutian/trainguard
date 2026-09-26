"""One fixed-size CPU DDP training attempt."""

from __future__ import annotations

import argparse
import os
import random
import resource
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from trainguard.checkpoint_io import finish_save, load_training_state, save_ready, start_save
from trainguard.config import load_config
from trainguard.data import BatchStream
from trainguard.events import append_event, write_json_atomic
from trainguard.model import TinyTransformer
from trainguard.strategy import bind_device, state_digest, wrap_model
from trainguard.training_state import TrainingState, complete_update


def _inject_fault(kind: str, active: bool, checkpoint_path: Path | None = None) -> None:
    if not active:
        return
    if kind in {"worker_exit", "save_interrupt"}:
        os._exit(71)
    if kind == "hang":
        while True:
            time.sleep(60)
    if kind == "corrupt":
        if checkpoint_path is None:
            raise ValueError("corruption fault needs a committed checkpoint")
        files = sorted((checkpoint_path / "dcp").glob("*.distcp"))
        if not files:
            raise FileNotFoundError("no DCP file available to corrupt")
        with files[0].open("ab") as stream:
            stream.write(b"corrupted after commit")
            stream.flush()
            os.fsync(stream.fileno())
        os._exit(72)


def _fault_active(
    attempt_id: str, rank: int, fault_rank: int, step: int, fault_step: int | None
) -> bool:
    return attempt_id == "attempt-001" and rank == fault_rank and step == fault_step


def train(
    config_path: Path,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    resume_checkpoint: Path | None = None,
) -> None:
    config = load_config(config_path)
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world_size != config.run.world_size:
        raise ValueError("launched world size differs from configuration")
    torch.set_num_threads(1)
    random.seed(config.run.seed)
    np.random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    device = bind_device(config)
    torch.use_deterministic_algorithms(config.run.deterministic)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    dist.init_process_group(backend=config.run.backend, timeout=timedelta(seconds=120))
    checkpoint_group = dist.new_group(backend="gloo")
    control_group = dist.new_group(backend="gloo")
    event_path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
    stream = None

    def event(event_type, **fields):
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type=event_type,
            **fields,
        )

    try:
        model = TinyTransformer(config.model)
        wrapped = wrap_model(model, config, device)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.total_steps
        )
        scaler = torch.amp.GradScaler("cuda") if config.training.precision == "fp16" else None
        state = TrainingState(scaler=scaler)
        event("group_initialized", global_step=0, device=str(device), strategy=config.run.strategy)
        if resume_checkpoint is not None:
            loaded = time.monotonic()
            _, data_start = load_training_state(
                resume_checkpoint,
                rank,
                wrapped,
                optimizer,
                scheduler,
                checkpoint_group,
                config.recovery.omit_state,
                state,
            )
            event(
                "state_loaded",
                global_step=state.optimizer_updates,
                consumed_batches=state.consumed_batches,
                load_seconds=time.monotonic() - loaded,
            )
        resumed_at = state.optimizer_updates
        stream = BatchStream(
            config, rank, data_start if resume_checkpoint else state.consumed_batches
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        training_started = time.monotonic()
        event(
            "training_started",
            global_step=state.optimizer_updates,
            consumed_batches=state.consumed_batches,
            world_size=world_size,
            parameter_count=parameter_count,
            resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
        )
        last_loss, pending, consecutive_skips = 0.0, None, 0
        accumulation = config.training.gradient_accumulation_steps
        precision = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(config.training.precision)
        while state.optimizer_updates < config.training.total_steps:
            optimizer.zero_grad(set_to_none=True)
            ids = []
            for micro in range(accumulation):
                batch_ids, tokens = stream.next()
                ids.extend(batch_ids)
                tokens = tokens.to(device)
                context = (
                    wrapped.no_sync()
                    if isinstance(wrapped, DistributedDataParallel) and micro < accumulation - 1
                    else nullcontext()
                )
                with context:
                    with torch.autocast(
                        device_type=device.type, dtype=precision, enabled=precision is not None
                    ):
                        logits = wrapped(tokens[:, :-1])
                        loss = nn.functional.cross_entropy(
                            logits.reshape(-1, config.model.vocab_size), tokens[:, 1:].reshape(-1)
                        )
                    scaled_loss = loss / accumulation
                    (scaler.scale(scaled_loss) if scaler is not None else scaled_loss).backward()
                state.consumed_batches += 1
                event(
                    "batch_consumed",
                    global_step=state.optimizer_updates,
                    consumed_batches=state.consumed_batches,
                    sample_ids=batch_ids,
                )
            if not complete_update(wrapped, optimizer, scheduler, state, control_group):
                consecutive_skips += 1
                event(
                    "update_skipped",
                    global_step=state.optimizer_updates,
                    consumed_batches=state.consumed_batches,
                    sample_ids=ids,
                )
                if consecutive_skips >= config.training.max_consecutive_skips:
                    raise RuntimeError("AMP exceeded consecutive skipped-update budget")
                continue
            consecutive_skips = 0
            completed = state.optimizer_updates
            logged_loss = None
            if (
                device.type == "cpu"
                or completed % config.training.loss_log_interval == 0
                or completed == config.training.total_steps
            ):
                last_loss = float(loss.detach())
                logged_loss = last_loss
            event(
                "step_completed",
                global_step=completed,
                consumed_batches=state.consumed_batches,
                sample_ids=ids,
                loss=logged_loss,
                learning_rate=scheduler.get_last_lr()[0],
            )
            if resume_checkpoint is not None and completed == resumed_at + 1:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                event("first_resumed_update", global_step=completed)
            if pending is not None and (
                completed % config.checkpoint.poll_interval_steps == 0
                or (
                    config.fault.require_committed_step == pending.step
                    and completed == config.fault.step
                )
            ):
                force = (
                    config.fault.require_committed_step == pending.step
                    and completed == config.fault.step
                )
                if force or save_ready(pending, control_group):
                    finish_save(
                        pending,
                        config,
                        run_id,
                        attempt_id,
                        rank,
                        event_path,
                        completed,
                        control_group,
                    )
                    if config.fault.kind == "corrupt":
                        _inject_fault(
                            "corrupt",
                            _fault_active(
                                attempt_id, rank, config.fault.rank, pending.step, config.fault.step
                            ),
                            pending.path,
                        )
                    pending = None
            active = _fault_active(
                attempt_id, rank, config.fault.rank, completed, config.fault.step
            )
            if config.fault.kind in {"worker_exit", "hang"}:
                if active:
                    event("fault_injected", global_step=completed, fault_kind=config.fault.kind)
                _inject_fault(config.fault.kind, active)
            if config.checkpoint.mode != "none" and (
                completed % config.checkpoint.interval_steps == 0
                or completed == config.training.total_steps
            ):
                if pending is not None:
                    finish_save(
                        pending,
                        config,
                        run_id,
                        attempt_id,
                        rank,
                        event_path,
                        completed,
                        control_group,
                    )
                    if config.fault.kind == "corrupt":
                        _inject_fault(
                            "corrupt",
                            _fault_active(
                                attempt_id, rank, config.fault.rank, pending.step, config.fault.step
                            ),
                            pending.path,
                        )
                pending = start_save(
                    config,
                    run_dir,
                    run_id,
                    attempt_id,
                    rank,
                    completed,
                    wrapped,
                    optimizer,
                    scheduler,
                    checkpoint_group,
                    state,
                )
                if config.fault.kind == "save_interrupt":
                    if active:
                        event("fault_injected", global_step=completed, fault_kind=config.fault.kind)
                    _inject_fault("save_interrupt", active)
                if config.checkpoint.mode == "sync":
                    finish_save(
                        pending,
                        config,
                        run_id,
                        attempt_id,
                        rank,
                        event_path,
                        completed,
                        control_group,
                    )
                    if config.fault.kind == "corrupt":
                        _inject_fault("corrupt", active, pending.path)
                    pending = None
        if pending is not None:
            finish_save(
                pending, config, run_id, attempt_id, rank, event_path, completed, control_group
            )
            if config.fault.kind == "corrupt":
                _inject_fault(
                    "corrupt",
                    _fault_active(
                        attempt_id, rank, config.fault.rank, pending.step, config.fault.step
                    ),
                    pending.path,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        dist.barrier(group=control_group)
        elapsed = time.monotonic() - training_started
        local = {
            "model_sha256": state_digest(model.state_dict()),
            "optimizer_sha256": state_digest(optimizer.state_dict()),
            "scheduler_sha256": state_digest(scheduler.state_dict()),
            "scaler_sha256": state_digest(scaler.state_dict() if scaler else None),
            "optimizer_updates": state.optimizer_updates,
            "consumed_batches": state.consumed_batches,
            "training_seconds": elapsed,
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        rank_states = [None] * world_size
        dist.all_gather_object(rank_states, local, group=control_group)
        if rank == 0:
            hashes = {
                field: state_digest([item[field] for item in rank_states])
                for field in (
                    "model_sha256",
                    "optimizer_sha256",
                    "scheduler_sha256",
                    "scaler_sha256",
                )
            }
            write_json_atomic(
                run_dir / "attempts" / attempt_id / "summary.json",
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "config_fingerprint": config.fingerprint(),
                    "global_step": state.optimizer_updates,
                    "parameter_count": parameter_count,
                    **hashes,
                    "workload_fingerprint": config.workload_fingerprint(),
                    "last_loss": last_loss,
                    "torch_version": torch.__version__,
                    "world_size": world_size,
                    "training_elapsed_seconds": max(
                        item["training_seconds"] for item in rank_states
                    ),
                    "rank_states": rank_states,
                    **state.snapshot(),
                },
            )
        event(
            "training_completed",
            global_step=state.optimizer_updates,
            consumed_batches=state.consumed_batches,
        )
    finally:
        if stream is not None:
            stream.close()
        dist.destroy_process_group(control_group)
        dist.destroy_process_group(checkpoint_group)
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()
    train(args.config, args.run_dir, args.run_id, args.attempt_id, args.resume_checkpoint)


if __name__ == "__main__":
    main()
