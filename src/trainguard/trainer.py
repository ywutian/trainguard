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
from torch.distributed.checkpoint.api import CheckpointException
from torch.nn.parallel import DistributedDataParallel

from trainguard.checkpoint import validate_checkpoint
from trainguard.checkpoint_io import finish_save, load_training_state, save_ready, start_save
from trainguard.config import load_config
from trainguard.data import BatchStream
from trainguard.events import append_event, sync_event_file, write_json_atomic
from trainguard.external_workload import (
    frozen_workload_path,
    load_verified_workload,
    read_verified_source,
)
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
    expected_config_fingerprint: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> None:
    if resume_checkpoint is not None and expected_checkpoint_sha256 is None:
        raise ValueError("resumed worker requires the selected checkpoint manifest digest")
    if resume_checkpoint is None and expected_checkpoint_sha256 is not None:
        raise ValueError("checkpoint manifest digest requires a resume checkpoint")
    config = load_config(config_path)
    if expected_config_fingerprint is not None and config.fingerprint() != expected_config_fingerprint:
        raise ValueError("worker configuration differs from controller-approved configuration")
    external = None
    if config.external_workload is not None:
        frozen = frozen_workload_path(run_dir)
        external = load_verified_workload(
            config, read_verified_source(config, path=frozen), frozen
        )
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

    def inject(kind: str, active: bool, step: int, checkpoint_path: Path | None = None) -> None:
        if active:
            event("fault_injected", global_step=step, fault_kind=kind)
            sync_event_file(event_path)
        _inject_fault(kind, active, checkpoint_path)

    def verify_resume_checkpoint(phase: str) -> None:
        if resume_checkpoint is None:
            return
        local_error = None
        try:
            validate_checkpoint(
                resume_checkpoint, config, run_id,
                expected_manifest_sha256=expected_checkpoint_sha256,
                require_trainable_state=True,
            )
        except Exception as exc:  # noqa: BLE001 - synchronize rejection across every rank
            local_error = f"{type(exc).__name__}: {exc}"
        failed = torch.tensor([int(local_error is not None)], dtype=torch.int64)
        dist.all_reduce(failed, op=dist.ReduceOp.SUM, group=control_group)
        if failed.item():
            detail = local_error or "another rank rejected the selected checkpoint"
            raise RuntimeError(f"checkpoint changed {phase}: {detail}")

    try:
        model = external.build_model(config) if external is not None else TinyTransformer(config.model)
        if not isinstance(model, nn.Module):
            raise TypeError("workload build_model must return a torch module")
        wrapped = wrap_model(model, config, device)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.total_steps
        )
        scaler = torch.amp.GradScaler("cuda") if config.training.precision == "fp16" else None
        state = TrainingState(scaler=scaler)
        event("group_initialized", global_step=0, device=str(device), strategy=config.run.strategy)
        if resume_checkpoint is not None:
            verify_resume_checkpoint("before load")
            loaded = time.monotonic()
            try:
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
            except (Exception, CheckpointException) as exc:
                write_json_atomic(
                    run_dir / "attempts" / attempt_id / f"rank-{rank}-restore-failure.json",
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "attempt_id": attempt_id,
                        "rank": rank,
                        "checkpoint_path": str(resume_checkpoint),
                        "manifest_sha256": expected_checkpoint_sha256,
                        "error_type": type(exc).__name__,
                    },
                )
                event(
                    "checkpoint_restore_failed",
                    checkpoint_path=str(resume_checkpoint),
                    manifest_sha256=expected_checkpoint_sha256,
                    error_type=type(exc).__name__,
                )
                sync_event_file(event_path)
                raise
            verify_resume_checkpoint("during load")
            event(
                "state_loaded",
                global_step=state.optimizer_updates,
                consumed_batches=state.consumed_batches,
                load_seconds=time.monotonic() - loaded,
            )
        resumed_at = state.optimizer_updates
        cursor = data_start if resume_checkpoint else state.consumed_batches
        stream = (
            external.build_stream(config, rank, cursor)
            if external is not None else BatchStream(config, rank, cursor)
        )
        if not callable(getattr(stream, "next", None)) or not callable(
            getattr(stream, "close", None)
        ):
            raise TypeError("workload build_stream must return a stream with next and close")
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
                if external is not None and (
                    not isinstance(batch_ids, list)
                    or not 1 <= len(batch_ids) <= config.training.batch_size_per_rank
                    or any(type(sample) is not int or sample < 0 for sample in batch_ids)
                    or not isinstance(tokens, torch.Tensor)
                    or tokens.shape[0] != len(batch_ids)
                ):
                    raise ValueError("external stream must return sample IDs and a Tensor batch")
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
                        if external is None:
                            logits = wrapped(tokens[:, :-1])
                            loss = nn.functional.cross_entropy(
                                logits.reshape(-1, config.model.vocab_size),
                                tokens[:, 1:].reshape(-1),
                            )
                        else:
                            loss = external.loss(wrapped(tokens), tokens, config)
                            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                                raise TypeError("external workload loss must return a scalar Tensor")
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
                        inject(
                            "corrupt",
                            _fault_active(
                                attempt_id, rank, config.fault.rank, pending.step, config.fault.step
                            ),
                            pending.step,
                            pending.path,
                        )
                    pending = None
            active = _fault_active(
                attempt_id, rank, config.fault.rank, completed, config.fault.step
            )
            if config.fault.kind in {"worker_exit", "hang"}:
                inject(config.fault.kind, active, completed)
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
                        inject(
                            "corrupt",
                            _fault_active(
                                attempt_id, rank, config.fault.rank, pending.step, config.fault.step
                            ),
                            pending.step,
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
                    inject("save_interrupt", active, completed)
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
                        inject("corrupt", active, pending.step, pending.path)
                    pending = None
        if pending is not None:
            finish_save(
                pending, config, run_id, attempt_id, rank, event_path, completed, control_group
            )
            if config.fault.kind == "corrupt":
                inject(
                    "corrupt",
                    _fault_active(
                        attempt_id, rank, config.fault.rank, pending.step, config.fault.step
                    ),
                    pending.step,
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
        completion_sync_failed = False
        try:
            sync_event_file(event_path)
        except OSError:
            completion_sync_failed = True
        completion_status = torch.tensor([int(completion_sync_failed)], dtype=torch.int64)
        dist.all_reduce(completion_status, op=dist.ReduceOp.SUM, group=control_group)
        if completion_status.item():
            raise RuntimeError("training completion evidence could not be persisted")
    finally:
        if stream is not None:
            stream.close()
        dist.destroy_process_group(control_group)
        dist.destroy_process_group(checkpoint_group)
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-fingerprint")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()
    train(
        args.config, args.run_dir, args.run_id, args.attempt_id,
        args.resume_checkpoint, args.expected_config_fingerprint,
        args.expected_checkpoint_sha256,
    )


if __name__ == "__main__":
    main()
