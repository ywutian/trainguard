"""One fixed-size CPU DDP training attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from trainguard.checkpoint_io import finish_save, load_training_state, save_ready, start_save
from trainguard.config import load_config
from trainguard.data import sample_ids_for_step, token_batch
from trainguard.events import append_event, write_json_atomic
from trainguard.model import TinyTransformer


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


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_digest(value: object) -> str:
    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            digest.update(str(item.dtype).encode())
            digest.update(str(tuple(item.shape)).encode())
            digest.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=str):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for part in item:
                update(part)
        else:
            digest.update(json.dumps(item, sort_keys=True).encode())

    update(value)
    return digest.hexdigest()


def train(
    config_path: Path,
    run_dir: Path,
    run_id: str,
    attempt_id: str,
    resume_checkpoint: Path | None = None,
) -> None:
    config = load_config(config_path)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != config.run.world_size:
        raise ValueError("launched world size differs from configuration")

    torch.set_num_threads(1)
    torch.manual_seed(config.run.seed)
    dist.init_process_group(backend=config.run.backend, timeout=timedelta(seconds=120))
    checkpoint_group = dist.new_group(backend="gloo")
    control_group = dist.new_group(backend="gloo")
    event_path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
    try:
        model = TinyTransformer(config.model)
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.total_steps
        )
        global_step = 0
        cursor = 0
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type="group_initialized",
            global_step=0,
        )
        if resume_checkpoint is not None:
            load_started = time.monotonic()
            global_step, cursor = load_training_state(
                resume_checkpoint,
                rank,
                wrapped,
                optimizer,
                scheduler,
                checkpoint_group,
                config.recovery.omit_state,
            )
            append_event(
                event_path,
                run_id=run_id,
                attempt_id=attempt_id,
                rank=rank,
                event_type="state_loaded",
                global_step=global_step,
                load_seconds=time.monotonic() - load_started,
            )
        parameter_count = sum(p.numel() for p in model.parameters())
        training_started = time.monotonic()
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type="training_started",
            global_step=global_step,
            world_size=world_size,
            parameter_count=parameter_count,
            resumed_from=str(resume_checkpoint) if resume_checkpoint else None,
        )

        last_loss = 0.0
        pending = None
        for step in range(global_step, config.training.total_steps):
            ids = sample_ids_for_step(cursor, rank, world_size, config.training.batch_size_per_rank)
            tokens = token_batch(
                ids, config.training.sequence_length, config.model.vocab_size, config.run.seed
            )
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(tokens[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, config.model.vocab_size), tokens[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            cursor += 1
            last_loss = float(loss.detach())
            append_event(
                event_path,
                run_id=run_id,
                attempt_id=attempt_id,
                rank=rank,
                event_type="step_completed",
                global_step=step + 1,
                sample_ids=ids,
                loss=last_loss,
                learning_rate=scheduler.get_last_lr()[0],
            )
            completed = step + 1
            if resume_checkpoint is not None and completed == global_step + 1:
                append_event(
                    event_path,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    rank=rank,
                    event_type="first_resumed_update",
                    global_step=completed,
                )
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
            fault_active = _fault_active(
                attempt_id, rank, config.fault.rank, completed, config.fault.step
            )
            if config.fault.kind in {"worker_exit", "hang"}:
                if fault_active:
                    append_event(
                        event_path,
                        run_id=run_id,
                        attempt_id=attempt_id,
                        rank=rank,
                        event_type="fault_injected",
                        global_step=completed,
                        fault_kind=config.fault.kind,
                    )
                _inject_fault(config.fault.kind, fault_active)
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
                                attempt_id,
                                rank,
                                config.fault.rank,
                                pending.step,
                                config.fault.step,
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
                )
                if config.fault.kind == "save_interrupt":
                    _inject_fault("save_interrupt", fault_active)
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
                        _inject_fault("corrupt", fault_active, pending.path)
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

        dist.barrier(group=control_group)
        training_elapsed_seconds = time.monotonic() - training_started
        rank_times = [None] * world_size
        dist.all_gather_object(rank_times, training_elapsed_seconds, group=control_group)
        if rank == 0:
            write_json_atomic(
                run_dir / "attempts" / attempt_id / "summary.json",
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "config_fingerprint": config.fingerprint(),
                    "global_step": config.training.total_steps,
                    "parameter_count": parameter_count,
                    "model_sha256": model_digest(model),
                    "optimizer_sha256": state_digest(optimizer.state_dict()),
                    "scheduler_sha256": state_digest(scheduler.state_dict()),
                    "workload_fingerprint": config.workload_fingerprint(),
                    "last_loss": last_loss,
                    "torch_version": torch.__version__,
                    "world_size": world_size,
                    "training_elapsed_seconds": max(rank_times),
                    "rank_training_seconds": rank_times,
                },
            )
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type="training_completed",
            global_step=config.training.total_steps,
        )
    finally:
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
