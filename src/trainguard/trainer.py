"""One fixed-size CPU DDP training attempt."""

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from trainguard.config import load_config
from trainguard.data import sample_ids_for_step, token_batch
from trainguard.events import append_event, write_json_atomic
from trainguard.model import TinyTransformer


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train(config_path: Path, run_dir: Path, run_id: str, attempt_id: str) -> None:
    config = load_config(config_path)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != config.run.world_size:
        raise ValueError("launched world size differs from configuration")

    torch.set_num_threads(1)
    torch.manual_seed(config.run.seed)
    dist.init_process_group(backend=config.run.backend, timeout=timedelta(seconds=120))
    event_path = run_dir / "attempts" / attempt_id / f"rank-{rank}.jsonl"
    try:
        model = TinyTransformer(config.model)
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.total_steps
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        append_event(
            event_path,
            run_id=run_id,
            attempt_id=attempt_id,
            rank=rank,
            event_type="training_started",
            global_step=0,
            world_size=world_size,
            parameter_count=parameter_count,
        )

        last_loss = 0.0
        for step in range(config.training.total_steps):
            ids = sample_ids_for_step(
                step, rank, world_size, config.training.batch_size_per_rank
            )
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

        dist.barrier()
        if rank == 0:
            write_json_atomic(
                run_dir / "summary.json",
                {
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "config_fingerprint": config.fingerprint(),
                    "global_step": config.training.total_steps,
                    "parameter_count": parameter_count,
                    "model_sha256": model_digest(model),
                    "last_loss": last_loss,
                    "torch_version": torch.__version__,
                    "world_size": world_size,
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
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int)
    args = parser.parse_args()
    train(args.config, args.run_dir, args.run_id, args.attempt_id)


if __name__ == "__main__":
    main()
