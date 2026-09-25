"""Stable, disjoint samples for fixed-size distributed training."""

from __future__ import annotations

import torch


def sample_ids_for_step(step: int, rank: int, world_size: int, batch_size: int) -> list[int]:
    if step < 0 or rank < 0 or rank >= world_size or world_size < 1 or batch_size < 1:
        raise ValueError("invalid step, rank, world size, or batch size")
    base = step * world_size * batch_size + rank * batch_size
    return list(range(base, base + batch_size))


def token_batch(sample_ids: list[int], sequence_length: int, vocab_size: int, seed: int) -> torch.Tensor:
    rows = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + sample_id)
        rows.append(torch.randint(0, vocab_size, (sequence_length + 1,), generator=generator))
    return torch.stack(rows)
