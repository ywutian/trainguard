"""Stable, disjoint samples for fixed-size distributed training."""

from __future__ import annotations

import torch


def sample_ids_for_step(step: int, rank: int, world_size: int, batch_size: int) -> list[int]:
    if step < 0 or rank < 0 or rank >= world_size or world_size < 1 or batch_size < 1:
        raise ValueError("invalid step, rank, world size, or batch size")
    base = step * world_size * batch_size + rank * batch_size
    return list(range(base, base + batch_size))


def token_batch(
    sample_ids: list[int], sequence_length: int, vocab_size: int, seed: int
) -> torch.Tensor:
    rows = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + sample_id)
        rows.append(torch.randint(0, vocab_size, (sequence_length + 1,), generator=generator))
    return torch.stack(rows)


class TokenDataset(torch.utils.data.Dataset):
    """Immutable token rows; augmentation is a pure function of row and epoch."""

    def __init__(self, config):
        self.config = config
        self.rows = read_token_rows(config) if config.data.kind == "jsonl" else None

    def __len__(self):
        return len(self.rows) if self.rows is not None else 2**60

    def __getitem__(self, index):
        sample_id, epoch = index
        config = self.config
        if self.rows is None:
            tokens = token_batch(
                [sample_id],
                config.training.sequence_length,
                config.model.vocab_size,
                config.run.seed,
            )[0]
        else:
            row = self.rows[sample_id]
            length = config.training.sequence_length + 1
            offset = 0
            if config.data.random_crop and len(row) > length:
                generator = torch.Generator().manual_seed(
                    config.run.seed + epoch * 1000003 + sample_id
                )
                offset = int(torch.randint(0, len(row) - length + 1, (), generator=generator))
            tokens = torch.tensor(row[offset : offset + length], dtype=torch.long)
        return sample_id, tokens


def read_token_rows(config):
    import hashlib
    import json
    from pathlib import Path

    path = Path(config.data.path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config.data.sha256:
        raise ValueError("dataset SHA-256 differs from immutable configuration")
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        record = json.loads(line)
        tokens = record.get("tokens") if isinstance(record, dict) else None
        if (
            not isinstance(tokens, list)
            or len(tokens) < config.training.sequence_length + 1
            or any(
                type(token) is not int or not 0 <= token < config.model.vocab_size
                for token in tokens
            )
        ):
            raise ValueError(f"dataset line {number} has invalid token IDs or length")
        rows.append(tokens)
    if len(rows) < config.run.world_size:
        raise ValueError("dataset requires at least one row per rank")
    return rows


class ConsumedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, config, rank, start, size=None):
        self.config, self.rank, self.start, self.size = config, rank, start, size

    def __iter__(self):
        import math

        config = self.config
        cursor = self.start
        batch_size, world_size = config.training.batch_size_per_rank, config.run.world_size
        while True:
            if self.size is None:
                yield [
                    (sample, 0)
                    for sample in sample_ids_for_step(cursor, self.rank, world_size, batch_size)
                ]
            else:
                per_rank = math.ceil(self.size / world_size)
                batches = math.ceil(per_rank / batch_size)
                epoch, offset = divmod(cursor, batches)
                generator = torch.Generator().manual_seed(config.run.seed + epoch)
                indices = (
                    torch.randperm(self.size, generator=generator).tolist()
                    if config.data.shuffle
                    else list(range(self.size))
                )
                indices += indices[: per_rank * world_size - self.size]
                local = indices[self.rank :: world_size]
                yield [
                    (sample, epoch)
                    for sample in local[offset * batch_size : (offset + 1) * batch_size]
                ]
            cursor += 1


class BatchStream:
    """Prefetched batches are reconstructed from the last consumed batch boundary."""

    def __init__(self, config, rank, consumed_batches=0):
        dataset = TokenDataset(config)
        sampler = ConsumedBatchSampler(
            config, rank, consumed_batches, len(dataset) if dataset.rows is not None else None
        )
        options = {
            "num_workers": config.training.dataloader_workers,
            "generator": torch.Generator().manual_seed(config.run.seed + rank),
            "batch_sampler": sampler,
        }
        if config.training.dataloader_workers:
            options.update(
                prefetch_factor=config.training.prefetch_factor, multiprocessing_context="spawn"
            )
        self.loader = torch.utils.data.DataLoader(dataset, **options)
        self.iterator = iter(self.loader)

    def next(self):
        ids, tokens = next(self.iterator)
        return ids.tolist(), tokens

    def close(self):
        # DataLoader has no public close API; release the iterator so it shuts down workers.
        self.iterator = None
        self.loader = None
