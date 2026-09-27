"""Independent, single-file CPU DDP adapter used for integration evaluation."""

from __future__ import annotations

import torch
from torch import nn

WORKLOAD_API_VERSION = 1


class RegressionModel(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.layers(batch)


class RegressionStream:
    def __init__(self, config, rank: int, consumed_batches: int) -> None:
        self.config = config
        self.rank = rank
        self.cursor = consumed_batches

    def next(self) -> tuple[list[int], torch.Tensor]:
        config = self.config
        maximum = config.training.batch_size_per_rank
        size = max(1, maximum - self.cursor % 2)
        base = 1000 + self.cursor * config.run.world_size * maximum + self.rank * maximum
        sample_ids = list(range(base, base + size))
        rows = []
        for sample_id in sample_ids:
            generator = torch.Generator().manual_seed(config.run.seed + sample_id)
            rows.append(torch.rand(config.training.sequence_length, generator=generator) * 2 - 1)
        self.cursor += 1
        return sample_ids, torch.stack(rows)

    def close(self) -> None:
        pass


def build_model(config) -> nn.Module:
    return RegressionModel(
        config.training.sequence_length, config.model.hidden_size, config.model.dropout
    )


def build_stream(config, rank: int, consumed_batches: int) -> RegressionStream:
    return RegressionStream(config, rank, consumed_batches)


def loss(output: torch.Tensor, batch: torch.Tensor, config) -> torch.Tensor:
    del config
    return nn.functional.mse_loss(output, batch.square())
