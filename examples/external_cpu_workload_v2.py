"""External SGD/StepLR workload with independently restorable input and extra state."""

from __future__ import annotations

import json

import torch
from external_v2_helper import mix_value
from torch import nn

WORKLOAD_API_VERSION = 2


class StatefulStream:
    def __init__(self, config, rank: int, consumed_batches: int, data_paths: dict) -> None:
        self.config = config
        self.rank = rank
        self.consumed_batches = consumed_batches
        self.rng_state = config.run.seed + rank + 1
        self.values = json.loads(data_paths["training"].read_text(encoding="utf-8"))
        if not isinstance(self.values, list) or len(self.values) < 4:
            raise ValueError("declared training data is invalid")

    def next(self) -> tuple[list[int], torch.Tensor]:
        ids, rows = [], []
        for _ in range(self.config.training.batch_size_per_rank):
            self.rng_state = (1103515245 * self.rng_state + 12345) % (2**31)
            position = self.rng_state % len(self.values)
            ids.append(10000 * self.rank + position)
            base = mix_value(self.values[position], self.rng_state)
            rows.append([
                (base + index / (self.config.training.sequence_length + 1)) % 1.0
                for index in range(self.config.training.sequence_length)
            ])
        self.consumed_batches += 1
        return ids, torch.tensor(rows, dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "consumed_batches": self.consumed_batches,
            "rng_state": self.rng_state,
        }

    def load_state_dict(self, value: dict) -> None:
        if value["consumed_batches"] != self.consumed_batches:
            raise ValueError("stream cursor differs from checkpoint")
        self.rng_state = value["rng_state"]

    def close(self) -> None:
        pass


class ExtraState:
    def __init__(self) -> None:
        self.calls = 0
        self.offset = 0.0

    def state_dict(self) -> dict:
        return {"calls": self.calls, "offset": self.offset}

    def load_state_dict(self, value: dict) -> None:
        self.calls = value["calls"]
        self.offset = value["offset"]


def build_model(config) -> nn.Module:
    return nn.Sequential(
        nn.Linear(config.training.sequence_length, config.model.hidden_size),
        nn.Tanh(),
        nn.Dropout(config.model.dropout),
        nn.Linear(config.model.hidden_size, config.training.sequence_length),
    )


def build_optimizer(model, config) -> torch.optim.Optimizer:
    del config
    return torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def build_scheduler(optimizer, config) -> torch.optim.lr_scheduler.LRScheduler:
    del config
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)


def build_stream(config, rank: int, consumed_batches: int, data_paths: dict) -> StatefulStream:
    return StatefulStream(config, rank, consumed_batches, data_paths)


def build_extra_state(config, rank: int) -> ExtraState:
    del config, rank
    return ExtraState()


def loss(output: torch.Tensor, batch: torch.Tensor, config, extra: ExtraState) -> torch.Tensor:
    del config
    extra.calls += 1
    extra.offset += 0.001 * extra.calls
    return nn.functional.mse_loss(output, batch.square() + extra.offset)
