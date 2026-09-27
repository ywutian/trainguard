"""Representative workload: byte-level GPT language modeling on a public-domain novel.

Corpus: Jane Austen, "Pride and Prejudice" (Project Gutenberg eBook #1342,
https://www.gutenberg.org/cache/epub/1342/pg1342.txt, downloaded 2026-09-27,
SHA-256 3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75).
The committed file keeps only the text between the Project Gutenberg start and
end markers, with LF line endings. Tokens are the UTF-8 bytes (vocabulary 256).

The v2 adapter fixes the optimizer to momentum SGD with StepLR. Model width,
depth and context come from the run configuration, so the same file serves a
small CPU check and a larger GPU-scale run.
"""

from __future__ import annotations

import torch
from charlm_model import ByteGPT
from torch import nn

WORKLOAD_API_VERSION = 2
VOCABULARY = 256


class CorpusStream:
    """Random fixed-length windows over the corpus with a resumable cursor."""

    def __init__(self, config, rank: int, consumed_batches: int, data_paths: dict) -> None:
        self.config = config
        self.rank = rank
        self.consumed_batches = consumed_batches
        self.rng_state = config.run.seed * 1000 + rank + 1
        corpus = data_paths["corpus"].read_bytes()
        self.window = config.training.sequence_length + 1
        if len(corpus) <= self.window:
            raise ValueError("declared corpus is shorter than one training window")
        self.corpus = torch.frombuffer(bytearray(corpus), dtype=torch.uint8)

    def next(self) -> tuple[list[int], torch.Tensor]:
        ids, rows = [], []
        for _ in range(self.config.training.batch_size_per_rank):
            self.rng_state = (6364136223846793005 * self.rng_state + 1442695040888963407) % 2**64
            position = (self.rng_state >> 16) % (len(self.corpus) - self.window)
            ids.append(self.rank * 10_000_000 + position)
            rows.append(self.corpus[position:position + self.window])
        self.consumed_batches += 1
        return ids, torch.stack(rows).long()

    def state_dict(self) -> dict:
        return {"consumed_batches": self.consumed_batches, "rng_state": self.rng_state}

    def load_state_dict(self, value: dict) -> None:
        if value["consumed_batches"] != self.consumed_batches:
            raise ValueError("stream cursor differs from checkpoint")
        self.rng_state = value["rng_state"]

    def close(self) -> None:
        pass


class TokenCounter:
    """Training statistics a real job would checkpoint alongside the model."""

    def __init__(self) -> None:
        self.tokens_seen = 0
        self.loss_sum = 0.0

    def state_dict(self) -> dict:
        return {"tokens_seen": self.tokens_seen, "loss_sum": self.loss_sum}

    def load_state_dict(self, value: dict) -> None:
        self.tokens_seen = value["tokens_seen"]
        self.loss_sum = value["loss_sum"]


def build_model(config) -> nn.Module:
    model = config.model
    return ByteGPT(
        VOCABULARY, model.hidden_size, model.num_heads, model.num_layers,
        config.training.sequence_length + 1, model.dropout,
    )


def build_optimizer(model, config) -> torch.optim.Optimizer:
    del config
    return torch.optim.SGD(model.parameters(), lr=0.3, momentum=0.9)


def build_scheduler(optimizer, config) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, config.training.total_steps // 3), gamma=0.5,
    )


def build_stream(config, rank: int, consumed_batches: int, data_paths: dict) -> CorpusStream:
    return CorpusStream(config, rank, consumed_batches, data_paths)


def build_extra_state(config, rank: int) -> TokenCounter:
    del config, rank
    return TokenCounter()


def loss(output: torch.Tensor, batch: torch.Tensor, config, extra: TokenCounter) -> torch.Tensor:
    del config
    value = nn.functional.cross_entropy(
        output[:, :-1].reshape(-1, output.shape[-1]), batch[:, 1:].reshape(-1),
    )
    extra.tokens_seen += batch[:, 1:].numel()
    extra.loss_sum += float(value.detach())
    return value
