"""Small language-model workload for CPU correctness runs."""

from __future__ import annotations

import torch
from torch import nn

from trainguard.config import ModelSettings


class TinyTransformer(nn.Module):
    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.embedding = nn.Embedding(settings.vocab_size, settings.hidden_size)
        self.position = nn.Embedding(512, settings.hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=settings.hidden_size,
            nhead=settings.num_heads,
            dim_feedforward=settings.hidden_size * 4,
            dropout=settings.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=settings.num_layers)
        self.output = nn.Linear(settings.hidden_size, settings.vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)
        return self.output(self.encoder(hidden, mask=mask))
