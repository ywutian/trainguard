"""Decoder-only byte-level transformer (GPT-style pre-norm blocks)."""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        h = self.norm_attention(x)
        x = x + self.attention(h, h, h, attn_mask=causal, need_weights=False)[0]
        return x + self.mlp(self.norm_mlp(x))


class ByteGPT(nn.Module):
    # ponytail: untied input/output embeddings; tying adds a shared-parameter DCP case to prove.
    def __init__(self, vocab: int, width: int, heads: int, layers: int, context: int,
                 dropout: float) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(context, width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1,
        )
        positions = torch.arange(length, device=tokens.device)
        x = self.dropout(self.token(tokens) + self.position(positions))
        for block in self.blocks:
            x = block(x, causal)
        return self.head(self.norm(x))
