"""Validated configuration for the runnable CPU baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSettings(StrictModel):
    seed: int = Field(default=42, ge=0)
    world_size: int = Field(default=2, ge=1)
    backend: Literal["gloo"] = "gloo"
    device: Literal["cpu"] = "cpu"
    timeout_seconds: int = Field(default=600, ge=1)


class TrainingSettings(StrictModel):
    total_steps: int = Field(ge=1)
    sequence_length: int = Field(ge=2, le=512)
    batch_size_per_rank: int = Field(ge=1)
    dataloader_workers: Literal[0] = 0


class ModelSettings(StrictModel):
    vocab_size: int = Field(default=128, ge=4)
    hidden_size: int = Field(default=64, ge=4)
    num_heads: int = Field(default=4, ge=1)
    num_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_attention(self) -> ModelSettings:
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self


class ProjectConfig(StrictModel):
    run: RunSettings
    training: TrainingSettings
    model: ModelSettings = Field(default_factory=ModelSettings)

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_config(path: Path) -> ProjectConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")
    return ProjectConfig.model_validate(raw)
