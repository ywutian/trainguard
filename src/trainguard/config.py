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


class CheckpointSettings(StrictModel):
    mode: Literal["none", "sync", "async"] = "none"
    interval_steps: int = Field(default=1, ge=1)
    poll_interval_steps: int = Field(default=5, ge=1)
    save_timeout_seconds: int = Field(default=120, ge=1)
    keep_last_k: int | None = Field(default=None, ge=2)
    max_retained_bytes: int | None = Field(default=None, ge=1)


class RecoverySettings(StrictModel):
    max_restarts: int = Field(default=2, ge=0)
    progress_timeout_seconds: int = Field(default=120, ge=1)
    omit_state: Literal["none", "rng", "optimizer", "cursor"] = "none"


class FaultSettings(StrictModel):
    kind: Literal["none", "worker_exit", "save_interrupt", "corrupt", "hang"] = "none"
    step: int | None = Field(default=None, ge=1)
    rank: int = Field(default=0, ge=0)
    require_committed_step: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_trigger(self) -> FaultSettings:
        if (self.kind == "none") != (self.step is None):
            raise ValueError("fault step must be set exactly when a fault kind is selected")
        return self


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
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    fault: FaultSettings = Field(default_factory=FaultSettings)

    @model_validator(mode="after")
    def validate_fault(self) -> ProjectConfig:
        if self.fault.rank >= self.run.world_size:
            raise ValueError("fault rank must be within world size")
        if self.fault.step is not None and self.fault.step > self.training.total_steps:
            raise ValueError("fault step must not exceed total steps")
        if self.fault.kind in {"save_interrupt", "corrupt"} and (
            self.checkpoint.mode == "none"
            or (self.fault.step % self.checkpoint.interval_steps != 0
                and self.fault.step != self.training.total_steps)
        ):
            raise ValueError("save fault must target an enabled checkpoint boundary")
        return self

    @staticmethod
    def _digest(value: dict) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        return self._digest(self.model_dump())

    def workload_fingerprint(self) -> str:
        return self._digest(
            {"seed": self.run.seed, "world_size": self.run.world_size,
             "training": self.training.model_dump(), "model": self.model.model_dump()}
        )

    def data_fingerprint(self) -> str:
        return self._digest(
            {"seed": self.run.seed, "world_size": self.run.world_size,
             "sequence_length": self.training.sequence_length,
             "batch_size_per_rank": self.training.batch_size_per_rank,
             "vocab_size": self.model.vocab_size, "generator": "sample-id-v1"}
        )


def load_config(path: Path) -> ProjectConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")
    return ProjectConfig.model_validate(raw)
