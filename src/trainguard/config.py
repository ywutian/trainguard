"""Validated fixed-topology training and recovery configuration."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunSettings(StrictModel):
    profile: Literal["experiment", "guarded"] = "experiment"
    seed: int = Field(default=42, ge=0)
    world_size: int = Field(default=2, ge=1)
    backend: Literal["gloo", "nccl"] = "gloo"
    device: Literal["cpu", "cuda"] = "cpu"
    strategy: Literal["ddp", "fsdp2"] = "ddp"
    deterministic: bool = True
    timeout_seconds: int = Field(default=600, ge=1)


class TrainingSettings(StrictModel):
    total_steps: int = Field(ge=1)
    sequence_length: int = Field(ge=2, le=512)
    batch_size_per_rank: int = Field(ge=1)
    dataloader_workers: int = Field(default=0, ge=0, le=16)
    prefetch_factor: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    max_consecutive_skips: int = Field(default=10, ge=1)
    loss_log_interval: int = Field(default=10, ge=1)


class CheckpointSettings(StrictModel):
    mode: Literal["none", "sync", "async"] = "none"
    interval_steps: int = Field(default=1, ge=1)
    poll_interval_steps: int = Field(default=5, ge=1)
    save_timeout_seconds: int = Field(default=120, ge=1)
    keep_last_k: int | None = Field(default=None, ge=2)
    max_retained_bytes: int | None = Field(default=None, ge=1)
    max_checkpoint_bytes: int | None = Field(default=None, ge=1)
    min_free_bytes: int | None = Field(default=None, ge=1)
    max_event_log_bytes: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_retention(self) -> CheckpointSettings:
        if self.max_retained_bytes is not None and self.keep_last_k is None:
            raise ValueError("max_retained_bytes requires keep_last_k")
        return self


class RecoverySettings(StrictModel):
    max_restarts: int = Field(default=2, ge=0)
    startup_timeout_seconds: int = Field(default=120, ge=1)
    progress_timeout_seconds: int = Field(default=120, ge=1)
    omit_state: Literal["none", "rng", "optimizer", "cursor", "stream", "extra"] = "none"


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


class DataSettings(StrictModel):
    kind: Literal["synthetic", "jsonl"] = "synthetic"
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    shuffle: bool = True
    random_crop: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> DataSettings:
        if self.kind == "jsonl" and (self.path is None or self.sha256 is None):
            raise ValueError("JSONL data requires an immutable path and SHA-256")
        if self.kind == "synthetic" and (self.path is not None or self.sha256 is not None):
            raise ValueError("synthetic data cannot specify an external source")
        return self


class WorkloadDependency(StrictModel):
    module: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkloadDataFile(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExternalWorkloadSettings(StrictModel):
    version: Literal[1, 2] = 1
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependencies: list[WorkloadDependency] = Field(default_factory=list)
    data_files: list[WorkloadDataFile] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_inputs(self) -> ExternalWorkloadSettings:
        if self.version == 1 and (self.dependencies or self.data_files):
            raise ValueError("external workload v1 has no bound dependency or data files")
        if self.version == 2 and (not self.dependencies or not self.data_files):
            raise ValueError("external workload v2 requires declared dependencies and data files")
        if len({item.module for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("external workload dependency modules must be unique")
        if any(
            item.module in sys.stdlib_module_names or item.module in {"torch", "numpy"}
            for item in self.dependencies
        ):
            raise ValueError("external workload dependency cannot shadow a runtime module")
        if len({item.name for item in self.data_files}) != len(self.data_files):
            raise ValueError("external workload data names must be unique")
        return self


class ProjectConfig(StrictModel):
    run: RunSettings
    training: TrainingSettings
    model: ModelSettings = Field(default_factory=ModelSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)
    recovery: RecoverySettings = Field(default_factory=RecoverySettings)
    fault: FaultSettings = Field(default_factory=FaultSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    external_workload: ExternalWorkloadSettings | None = None

    @model_validator(mode="after")
    def validate_fault(self) -> ProjectConfig:
        if self.external_workload is not None:
            if self.run.device != "cpu" or self.run.strategy != "ddp":
                raise ValueError(f"external workload v{self.external_workload.version} requires CPU DDP")
            if self.data.kind != "synthetic":
                raise ValueError(f"external workload v{self.external_workload.version} owns its data stream")
            if self.external_workload.version == 2 and (
                self.run.world_size != 2 or self.run.backend != "gloo"
                or self.training.precision != "fp32"
                or self.training.dataloader_workers != 0
            ):
                raise ValueError("external workload v2 requires two CPU/Gloo ranks, FP32 and no data workers")
        if self.recovery.omit_state in {"stream", "extra"} and (
            self.external_workload is None or self.external_workload.version != 2
        ):
            raise ValueError("stream/extra omission requires external workload v2")
        if self.run.profile == "guarded":
            if self.fault.kind != "none" or self.recovery.omit_state != "none":
                raise ValueError("guarded runs forbid fault injection and omitted recovery state")
            if self.checkpoint.mode == "none":
                raise ValueError("guarded runs require an enabled checkpoint mode")
            if any(
                value is None for value in (
                    self.checkpoint.keep_last_k,
                    self.checkpoint.max_retained_bytes,
                    self.checkpoint.max_checkpoint_bytes,
                    self.checkpoint.min_free_bytes,
                    self.checkpoint.max_event_log_bytes,
                )
            ):
                raise ValueError("guarded runs require checkpoint, retention, free-space and event budgets")
            if self.checkpoint.max_retained_bytes < 2 * self.checkpoint.max_checkpoint_bytes:
                raise ValueError("guarded retention budget must hold two maximum-size checkpoints")
            if self.checkpoint.interval_steps >= self.training.total_steps:
                raise ValueError("guarded runs require two distinct checkpoint boundaries")
        if self.run.strategy == "fsdp2" and self.run.device != "cuda":
            raise ValueError("FSDP2 requires CUDA")
        if (self.run.device == "cuda") != (self.run.backend == "nccl"):
            raise ValueError("CUDA requires NCCL; CPU requires Gloo")
        if self.training.precision == "fp16" and self.run.device != "cuda":
            raise ValueError("FP16 scaler training requires CUDA")
        if self.fault.require_committed_step is not None and (
            self.checkpoint.mode == "none"
            or self.fault.step is None
            or self.fault.require_committed_step >= self.fault.step
        ):
            raise ValueError("required committed step must precede an enabled fault")
        if self.fault.rank >= self.run.world_size:
            raise ValueError("fault rank must be within world size")
        if self.fault.step is not None and self.fault.step > self.training.total_steps:
            raise ValueError("fault step must not exceed total steps")
        if self.fault.kind in {"save_interrupt", "corrupt"} and (
            self.checkpoint.mode == "none"
            or (
                self.fault.step % self.checkpoint.interval_steps != 0
                and self.fault.step != self.training.total_steps
            )
        ):
            raise ValueError("save fault must target an enabled checkpoint boundary")
        return self

    @staticmethod
    def _digest(value: dict) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        value = self.model_dump()
        for key in ("max_checkpoint_bytes", "min_free_bytes", "max_event_log_bytes"):
            if value["checkpoint"][key] is None:
                value["checkpoint"].pop(key)
        if self.external_workload is not None and self.external_workload.version == 1:
            value["external_workload"].pop("dependencies")
            value["external_workload"].pop("data_files")
        return self._digest(value)

    def external_identity(self) -> dict | None:
        if self.external_workload is None:
            return None
        value = self.external_workload.model_dump(exclude={"path"})
        if self.external_workload.version == 1:
            value.pop("dependencies")
            value.pop("data_files")
            return value
        for entry in value["dependencies"]:
            entry.pop("path")
        for entry in value["data_files"]:
            entry.pop("path")
        return value

    def workload_fingerprint(self) -> str:
        return self._digest(
            {
                "seed": self.run.seed,
                "world_size": self.run.world_size,
                "device": self.run.device,
                "strategy": self.run.strategy,
                "deterministic": self.run.deterministic,
                "training": self.training.model_dump(),
                "model": self.model.model_dump(),
                "data": self.data.model_dump(exclude={"path"}),
                "external_workload": self.external_identity(),
            }
        )

    def data_fingerprint(self) -> str:
        return self._digest(
            {
                "seed": self.run.seed,
                "world_size": self.run.world_size,
                "sequence_length": self.training.sequence_length,
                "batch_size_per_rank": self.training.batch_size_per_rank,
                "vocab_size": self.model.vocab_size,
                "data": self.data.model_dump(exclude={"path"}),
                "generator": "sample-id-v1"
                if self.data.kind == "synthetic"
                else "epoch-shuffle-v1",
                "external_workload": self.external_identity(),
            }
        )


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
        seen: set[object] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ValueError("YAML merge keys are not supported in run configuration")
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    raise ValueError(f"duplicate YAML configuration key: {key!r}")
                seen.add(key)
            except TypeError as exc:
                raise ValueError("YAML configuration keys must be hashable") from exc
        return super().construct_mapping(node, deep=deep)


def load_config(path: Path) -> ProjectConfig:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.load(stream, Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise TypeError("configuration must be a YAML mapping")
    config = ProjectConfig.model_validate(raw)
    if config.data.path is not None:
        config.data.path = str((path.parent / config.data.path).resolve())
    if config.external_workload is not None:
        config.external_workload.path = str((path.parent / config.external_workload.path).resolve())
        for entry in (*config.external_workload.dependencies, *config.external_workload.data_files):
            entry.path = str((path.parent / entry.path).resolve())
    return config
