"""Device binding, fixed-topology strategies and local shard digests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel


def preflight(config, *, workload_source: Path | None = None) -> bytes | None:
    if config.run.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < config.run.world_size:
            raise RuntimeError(
                f"CUDA requires {config.run.world_size} visible GPUs; available={torch.cuda.device_count()}"
            )
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("CUDA training requires a PyTorch build with NCCL")
        if config.training.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support BF16")
    if config.data.kind == "jsonl":
        from trainguard.data import read_token_rows

        read_token_rows(config)
    if config.external_workload is not None:
        from trainguard.external_workload import load_verified_workload, read_verified_source

        source = read_verified_source(config, path=workload_source)
        load_verified_workload(
            config, source,
            workload_source if workload_source is not None else Path(config.external_workload.path),
        )
        return source
    return None


def bind_device(config):
    if config.run.device == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


def wrap_model(model, config, device):
    model = model.to(device)
    if config.run.strategy == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (config.run.world_size,))
        for layer in model.encoder.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        return model
    return DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None
    )


def state_digest(value: object) -> str:
    digest = hashlib.sha256()

    def frame(tag: bytes, payload: bytes) -> None:
        digest.update(len(tag).to_bytes(4, "big"))
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def update(item):
        if isinstance(item, torch.Tensor):
            if hasattr(item, "to_local"):
                item = item.to_local()
            frame(b"tensor-dtype", str(item.dtype).encode())
            frame(b"tensor-shape", str(tuple(item.shape)).encode())
            frame(b"tensor-bytes",
                item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(item, dict):
            frame(b"dict-count", str(len(item)).encode())
            for key in sorted(
                item,
                key=lambda value: (type(value).__name__, json.dumps(value, sort_keys=True)),
            ):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            frame(b"list-count" if isinstance(item, list) else b"tuple-count",
                  str(len(item)).encode())
            for part in item:
                update(part)
        else:
            frame(type(item).__name__.encode(), json.dumps(item, sort_keys=True).encode())

    update(value)
    return digest.hexdigest()
