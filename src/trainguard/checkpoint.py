"""Application-level checkpoint transaction and rank-local training state."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from trainguard import __version__
from trainguard.config import ProjectConfig
from trainguard.environment import source_sha256
from trainguard.events import sync_directory, write_json_atomic


class CheckpointInvalid(ValueError):
    """A checkpoint cannot be used to resume training."""


@dataclass(frozen=True)
class CheckpointRecord:
    path: Path
    global_step: int
    attempt_id: str
    manifest: dict[str, Any]


def candidate_path(run_dir: Path, attempt_id: str, step: int) -> Path:
    return run_dir / "checkpoints" / f"step-{step:06d}-{attempt_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rank_state(
    config: ProjectConfig,
    run_id: str,
    attempt_id: str,
    rank: int,
    global_step: int,
    scheduler_state: dict[str, Any],
    next_data_step: int | None = None,
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "rank": rank,
        "global_step": global_step,
        "next_data_step": global_step * config.training.gradient_accumulation_steps
        if next_data_step is None
        else next_data_step,
        "state_schema_version": 2,
        "optimizer_updates": global_step,
        "consumed_batches": global_step * config.training.gradient_accumulation_steps
        if next_data_step is None
        else next_data_step,
        "accumulation_phase": 0,
        "scaler": None,
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": config.data_fingerprint(),
        "world_size": config.run.world_size,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "torch_version": torch.__version__,
        "scheduler": scheduler_state,
        "rng": {
            "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_cpu": torch.get_rng_state().tolist(),
            **(
                {"torch_cuda": torch.cuda.get_rng_state().tolist()}
                if config.run.device == "cuda"
                else {}
            ),
        },
    }


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def restore_rng(state: dict[str, Any]) -> None:
    rng = state["rng"]
    random.setstate(_as_tuple(rng["python"]))
    numpy = rng["numpy"]
    np.random.set_state(
        (numpy[0], np.array(numpy[1], dtype=np.uint32), numpy[2], numpy[3], numpy[4])
    )
    torch.set_rng_state(torch.tensor(rng["torch_cpu"], dtype=torch.uint8))
    if "torch_cuda" in rng:
        torch.cuda.set_rng_state(torch.tensor(rng["torch_cuda"], dtype=torch.uint8))


def _files(path: Path) -> list[Path]:
    if not path.is_dir() or path.is_symlink():
        raise CheckpointInvalid("checkpoint directory is missing or is a link")
    files = []
    for entry in path.rglob("*"):
        if entry.is_symlink():
            raise CheckpointInvalid("checkpoint contains a symbolic link")
        if entry.is_file() and entry.relative_to(path).as_posix() not in {
            "manifest.json",
            "COMMITTED",
        }:
            files.append(entry)
    return sorted(files)


def _check_rank_states(
    path: Path, config: ProjectConfig, run_id: str, attempt_id: str, step: int
) -> None:
    consumed = []
    for rank in range(config.run.world_size):
        rank_path = path / f"rank-{rank}.json"
        if not rank_path.is_file():
            raise CheckpointInvalid(f"rank {rank} state is missing")
        try:
            state = json.loads(rank_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointInvalid(f"rank {rank} state is unreadable") from exc
        if not isinstance(state, dict):
            raise CheckpointInvalid(f"rank {rank} state is not a mapping")
        expected = {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "rank": rank,
            "global_step": step,
            "state_schema_version": 2,
            "optimizer_updates": step,
            "accumulation_phase": 0,
            "config_fingerprint": config.fingerprint(),
            "data_fingerprint": config.data_fingerprint(),
            "world_size": config.run.world_size,
            "software_version": __version__,
            "source_sha256": source_sha256(),
            "torch_version": torch.__version__,
        }
        for key, value in expected.items():
            if state.get(key) != value or (type(value) is int and type(state.get(key)) is not int):
                raise CheckpointInvalid(f"rank {rank} {key} differs at checkpoint step {step}")
        if not isinstance(state.get("scheduler"), dict) or not isinstance(state.get("rng"), dict):
            raise CheckpointInvalid(f"rank {rank} scheduler or RNG state is missing")
        cursor = state.get("consumed_batches")
        if (
            type(cursor) is not int
            or cursor < step * config.training.gradient_accumulation_steps
            or state.get("next_data_step") != cursor
            or cursor % config.training.gradient_accumulation_steps
        ):
            raise CheckpointInvalid(f"rank {rank} consumed batch boundary is invalid")
        consumed.append(cursor)
        if config.training.precision == "fp16" and not isinstance(state.get("scaler"), dict):
            raise CheckpointInvalid(f"rank {rank} scaler state is missing")
        expected_rng = {"python", "numpy", "torch_cpu"}
        if config.run.device == "cuda":
            expected_rng.add("torch_cuda")
        if set(state["rng"]) != expected_rng:
            raise CheckpointInvalid(f"rank {rank} RNG state is incomplete")
    if len(set(consumed)) != 1:
        raise CheckpointInvalid("rank consumed batch boundaries differ at checkpoint")


def _check_dcp_files(path: Path, config: ProjectConfig) -> None:
    if not (path / "dcp" / ".metadata").is_file():
        raise CheckpointInvalid("DCP metadata is missing")
    if (path / "dcp" / ".metadata").stat().st_size == 0:
        raise CheckpointInvalid("DCP metadata is empty")
    for rank in range(config.run.world_size):
        rank_files = list((path / "dcp").glob(f"__{rank}_*.distcp"))
        if not rank_files or any(entry.stat().st_size == 0 for entry in rank_files):
            raise CheckpointInvalid(f"DCP rank {rank} files are missing")


def _write_marker_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def commit_checkpoint(
    path: Path, config: ProjectConfig, run_id: str, attempt_id: str, step: int
) -> CheckpointRecord:
    if (path / "COMMITTED").exists():
        raise CheckpointInvalid("checkpoint is already committed")
    _check_rank_states(path, config, run_id, attempt_id, step)
    _check_dcp_files(path, config)
    files = [
        {
            "path": entry.relative_to(path).as_posix(),
            "size": entry.stat().st_size,
            "sha256": _sha256(entry),
        }
        for entry in _files(path)
    ]
    manifest = {
        "format_version": 2,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "global_step": step,
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": config.data_fingerprint(),
        "world_size": config.run.world_size,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "torch_version": torch.__version__,
        "files": files,
    }
    # DCP syncs payload files; persist the directory entries before publishing the transaction.
    sync_directory(path / "dcp")
    sync_directory(path)
    sync_directory(path.parent)
    write_json_atomic(path / "manifest.json", manifest)
    marker = _sha256(path / "manifest.json") + "\n"
    _write_marker_atomic(path / "COMMITTED", marker)
    return validate_checkpoint(path, config, run_id)


def validate_checkpoint(path: Path, config: ProjectConfig, run_id: str) -> CheckpointRecord:
    if (
        path.is_symlink()
        or (path / "COMMITTED").is_symlink()
        or (path / "manifest.json").is_symlink()
    ):
        raise CheckpointInvalid("checkpoint transaction contains a symbolic link")
    if not (path / "COMMITTED").is_file():
        raise CheckpointInvalid("checkpoint commit marker is missing")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise CheckpointInvalid("checkpoint manifest is missing")
    if (path / "COMMITTED").read_bytes() != (_sha256(manifest_path) + "\n").encode("ascii"):
        raise CheckpointInvalid("manifest hash differs from commit marker")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CheckpointInvalid("checkpoint manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise CheckpointInvalid("checkpoint manifest is not a mapping")
    expected = {
        "format_version": 2,
        "run_id": run_id,
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": config.data_fingerprint(),
        "world_size": config.run.world_size,
        "software_version": __version__,
        "source_sha256": source_sha256(),
        "torch_version": torch.__version__,
    }
    for key, value in expected.items():
        if manifest.get(key) != value or (
            type(value) is int and type(manifest.get(key)) is not int
        ):
            raise CheckpointInvalid(f"checkpoint {key} differs from requested run or config")
    step = manifest.get("global_step")
    attempt_id = manifest.get("attempt_id")
    if (
        type(step) is not int
        or not 1 <= step <= config.training.total_steps
        or not isinstance(attempt_id, str)
    ):
        raise CheckpointInvalid("checkpoint step or attempt is invalid")
    if path.name != candidate_path(Path(), attempt_id, step).name:
        raise CheckpointInvalid("checkpoint directory name differs from manifest")
    _check_rank_states(path, config, run_id, attempt_id, step)
    _check_dcp_files(path, config)
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise CheckpointInvalid("checkpoint file list is missing")
    actual = {entry.relative_to(path).as_posix() for entry in _files(path)}
    names = set()
    for item in listed:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise CheckpointInvalid("checkpoint file entry is invalid")
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in names:
            raise CheckpointInvalid("checkpoint file path is invalid")
        names.add(item["path"])
        entry = path / relative
        if not entry.is_file() or entry.is_symlink():
            raise CheckpointInvalid(f"checkpoint file {relative} is missing")
        if entry.stat().st_size != item.get("size") or _sha256(entry) != item.get("sha256"):
            raise CheckpointInvalid(f"checkpoint file {relative} size or hash differs")
    if names != actual:
        raise CheckpointInvalid("checkpoint file list differs from directory")
    return CheckpointRecord(path, step, attempt_id, manifest)


def latest_valid_checkpoint(
    run_dir: Path, config: ProjectConfig, run_id: str
) -> CheckpointRecord | None:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return None
    for path in ordered_candidates(run_dir):
        try:
            return validate_checkpoint(path, config, run_id)
        except (CheckpointInvalid, OSError):
            continue
    return None


def ordered_candidates(run_dir: Path) -> list[Path]:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return []

    def order(path: Path) -> tuple[int, str]:
        match = re.fullmatch(r"step-(\d+)-attempt-(\d+)", path.name)
        return (int(match[1]) if match else -1, path.name)

    return sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=order,
        reverse=True,
    )
