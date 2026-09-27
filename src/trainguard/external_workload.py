"""Versioned, single-file CPU DDP workload adapter for local evaluation.

The adapter is trusted Python code executed during preflight and training, not a
sandboxed plugin. Its digest binds this file only; imported code, data sources and
external side effects require separate inventory and acceptance.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import uuid
from pathlib import Path
from types import ModuleType

from trainguard.config import ProjectConfig
from trainguard.events import sync_directory

FROZEN_WORKLOAD_NAME = "external-workload.py"
_MAX_SOURCE_BYTES = 1024 * 1024


def frozen_workload_path(run_dir: Path) -> Path:
    return run_dir / FROZEN_WORKLOAD_NAME


def read_verified_source(config: ProjectConfig, *, path: Path | None = None) -> bytes:
    """Read one file descriptor once and bind the returned bytes to the configured digest."""
    settings = config.external_workload
    if settings is None:
        raise ValueError("external workload is not configured")
    source_path = path if path is not None else Path(settings.path)
    try:
        descriptor = os.open(
            source_path,
            os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_SOURCE_BYTES:
                raise ValueError("external workload must be a regular file of at most 1 MiB")
            source = stream.read(_MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"external workload is unreadable: {source_path}") from exc
    if len(source) > _MAX_SOURCE_BYTES:
        raise ValueError("external workload must be at most 1 MiB")
    if hashlib.sha256(source).hexdigest() != settings.sha256:
        raise ValueError("external workload SHA-256 differs from configuration")
    return source


def load_verified_workload(config: ProjectConfig, source: bytes, filename: Path) -> ModuleType:
    """Compile the verified byte buffer itself; never import the path after checking it."""
    settings = config.external_workload
    if settings is None or hashlib.sha256(source).hexdigest() != settings.sha256:
        raise ValueError("external workload SHA-256 differs from configuration")
    name = f"_external_workload_{settings.sha256[:12]}_{uuid.uuid4().hex}"
    module = ModuleType(name)
    module.__file__ = str(filename)
    sys.modules[name] = module
    try:
        exec(compile(source, str(filename), "exec"), module.__dict__)  # noqa: S102
        if type(getattr(module, "WORKLOAD_API_VERSION", None)) is not int or (
            module.WORKLOAD_API_VERSION != settings.version
        ):
            raise ValueError("external workload API version differs from configuration")
        for name in ("build_model", "build_stream", "loss"):
            if not callable(getattr(module, name, None)):
                raise TypeError(f"external workload is missing callable {name}")
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def freeze_source(run_dir: Path, source: bytes) -> Path:
    """Persist exactly the bytes approved at preflight before any worker is launched."""
    path = frozen_workload_path(run_dir)
    temporary = run_dir / f".{FROZEN_WORKLOAD_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(run_dir)
    finally:
        temporary.unlink(missing_ok=True)
    return path
