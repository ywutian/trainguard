"""Versioned, trusted CPU DDP workload adapter for local evaluation."""

from __future__ import annotations

import ast
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
FROZEN_INPUTS_NAME = "workload-inputs"


def frozen_workload_path(run_dir: Path) -> Path:
    return run_dir / FROZEN_WORKLOAD_NAME


def frozen_dependency_path(run_dir: Path, module: str) -> Path:
    return run_dir / FROZEN_INPUTS_NAME / f"{module}.py"


def frozen_data_path(run_dir: Path, name: str) -> Path:
    return run_dir / FROZEN_INPUTS_NAME / f"data-{name}.bin"


def _read_bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_SOURCE_BYTES:
                raise ValueError(f"{label} must be a regular file of at most 1 MiB")
            content = stream.read(_MAX_SOURCE_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if len(content) > _MAX_SOURCE_BYTES:
        raise ValueError(f"{label} must be at most 1 MiB")
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs from configuration")
    return content


def read_verified_source(config: ProjectConfig, *, path: Path | None = None) -> bytes:
    """Read one file descriptor once and bind the returned bytes to the configured digest."""
    settings = config.external_workload
    if settings is None:
        raise ValueError("external workload is not configured")
    source_path = path if path is not None else Path(settings.path)
    return _read_bound_file(source_path, settings.sha256, "external workload")


def verify_v2_inputs(config: ProjectConfig, *, run_dir: Path | None = None) -> dict[str, bytes]:
    settings = config.external_workload
    if settings is None or settings.version != 2:
        return {}
    verified = {}
    for entry in settings.dependencies:
        path = frozen_dependency_path(run_dir, entry.module) if run_dir else Path(entry.path)
        verified[entry.module] = _read_bound_file(path, entry.sha256, f"dependency {entry.module}")
    for entry in settings.data_files:
        path = frozen_data_path(run_dir, entry.name) if run_dir else Path(entry.path)
        verified[f"data:{entry.name}"] = _read_bound_file(path, entry.sha256, f"data {entry.name}")
    return verified


def _check_declared_imports(source: bytes, declared: set[str], filename: Path) -> None:
    tree = ast.parse(source, filename=str(filename))
    allowed = sys.stdlib_module_names | {"torch", "numpy"} | declared
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""] if node.level == 0 else [""]
        else:
            continue
        for module in modules:
            if module.split(".")[0] not in allowed:
                raise ValueError(f"undeclared import in {filename.name}: {module}")


def load_verified_workload(
    config: ProjectConfig, source: bytes, filename: Path, *, run_dir: Path | None = None
) -> ModuleType:
    """Compile the verified byte buffer itself; never import the path after checking it."""
    settings = config.external_workload
    if settings is None or hashlib.sha256(source).hexdigest() != settings.sha256:
        raise ValueError("external workload SHA-256 differs from configuration")
    if settings.version == 2:
        inputs = verify_v2_inputs(config, run_dir=run_dir)
        declared = {entry.module for entry in settings.dependencies}
        _check_declared_imports(source, declared, filename)
        for entry in settings.dependencies:
            module_source = inputs[entry.module]
            path = frozen_dependency_path(run_dir, entry.module) if run_dir else Path(entry.path)
            _check_declared_imports(module_source, set(), path)
            dependency = ModuleType(entry.module)
            dependency.__file__ = str(path)
            exec(compile(module_source, str(path), "exec"), dependency.__dict__)  # noqa: S102
            sys.modules[entry.module] = dependency
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
        required = ("build_model", "build_stream", "loss")
        if settings.version == 2:
            required += ("build_optimizer", "build_scheduler", "build_extra_state")
        for name in required:
            if not callable(getattr(module, name, None)):
                raise TypeError(f"external workload is missing callable {name}")
    except BaseException:
        sys.modules.pop(module.__name__, None)
        raise
    return module


def _freeze_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_v2_inputs(config: ProjectConfig, run_dir: Path) -> None:
    settings = config.external_workload
    if settings is None or settings.version != 2:
        return
    inputs = verify_v2_inputs(config)
    for entry in settings.dependencies:
        _freeze_file(frozen_dependency_path(run_dir, entry.module), inputs[entry.module])
    for entry in settings.data_files:
        _freeze_file(frozen_data_path(run_dir, entry.name), inputs[f"data:{entry.name}"])
    verify_v2_inputs(config, run_dir=run_dir)


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
