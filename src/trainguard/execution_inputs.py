"""Stable identity of every file that can define the local verification run."""

from __future__ import annotations

import hashlib
from pathlib import Path

INPUT_DIRECTORIES = ("src", "tests", "scripts", "configs")
INPUT_FILES = ("pyproject.toml", "uv.lock")
GENERATED_DIRECTORIES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
GENERATED_SUFFIXES = {".pyc", ".pyo"}


def execution_inputs_sha256(root: Path) -> str:
    """Hash canonical paths and bytes, ignoring only generated interpreter caches."""
    root = root.resolve()
    paths = []
    for name in INPUT_DIRECTORIES:
        directory = root / name
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"execution input directory is missing or linked: {name}")
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if any(part in GENERATED_DIRECTORIES for part in relative.parts):
                continue
            if path.suffix in GENERATED_SUFFIXES:
                continue
            if path.is_symlink():
                raise ValueError(f"execution input is linked: {relative.as_posix()}")
            if path.is_file():
                paths.append(path)
            elif not path.is_dir():
                raise ValueError(f"execution input has unsupported type: {relative.as_posix()}")
    for name in INPUT_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"execution input is missing or linked: {name}")
        paths.append(path)
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
