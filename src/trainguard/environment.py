"""Reproducible source identity and non-secret runtime measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def environment_snapshot(world_size: int, device: str, storage_path: Path) -> dict:
    source = Path(__file__).resolve().parent
    repository = source.parents[1]
    digest = hashlib.sha256()
    for path in sorted(source.glob("*.py")) + [
        repository / "pyproject.toml",
        repository / "uv.lock",
    ]:
        if path.is_file():
            digest.update(path.relative_to(repository).as_posix().encode())
            digest.update(path.read_bytes())

    def git(*arguments):
        result = subprocess.run(
            ["git", *arguments], cwd=repository, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    memory = None
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            memory = int(result.stdout)
    elif hasattr(os, "sysconf"):
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(storage_path)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pydantic", "pyyaml", "typer")
    }
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory,
        "world_size": world_size,
        "worker_threads": 1,
        "device": device,
        "storage": "local filesystem",
        "storage_device": storage_path.stat().st_dev,
        "disk_free_bytes": disk.free,
        "versions": versions,
        "source_sha256": digest.hexdigest(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "environment_options": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
