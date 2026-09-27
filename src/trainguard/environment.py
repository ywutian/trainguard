"""Reproducible source identity and non-secret runtime measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def installed_distributions() -> list[dict[str, str | None]]:
    """Record the installed package metadata used by this interpreter."""
    packages = []
    names = set()
    for distribution in importlib.metadata.distributions():
        original_name = distribution.metadata.get("Name")
        version = distribution.version
        record = distribution.read_text("RECORD")
        if not original_name or not version or not record or not record.strip():
            raise ValueError("installed package identity is incomplete")
        name = re.sub(r"[-_.]+", "-", original_name).lower()
        if name in names:
            raise ValueError(f"installed package identity is ambiguous: {name}")
        names.add(name)
        direct_url = distribution.read_text("direct_url.json")
        if direct_url is not None:
            try:
                origin = json.loads(direct_url)
            except json.JSONDecodeError as exc:
                raise ValueError(f"installed package origin is invalid: {name}") from exc
            if not isinstance(origin, dict):
                raise ValueError(f"installed package origin is invalid: {name}")
            directory_info = origin.get("dir_info", {})
            if not isinstance(directory_info, dict):
                raise ValueError(f"installed package origin is invalid: {name}")
            if directory_info.get("editable") and name != "trainguard":
                raise ValueError(f"editable dependency has no frozen source identity: {name}")
        packages.append(
            {
                "name": name,
                "version": version,
                "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
                "direct_url_sha256": (
                    hashlib.sha256(direct_url.encode("utf-8")).hexdigest()
                    if direct_url is not None
                    else None
                ),
            }
        )
    return sorted(packages, key=lambda package: package["name"])


def source_sha256() -> str:
    source = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    # Package bytes have the same identity in a checkout and an installed wheel.
    # Installed package metadata and Python are checked separately at resume.
    for path in sorted(source.rglob("*.py")):
        name = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    return digest.hexdigest()


def environment_snapshot(world_size: int, device: str, storage_path: Path) -> dict:
    repository = Path(__file__).resolve().parents[2]
    checkout_source = repository / "src" / "trainguard" / "environment.py"
    is_checkout = checkout_source.resolve() == Path(__file__).resolve()

    def git(*arguments):
        if not is_checkout:
            return None
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
    git_status = git("status", "--porcelain")
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
        "installed_distributions": installed_distributions(),
        "source_sha256": source_sha256(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "environment_options": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
