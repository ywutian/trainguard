"""Reject generated or untracked files in a source distribution."""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path

ROOT_FILES = {".gitignore", "LICENSE", "README.md", "pyproject.toml", "uv.lock"}
ROOT_DIRECTORIES = {"configs", "docs", "examples", "scripts", "src", "tests"}


def _selected(relative: str) -> bool:
    if relative in ROOT_FILES:
        return True
    if relative == "docs/commercial/release-gates.json" or relative.startswith(
        "docs/commercial/evidence/"
    ):
        return False
    return "/" in relative and relative.split("/", 1)[0] in ROOT_DIRECTORIES


def verify_sdist(root: Path, archive_path: Path, version: str) -> int:
    """Compare every source member with the tracked, explicitly selected checkout files."""
    root = root.resolve()
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.decode("utf-8").split("\0")
    selected = {relative for relative in tracked if relative and _selected(relative)}
    prefix = f"trainguard-{version}/"
    expected = {prefix + relative for relative in selected} | {prefix + "PKG-INFO"}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != expected:
            unexpected = sorted(set(names) - expected)
            missing = sorted(expected - set(names))
            raise ValueError(
                f"source distribution file set differs from selected tracked source: "
                f"unexpected={unexpected[:3]}, missing={missing[:3]}"
            )
        for member in members:
            if not member.isfile():
                raise ValueError("source distribution contains a non-file member")
            if member.name == prefix + "PKG-INFO":
                continue
            relative = member.name.removeprefix(prefix)
            source_path = root / relative
            if source_path.is_symlink() or not source_path.is_file():
                raise ValueError(f"selected source file is missing or linked: {relative}")
            content = archive.extractfile(member)
            if content is None or hashlib.sha256(content.read()).digest() != hashlib.sha256(
                source_path.read_bytes()
            ).digest():
                raise ValueError(f"source distribution member differs from checkout: {relative}")
    return len(members)
