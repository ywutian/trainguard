"""Reject generated or untracked files in a source distribution."""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT_FILES = {
    ".gitignore", "LICENSE", "README.md", "pyproject.toml", "uv.lock",
    "build-requirements.in", "build-constraints.txt",
}
ROOT_DIRECTORIES = {"configs", "examples", "scripts", "src", "tests"}
DOCUMENTS = {
    "docs/commercial/customer-pilot-template.md",
    "docs/commercial/market-evidence-2026-09-26.md",
    "docs/commercial/operations-runbook.md",
    "docs/commercial/pilot-ledger-template.json",
    "docs/plans/product-closure-2026-09-26.md",
}


def _selected(relative: str) -> bool:
    if relative in ROOT_FILES or relative in DOCUMENTS:
        return True
    return "/" in relative and relative.split("/", 1)[0] in ROOT_DIRECTORIES


def _git_paths(root: Path, *options: str) -> set[str]:
    output = subprocess.run(
        ["git", "ls-files", *options, "-z"], cwd=root, capture_output=True, check=True
    ).stdout
    return {relative for relative in output.decode("utf-8").split("\0") if relative}


def verify_build_inputs(root: Path) -> None:
    """Fail before building if an eligible source path has not been reviewed."""
    untracked = sorted(
        relative for relative in _git_paths(root, "--others", "--exclude-standard")
        if _selected(relative)
    )
    if untracked:
        raise ValueError(f"untracked source distribution input: {untracked[:3]}")


def verify_sdist(root: Path, archive_path: Path, version: str) -> int:
    """Compare every source member with the tracked, explicitly selected checkout files."""
    root = root.resolve()
    verify_build_inputs(root)
    tracked = _git_paths(root)
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
    with tempfile.TemporaryDirectory(prefix="source-rebuild-") as directory:
        rebuilt = subprocess.run(
            ["uv", "build", "--sdist", "--build-constraints",
             "build-constraints.txt", "--require-hashes", "--out-dir", directory],
            cwd=root, capture_output=True, text=True, check=False,
        )
        if rebuilt.returncode:
            raise ValueError("source distribution cannot be rebuilt")
        rebuilt_path = Path(directory) / archive_path.name
        if not rebuilt_path.is_file() or hashlib.sha256(archive_path.read_bytes()).digest() != (
            hashlib.sha256(rebuilt_path.read_bytes()).digest()
        ):
            raise ValueError("complete source distribution differs from reviewed build")
    return len(members)
