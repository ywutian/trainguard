"""Source archives exclude generated run output and reject added members."""

from __future__ import annotations

import copy
import importlib.util
import io
import subprocess
import tarfile
import tomllib
import uuid
from pathlib import Path

import pytest


def _verifier():
    source = Path(__file__).parents[1] / "scripts" / "verify_sdist.py"
    spec = importlib.util.spec_from_file_location("verify_sdist", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_distribution_excludes_generated_output_and_rejects_injection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    probe = root / f"release-output-probe-{uuid.uuid4().hex}"
    probe.mkdir()
    try:
        (probe / "unreviewed.json").write_text('{"sample": "private"}')
        subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
            cwd=root, capture_output=True, text=True, check=True,
        )
    finally:
        (probe / "unreviewed.json").unlink()
        probe.rmdir()
    archive_path = tmp_path / f"trainguard-{version}.tar.gz"
    verifier = _verifier()
    assert verifier.verify_sdist(root, archive_path, version) > 0
    with tarfile.open(archive_path, "r:gz") as archive:
        assert not any("release-output-probe" in member.name for member in archive)
        assert not any("/verification/" in member.name for member in archive)
        assert not any("/docs/experiments/" in member.name for member in archive)

    injected = tmp_path / "injected.tar.gz"
    with tarfile.open(archive_path, "r:gz") as reviewed, tarfile.open(
        injected, "w:gz"
    ) as altered:
        for member in reviewed:
            altered.addfile(member, reviewed.extractfile(member))
        payload = b'{"sample": "private"}'
        extra = tarfile.TarInfo(f"trainguard-{version}/verification/raw.json")
        extra.size = len(payload)
        altered.addfile(extra, io.BytesIO(payload))
    with pytest.raises(ValueError, match="file set differs"):
        verifier.verify_sdist(root, injected, version)

    altered_metadata = tmp_path / "altered-metadata.tar.gz"
    with tarfile.open(archive_path, "r:gz") as reviewed, tarfile.open(
        altered_metadata, "w:gz"
    ) as altered:
        for member in reviewed:
            if member.name.endswith("/PKG-INFO"):
                content = reviewed.extractfile(member).read() + b"Changed: true\n"
                replacement = copy.copy(member)
                replacement.size = len(content)
                altered.addfile(replacement, io.BytesIO(content))
            else:
                altered.addfile(member, reviewed.extractfile(member))
    with pytest.raises(ValueError, match="complete source distribution differs"):
        verifier.verify_sdist(root, altered_metadata, version)


def test_source_distribution_preflight_rejects_untracked_eligible_file() -> None:
    root = Path(__file__).parents[1]
    probe = root / "examples" / f"unreviewed-{uuid.uuid4().hex}.txt"
    try:
        probe.write_text("unreviewed build input")
        with pytest.raises(ValueError, match="untracked source distribution input"):
            _verifier().verify_build_inputs(root)
    finally:
        probe.unlink(missing_ok=True)
