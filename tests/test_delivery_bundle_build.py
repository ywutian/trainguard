"""Delivery construction must preserve the evidence checked before copying."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundle_rejects_candidate_replaced_after_readiness_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    builder = importlib.import_module("build_delivery_bundle")
    wheel = tmp_path / "trainguard-0.3.2-py3-none-any.whl"
    wheel.write_bytes(b"candidate reviewed before copy")
    sdist = tmp_path / "trainguard-0.3.2.tar.gz"
    sdist.write_bytes(b"source reviewed before copy")
    fresh = {
        "checked_at": "2026-09-27T00:00:00+00:00",
        "evaluation_allowed": True,
        "candidate_source_sha256": "a" * 64,
        "git_commit": "b" * 40,
        "gates": [],
        "artifacts": {
            wheel.name: _digest(wheel),
            sdist.name: _digest(sdist),
            "uv.lock": _digest(root / "uv.lock"),
        },
    }
    report = tmp_path / "readiness.json"
    report.write_text(json.dumps(fresh), encoding="utf-8")
    monkeypatch.setattr(builder, "evaluate", lambda *args: fresh)
    monkeypatch.setattr(
        builder.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="locked\n")
    )
    original_copy = shutil.copy2
    replaced = False

    def copy_during_replacement(source: Path, target: Path, *args, **kwargs) -> Path:
        nonlocal replaced
        if Path(source) == wheel and not replaced:
            wheel.write_bytes(b"candidate changed after review")
            replaced = True
        return original_copy(source, target, *args, **kwargs)

    monkeypatch.setattr(builder.shutil, "copy2", copy_during_replacement)
    output = tmp_path / "bundle"
    monkeypatch.setattr(sys, "argv", [
        "build_delivery_bundle", "--wheel", str(wheel), "--sdist", str(sdist),
        "--readiness-report", str(report), "--output-dir", str(output),
    ])
    with pytest.raises(ValueError, match="staged delivery file differs from reviewed evidence"):
        builder.main()
    assert replaced
    assert not output.exists()
