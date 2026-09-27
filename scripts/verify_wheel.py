"""Check that a built wheel carries the same runtime identity as the source tree."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

from verify_sdist import verify_sdist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    sdist = args.wheel.with_name(f"trainguard-{expected_version}.tar.gz")
    if not sdist.is_file():
        raise SystemExit("matching source distribution is missing")
    sdist_members = verify_sdist(root, sdist, expected_version)
    from trainguard.environment import source_sha256

    expected_digest = source_sha256()
    # Extract inside the ignored environment to catch accidental Git discovery
    # through a parent checkout when the installed package is not editable.
    temporary_root = root / ".venv" if (root / ".venv").is_dir() else None
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        with zipfile.ZipFile(args.wheel) as archive:
            archive.extractall(directory)
        code = """
import json
from pathlib import Path
import trainguard
from trainguard.environment import environment_snapshot, source_sha256
print(json.dumps({
    'version': trainguard.__version__,
    'source_sha256': source_sha256(),
    'package_path': str(Path(trainguard.__file__).resolve()),
    'git_dirty': environment_snapshot(1, 'cpu', Path('.'))['git_dirty'],
}))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = directory
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        actual = json.loads(result.stdout)
        if (
            actual["version"] != expected_version
            or actual["source_sha256"] != expected_digest
            or not Path(actual["package_path"]).is_relative_to(Path(directory).resolve())
            or actual["git_dirty"] is not None
        ):
            raise SystemExit(f"wheel identity differs from source: {actual}")
    print(json.dumps({
        "version": expected_version, "source_sha256": expected_digest,
        "sdist_members": sdist_members, "passed": True,
    }))


if __name__ == "__main__":
    main()
