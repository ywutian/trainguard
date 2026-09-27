"""Install the wheel in a clean environment and exercise its public entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def _run(command: list[str], cwd: Path, environment: dict[str, str]) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"installation check failed ({result.returncode}): {' '.join(command[:3])}\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return result.stdout


def _directory_digest(root: Path) -> dict[str, str]:
    digest = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("run data contains an unexpected symbolic link")
        if path.is_file():
            digest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    with tempfile.TemporaryDirectory(prefix="trainguard-install-") as temporary:
        root = Path(temporary)
        virtual_environment = root / "environment"
        _run(["uv", "venv", "--python", sys.executable, str(virtual_environment)], root, environment)
        python = virtual_environment / "bin" / "python"
        executable = virtual_environment / "bin" / "trainguard"
        requirements = root / "requirements.txt"
        _run(
            ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--format",
             "requirements.txt", "--output-file", str(requirements)],
            project_root,
            environment,
        )
        _run(
            ["uv", "pip", "install", "--python", str(python), "--require-hashes", "-r",
             str(requirements)], root, environment
        )
        _run(
            ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
            root,
            environment,
        )
        version = _run([str(executable), "version"], root, environment).strip()
        config = root / "cpu.yaml"
        _run([str(executable), "init-config", "--output", str(config)], root, environment)
        _run([str(executable), "validate-config", "--config", str(config)], root, environment)
        runs = root / "customer-runs"
        _run(
            [str(executable), "run", "--config", str(config), "--output-root", str(runs)],
            root,
            environment,
        )
        reference_dirs = list(runs.iterdir())
        if len(reference_dirs) != 1 or not (reference_dirs[0] / "summary.json").is_file():
            raise RuntimeError("installed CLI did not create one completed reference run")
        recovery_config = root / "recovery.yaml"
        recovery_settings = yaml.safe_load(config.read_text(encoding="utf-8"))
        recovery_settings["checkpoint"] = {"mode": "sync", "interval_steps": 1}
        recovery_settings["recovery"] = {"max_restarts": 2, "progress_timeout_seconds": 30}
        recovery_settings["fault"] = {"kind": "worker_exit", "step": 2, "rank": 0}
        recovery_config.write_text(yaml.safe_dump(recovery_settings), encoding="utf-8")
        _run(
            [str(executable), "run", "--config", str(recovery_config), "--output-root",
             str(runs), "--allow-experiment"], root, environment
        )
        recovered_dirs = [path for path in runs.iterdir() if path != reference_dirs[0]]
        if len(recovered_dirs) != 1:
            raise RuntimeError("installed CLI did not create one recovered run")
        recovered = recovered_dirs[0]
        run_id = json.loads((recovered / "run.json").read_text(encoding="utf-8"))["run_id"]
        with sqlite3.connect((recovered / "run.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
            attempts = db.execute(
                "SELECT status, resume_step FROM attempts WHERE run_id=? ORDER BY number", (run_id,)
            ).fetchall()
            recoveries = db.execute(
                "SELECT resume_step FROM recoveries WHERE run_id=?", (run_id,)
            ).fetchall()
        events = [
            json.loads(line) for line in
            (recovered / "attempts/attempt-001/rank-0.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        injections = [event for event in events if event.get("event_type") == "fault_injected"]
        if (
            attempts != [("FAILED", 0), ("SUCCEEDED", 1)]
            or recoveries != [(1,)]
            or len(injections) != 1
            or injections[0].get("fault_kind") != "worker_exit"
            or injections[0].get("global_step") != 2
            or not list(recovered.glob("checkpoints/*/COMMITTED"))
        ):
            raise RuntimeError("installed recovery lacks one attributed fault and checkpoint resume")
        _run(
            [str(executable), "validate", "--reference", str(reference_dirs[0]),
             "--recovered", str(recovered)], root, environment
        )
        support = root / "support.json"
        _run(
            [str(executable), "support-bundle", str(recovered), "--output",
             str(support)], root, environment
        )
        if "sample_ids" in support.read_text():
            raise RuntimeError("support bundle copied raw sample identifiers")
        before = _directory_digest(runs)
        _run(["uv", "pip", "uninstall", "--python", str(python), "trainguard"], root, environment)
        if executable.exists() or _directory_digest(runs) != before:
            raise RuntimeError("uninstall removed the run data or left the entry point installed")
        print(json.dumps({
            "version": version,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "installed_outside_checkout": True,
            "completed_run": True,
            "recovered_run_matches_reference": True,
            "attributed_faults": 1,
            "recoveries": 1,
            "support_export_checked": True,
            "run_data_preserved_after_uninstall": True,
            "run_data_files_checked": len(before),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
