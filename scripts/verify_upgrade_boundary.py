"""Exercise separate locked versions against one interrupted, recoverable run."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path


def _run(command: list[str], cwd: Path, environment: dict[str, str], ok: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, env=environment, capture_output=True, text=True, check=False
    )
    if ok and result.returncode:
        raise RuntimeError(f"upgrade check command failed: {result.stderr[-1500:]}")
    return result


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_argument(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("expected prior artifact digest must be lowercase SHA-256")
    return value


def _directory_digest(root: Path) -> dict[str, str]:
    digests = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("old run contains an unexpected symbolic link")
        if path.is_file():
            digests[path.relative_to(root).as_posix()] = _digest(path)
    return digests


def _environment(
    sandbox: Path, name: str, source: Path, wheel: Path, environ: dict[str, str]
) -> Path:
    virtual_environment = sandbox / name
    _run(["uv", "venv", "--python", sys.executable, str(virtual_environment)], sandbox, environ)
    python = virtual_environment / "bin" / "python"
    requirements = sandbox / f"{name}-requirements.txt"
    _run(
        ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--format",
         "requirements.txt", "--output-file", str(requirements)], source, environ
    )
    active_environment = {**environ, "VIRTUAL_ENV": str(virtual_environment)}
    _run(["uv", "sync", "--locked", "--no-dev", "--no-install-project", "--active"],
         source, active_environment)
    _run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        sandbox, environ
    )
    return virtual_environment / "bin" / "trainguard"


def _new_runtime_profile(executable: Path, sandbox: Path, environ: dict[str, str]) -> dict:
    python = executable.parent / "python"
    profile_code = (
        "import importlib.metadata as m, json, torch; "
        "print(json.dumps({'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'packages': sorted({d.metadata['Name'].lower().replace('_', '-') "
        "for d in m.distributions()})}))"
    )
    profile = json.loads(_run([
        str(python), "-c", profile_code,
    ], sandbox, environ).stdout)
    if sys.platform == "linux" and (
        profile["torch_version"] != "2.14.0+cpu"
        or profile["torch_cuda_version"] is not None
        or any(name.startswith(("nvidia-", "cuda-")) or name == "triton"
               for name in profile["packages"])
    ):
        raise RuntimeError("new Linux environment contains a non-CPU PyTorch closure")
    return profile


def _interrupted_run(executable: Path, config: Path, sandbox: Path, environ: dict[str, str]) -> Path:
    runs = sandbox / "interrupted-runs"
    with (sandbox / "interrupted-launcher.txt").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(executable), "run", "--config", str(config), "--output-root", str(runs)],
            cwd=sandbox, env=environ, stdout=log, stderr=subprocess.STDOUT, text=True
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if list(runs.glob("*/checkpoints/*/COMMITTED")):
                    if process.poll() is not None:
                        raise RuntimeError("previous run exited before interruption")
                    process.send_signal(signal.SIGINT)
                    break
                if process.poll() is not None:
                    raise RuntimeError("previous run ended before a committed checkpoint")
                time.sleep(0.05)
            else:
                raise RuntimeError("previous run did not commit a checkpoint in time")
            process.wait(timeout=30)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=30)
    run_dirs = list(runs.iterdir())
    if len(run_dirs) != 1:
        raise RuntimeError("previous version did not create one run")
    run_dir = run_dirs[0]
    if (
        json.loads((run_dir / "run.json").read_text())["status"] != "INTERRUPTED"
        or not list(run_dir.glob("checkpoints/*/COMMITTED"))
        or (run_dir / "summary.json").exists()
    ):
        raise RuntimeError("previous run is not interrupted with a committed checkpoint")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-ref", required=True)
    parser.add_argument("--expected-previous-commit", required=True)
    parser.add_argument("--expected-previous-wheel-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-previous-lock-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--current-wheel", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    environ = os.environ.copy()
    environ.pop("PYTHONPATH", None)
    environ.pop("VIRTUAL_ENV", None)
    environ.pop("UV_PROJECT_ENVIRONMENT", None)
    resolved_previous = _run(
        ["git", "rev-parse", "--verify", f"{args.previous_ref}^{{commit}}"], root, environ
    ).stdout.strip()
    if resolved_previous != args.expected_previous_commit:
        raise RuntimeError("previous ref differs from the approved release commit")
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", resolved_previous, "HEAD"],
        root, environ, ok=False,
    )
    if ancestor.returncode:
        raise RuntimeError("approved previous release is not in candidate history")
    current_wheel = args.current_wheel.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="trainguard-upgrade-") as temporary:
        sandbox = Path(temporary)
        source = sandbox / "previous-source"
        source.mkdir()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", args.previous_ref],
            cwd=root, env=environ, capture_output=True, check=True
        )
        with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as stream:
            stream.extractall(source, filter="data")
        previous_version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
        current_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        try:
            old_parts = tuple(int(part) for part in previous_version.split("."))
            new_parts = tuple(int(part) for part in current_version.split("."))
        except ValueError as exc:
            raise RuntimeError("upgrade rehearsal needs numeric release versions") from exc
        if len(old_parts) != 3 or len(new_parts) != 3 or (
            old_parts[:2] != new_parts[:2] or old_parts[2] + 1 != new_parts[2]
        ):
            raise RuntimeError("previous ref must be the immediately preceding patch release")
        previous_dist = sandbox / "previous-dist"
        _run(["uv", "build", "--wheel", "--out-dir", str(previous_dist)], source, environ)
        previous_wheels = list(previous_dist.glob("*.whl"))
        if len(previous_wheels) != 1:
            raise RuntimeError("previous ref did not produce one wheel")
        previous_wheel = previous_wheels[0]
        if _digest(previous_wheel) != args.expected_previous_wheel_sha256:
            raise RuntimeError("previous release wheel differs from approved artifact")
        if _digest(source / "uv.lock") != args.expected_previous_lock_sha256:
            raise RuntimeError("previous release lock differs from approved artifact")
        old_executable = _environment(sandbox, "old-environment", source, previous_wheel, environ)
        new_executable = _environment(sandbox, "new-environment", root, current_wheel, environ)
        new_profile = _new_runtime_profile(new_executable, sandbox, environ)
        config = sandbox / "workload.json"
        config.write_text(json.dumps({
            "run": {"seed": 42, "world_size": 2, "backend": "gloo", "device": "cpu"},
            "training": {"total_steps": 40, "sequence_length": 16,
                         "batch_size_per_rank": 2, "dataloader_workers": 0},
            "model": {"vocab_size": 128, "hidden_size": 64, "num_heads": 4,
                      "num_layers": 1, "dropout": 0.0},
            "checkpoint": {"mode": "sync", "interval_steps": 1},
            "recovery": {"max_restarts": 2, "progress_timeout_seconds": 30},
        }))
        run_dir = _interrupted_run(old_executable, config, sandbox, environ)
        before = _directory_digest(run_dir)
        refusal = _run([str(new_executable), "resume", str(run_dir)], sandbox, environ, ok=False)
        if refusal.returncode == 0 or "saved run source or runtime source_sha256 differs" not in refusal.stderr:
            raise RuntimeError("new version did not explicitly reject the interrupted old run")
        if _directory_digest(run_dir) != before:
            raise RuntimeError("rejected resume changed prior run data")
        _run([str(old_executable), "resume", str(run_dir)], sandbox, environ)
        if json.loads((run_dir / "run.json").read_text())["status"] != "SUCCEEDED":
            raise RuntimeError("old environment did not complete interrupted run")
        _run(
            [str(old_executable), "run", "--config", str(config), "--output-root",
             str(sandbox / "reference-runs")], sandbox, environ
        )
        reference_dirs = list((sandbox / "reference-runs").iterdir())
        if len(reference_dirs) != 1:
            raise RuntimeError("reference run is missing")
        _run(
            [str(old_executable), "validate", "--reference", str(reference_dirs[0]),
             "--recovered", str(run_dir)], sandbox, environ
        )
        print(json.dumps({
            "previous_ref": args.previous_ref,
            "previous_commit_sha": resolved_previous,
            "previous_version": previous_version,
            "current_version": current_version,
            "previous_wheel_sha256": _digest(previous_wheel),
            "previous_lock_sha256": _digest(source / "uv.lock"),
            "current_wheel_sha256": _digest(current_wheel),
            "current_lock_sha256": _digest(root / "uv.lock"),
            "new_torch_version": new_profile["torch_version"],
            "new_torch_cuda_version": new_profile["torch_cuda_version"],
            "new_linux_cpu_profile_checked": sys.platform == "linux",
            "new_version_rejected_interrupted_old_run": True,
            "old_run_files_unchanged_after_rejection": len(before),
            "old_locked_environment_resumed_exactly": True,
        }, sort_keys=True))


if __name__ == "__main__":
    main()
