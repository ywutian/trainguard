"""Check downloaded hosted Linux CPU evidence for the current candidate commit.

The input directories must be downloaded independently from one trusted workflow
run. This offline check proves internal consistency; it does not authenticate the
origin of those downloaded files or validate a customer environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from check_release_readiness import (
    _previous_release,
    _validate_pytest_result,
    package_source_sha256,
)
from run_simulation_closure import _acceptance_complete

from trainguard.events import write_json_atomic

PYTHON_LANES = ("3.11", "3.12")
REQUIRED_GATES = {
    "static", "tests", "cpu-acceptance", "package", "wheel", "fresh-install",
    "upgrade-boundary",
}
WORKFLOW_NAME = "Verify recovery package"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"evidence record is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"evidence record is not a mapping: {path.name}")
    return value


def _single_result_dir(directory: Path) -> Path:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("hosted summary directory is missing or linked")
    matches = list(directory.rglob("result.json"))
    if len(matches) != 1 or matches[0].is_symlink():
        raise ValueError("hosted summary must contain exactly one result")
    return matches[0].parent


def _package_files(directory: Path, wheel_name: str, sdist_name: str) -> tuple[Path, Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("hosted package directory is missing or linked")
    candidates = [path for path in directory.rglob("*") if path.is_file() and (
        path.name.endswith(".whl") or path.name.endswith(".tar.gz")
    )]
    names = [path.name for path in candidates]
    if len(candidates) != 2 or sorted(names) != sorted((wheel_name, sdist_name)) or any(
        path.is_symlink() for path in candidates
    ):
        raise ValueError("hosted package set differs from the candidate")
    wheel = next(path for path in candidates if path.name == wheel_name)
    sdist = next(path for path in candidates if path.name == sdist_name)
    return wheel, sdist


def _last_json(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"hosted gate output is empty: {path.name}")
    try:
        value = json.loads(lines[-1])
    except ValueError as exc:
        raise ValueError(f"hosted gate output is not JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"hosted gate output is not a mapping: {path.name}")
    return value


def _junit_counts(directory: Path) -> dict[str, int]:
    path = directory / "pytest.xml"
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise ValueError("hosted JUnit evidence is unreadable") from exc
    cases = list(root.iter("testcase"))
    gpu_skipped = [case for case in cases if case.get("classname") == "tests.test_gpu_acceptance"
                   and case.find("skipped") is not None]
    skipped = [case for case in cases if case.find("skipped") is not None]
    details = {"tests": {
        "passed": len(cases) - len(skipped),
        "skipped_no_gpu": len(gpu_skipped),
        "skipped_device_preflight": 0,
    }}
    _validate_pytest_result(directory, details)
    if len(gpu_skipped) != 4 or len(skipped) != 4:
        raise ValueError("hosted CPU lane has an unexpected skip set")
    return details["tests"]


def _workflow_jobs(metadata: dict, commit: str) -> tuple[int, dict[str, dict]]:
    if (
        metadata.get("headSha") != commit
        or metadata.get("conclusion") != "success"
        or metadata.get("event") != "push"
        or metadata.get("workflowName") != WORKFLOW_NAME
        or type(metadata.get("databaseId")) is not int
        or metadata["databaseId"] < 1
        or not isinstance(metadata.get("jobs"), list)
    ):
        raise ValueError("workflow run does not identify a successful current-commit run")
    jobs = metadata["jobs"]
    expected = {f"cpu-and-package ({lane})" for lane in PYTHON_LANES}
    if len(jobs) != len(expected) or any(not isinstance(job, dict) for job in jobs) or {
        job.get("name") for job in jobs
    } != expected or any(
        job.get("status") != "completed" or job.get("conclusion") != "success"
        or type(job.get("databaseId")) is not int or job["databaseId"] < 1
        for job in jobs
    ):
        raise ValueError("workflow Python 3.11 and 3.12 jobs are not both successful")
    return metadata["databaseId"], {job["name"].removeprefix("cpu-and-package (").removesuffix(")"): job
                                    for job in jobs}


def verify_evidence(
    metadata: dict,
    summary_directories: dict[str, Path],
    package_directories: dict[str, Path],
    wheel: Path,
    sdist: Path,
    *,
    commit: str,
    version: str,
    source_sha256: str,
    lock_sha256: str,
    previous_release: dict,
) -> dict:
    """Return a scoped report only when both downloaded matrix lanes agree."""
    if set(summary_directories) != set(PYTHON_LANES) or set(package_directories) != set(
        PYTHON_LANES
    ):
        raise ValueError("both hosted Python lanes are required")
    if (
        wheel.name != f"trainguard-{version}-py3-none-any.whl"
        or sdist.name != f"trainguard-{version}.tar.gz"
        or not wheel.is_file()
        or not sdist.is_file()
        or not zipfile.is_zipfile(wheel)
        or not tarfile.is_tarfile(sdist)
    ):
        raise ValueError("local candidate packages are missing or malformed")
    artifact_sha256 = {wheel.name: _sha256(wheel), sdist.name: _sha256(sdist)}
    workflow_run_id, jobs = _workflow_jobs(metadata, commit)
    matrix = {}
    for lane in PYTHON_LANES:
        directory = _single_result_dir(summary_directories[lane])
        result = _mapping(directory / "result.json")
        hosted_wheel, hosted_sdist = _package_files(
            package_directories[lane], wheel.name, sdist.name
        )
        if {
            wheel.name: _sha256(hosted_wheel), sdist.name: _sha256(hosted_sdist)
        } != artifact_sha256:
            raise ValueError(f"Python {lane} hosted package bytes differ from the local candidate")
        gates = result.get("gates")
        if (
            result.get("status") != "SUCCEEDED"
            or result.get("version") != version
            or result.get("source_sha256") != source_sha256
            or result.get("lock_sha256") != lock_sha256
            or result.get("artifact_sha256") != artifact_sha256
            or not isinstance(result.get("python"), str)
            or not result["python"].startswith(f"{lane}.")
            or not isinstance(result.get("platform"), str)
            or "linux" not in result["platform"].lower()
            or not isinstance(gates, list)
            or len(gates) != len(REQUIRED_GATES)
            or any(not isinstance(gate, dict) for gate in gates)
            or {gate.get("name") for gate in gates} != REQUIRED_GATES
            or any(
                gate.get("exit_code") != 0 or gate.get("timed_out", False) is not False
                or Path(gate.get("output", "")).name != f"{gate['name']}.txt"
                or not (directory / f"{gate['name']}.txt").is_file()
                for gate in gates
            )
        ):
            raise ValueError(f"Python {lane} hosted result has incomplete or mismatched gates")
        acceptance = result.get("acceptance")
        if not isinstance(acceptance, dict) or not _acceptance_complete(acceptance):
            raise ValueError(f"Python {lane} hosted CPU fault matrix is incomplete")
        settings = acceptance.get("config", {}).get("run", {})
        environment = acceptance.get("environment", {})
        if (
            not isinstance(settings, dict)
            or settings.get("device") != "cpu"
            or settings.get("backend") != "gloo"
            or settings.get("world_size") != 2
            or not isinstance(environment, dict)
            or environment.get("source_sha256") != source_sha256
            or environment.get("git_commit") != commit
            or not isinstance(environment.get("python"), str)
            or not environment["python"].startswith(f"{lane}.")
        ):
            raise ValueError(f"Python {lane} hosted acceptance used another environment")
        tests = _junit_counts(directory)
        verified_wheel = _last_json(directory / "wheel.txt")
        installed = _last_json(directory / "fresh-install.txt")
        upgrade = _last_json(directory / "upgrade-boundary.txt")
        if (
            verified_wheel.get("passed") is not True
            or verified_wheel.get("version") != version
            or verified_wheel.get("source_sha256") != source_sha256
            or type(verified_wheel.get("sdist_members")) is not int
            or verified_wheel["sdist_members"] < 1
            or installed.get("version") != version
            or installed.get("wheel_sha256") != artifact_sha256[wheel.name]
            or any(installed.get(field) is not True for field in (
                "installed_outside_checkout", "completed_run", "recovered_run_matches_reference",
                "support_export_checked", "run_data_preserved_after_uninstall",
            ))
            or installed.get("attributed_faults") != 1
            or installed.get("recoveries") != 1
            or type(installed.get("run_data_files_checked")) is not int
            or installed["run_data_files_checked"] < 1
            or upgrade.get("current_version") != version
            or upgrade.get("current_wheel_sha256") != artifact_sha256[wheel.name]
            or upgrade.get("current_lock_sha256") != lock_sha256
            or upgrade.get("previous_commit_sha") != previous_release["git_commit"]
            or upgrade.get("previous_version") != previous_release["version"]
            or upgrade.get("previous_wheel_sha256") != previous_release["wheel_sha256"]
            or upgrade.get("previous_lock_sha256") != previous_release["lock_sha256"]
            or upgrade.get("new_version_rejected_interrupted_old_run") is not True
            or upgrade.get("old_locked_environment_resumed_exactly") is not True
            or type(upgrade.get("old_run_files_unchanged_after_rejection")) is not int
            or upgrade["old_run_files_unchanged_after_rejection"] < 1
        ):
            raise ValueError(f"Python {lane} hosted package or upgrade evidence is incomplete")
        matrix[lane] = {
            "job_id": jobs[lane]["databaseId"],
            "result_sha256": _sha256(directory / "result.json"),
            "junit_sha256": _sha256(directory / "pytest.xml"),
            "tests": tests,
            "cpu_acceptance_cases_passed": len(acceptance["cases"]),
            "artifacts": artifact_sha256,
        }
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "HOSTED_LINUX_EVIDENCE_CONSISTENT",
        "scope": "hosted Linux x86_64 CPU/Gloo and package verification; downloaded evidence only",
        "candidate_git_commit": commit,
        "candidate_version": version,
        "candidate_source_sha256": source_sha256,
        "lock_sha256": lock_sha256,
        "workflow_run_id": workflow_run_id,
        "workflow_event": metadata["event"],
        "download_origin_authenticated": False,
        "matrix": matrix,
        "artifacts": artifact_sha256,
        "customer_environment_validated": False,
        "production_release_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-metadata", type=Path, required=True)
    for lane, label in (("3.11", "python311"), ("3.12", "python312")):
        parser.add_argument(f"--{label}-summary", type=Path, required=True)
        parser.add_argument(f"--{label}-packages", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.report.resolve()
    if output.is_relative_to(root):
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", str(output)], cwd=root, check=False
        )
        if ignored.returncode:
            parser.error("report inside the checkout must be ignored")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise ValueError("current candidate checkout is not clean")
        version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
        previous = _previous_release(
            root, _mapping(root / "docs/commercial/release-gates.json").get("previous_release"),
            version,
        )
        report = verify_evidence(
            _mapping(args.workflow_metadata),
            {"3.11": args.python311_summary, "3.12": args.python312_summary},
            {"3.11": args.python311_packages, "3.12": args.python312_packages},
            args.wheel, args.sdist,
            commit=commit, version=version, source_sha256=package_source_sha256(root),
            lock_sha256=_sha256(root / "uv.lock"), previous_release=previous,
        )
    except (OSError, ValueError, TypeError, KeyError, ElementTree.ParseError,
            subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        report = {
            "schema_version": 1,
            "checked_at": datetime.now(UTC).isoformat(),
            "status": "INVALID",
            "reason": str(exc),
            "customer_environment_validated": False,
            "production_release_authorized": False,
        }
    write_json_atomic(output, report)
    print(report["status"])
    return 0 if report["status"] == "HOSTED_LINUX_EVIDENCE_CONSISTENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
