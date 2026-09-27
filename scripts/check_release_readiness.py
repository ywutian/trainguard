"""Fail closed unless every commercial release gate has source-bound evidence."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from trainguard.events import write_json_atomic

REQUIRED_GATES = {
    "local_package", "local_cpu", "customer_workload", "persistent_checkpoint",
    "cross_host_fencing", "real_gpu_matrix", "security_operations",
    "commercial_contract", "paid_pilot", "supported_matrix", "sustained_operations",
}
LOCAL_RAW_FILES = {
    "result.json", "pytest.xml", "acceptance.json", "static.txt", "tests.txt",
    "cpu-acceptance.txt", "package.txt", "wheel.txt", "fresh-install.txt",
    "upgrade-boundary.txt",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_source_sha256(root: Path) -> str:
    source = root / "src" / "trainguard"
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _archive_source_sha256(items: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(items.items()):
        digest.update(name.encode())
        digest.update(content)
    return digest.hexdigest()


def _verify_artifacts(root: Path, wheel: Path, sdist: Path, source_digest: str) -> None:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if (
        not wheel.name.startswith(f"trainguard-{version}-")
        or not wheel.name.endswith(".whl")
        or sdist.name != f"trainguard-{version}.tar.gz"
        or not zipfile.is_zipfile(wheel)
        or not tarfile.is_tarfile(sdist)
    ):
        raise ValueError("release artifacts have invalid names or archive formats")
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()
        if len(files) != len(set(files)):
            raise ValueError("wheel contains duplicate members")
        code = {
            name.removeprefix("trainguard/"): archive.read(name)
            for name in files if name.startswith("trainguard/") and name.endswith(".py")
        }
        metadata = f"trainguard-{version}.dist-info/METADATA"
        template = "trainguard/templates/cpu_demo.yaml"
        if metadata not in files or template not in files:
            raise ValueError("wheel metadata or packaged template is missing")
        details = archive.read(metadata).decode("utf-8")
        if "Name: trainguard\n" not in details or f"Version: {version}\n" not in details:
            raise ValueError("wheel package metadata differs")
        if archive.read(template) != (root / "src" / template).read_bytes():
            raise ValueError("wheel template differs from source")
        entry_points = f"trainguard-{version}.dist-info/entry_points.txt"
        record_name = f"trainguard-{version}.dist-info/RECORD"
        if entry_points not in files or record_name not in files:
            raise ValueError("wheel entry point or record is missing")
        if archive.read(entry_points).decode("utf-8").strip() != (
            "[console_scripts]\ntrainguard = trainguard.cli:app"
        ):
            raise ValueError("wheel command entry point differs")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        if len(rows) != len(files) or {row[0] for row in rows} != set(files):
            raise ValueError("wheel record does not cover all members")
        for row in rows:
            if len(row) != 3:
                raise ValueError("wheel record row is malformed")
            name, encoded, size = row
            if name == record_name:
                if encoded or size:
                    raise ValueError("wheel self record is malformed")
                continue
            content = archive.read(name)
            expected = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).rstrip(b"=").decode("ascii")
            if encoded != expected or size != str(len(content)):
                raise ValueError("wheel record digest differs")
    if _archive_source_sha256(code) != source_digest:
        raise ValueError("wheel package code differs from source")
    prefix = f"trainguard-{version}/"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("source distribution contains duplicate members")
        code = {
            member.name.removeprefix(prefix + "src/trainguard/"): archive.extractfile(member).read()
            for member in members
            if member.isfile() and member.name.startswith(prefix + "src/trainguard/")
            and member.name.endswith(".py")
        }
        lock = archive.extractfile(prefix + "uv.lock")
        if lock is None or lock.read() != (root / "uv.lock").read_bytes():
            raise ValueError("source distribution dependency lock differs")
    if _archive_source_sha256(code) != source_digest:
        raise ValueError("source distribution package code differs from source")
    with tempfile.TemporaryDirectory(prefix="release-rebuild-") as temporary:
        result = subprocess.run(
            ["uv", "build", "--wheel", "--sdist", "--out-dir", temporary],
            cwd=root, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise ValueError("candidate source cannot be rebuilt")
        for submitted in (wheel, sdist):
            rebuilt = Path(temporary) / submitted.name
            if not rebuilt.is_file() or _digest(submitted) != _digest(rebuilt):
                raise ValueError(f"{submitted.name}: complete artifact differs from current source build")


def _local_evidence(root: Path, details: dict, source_digest: str,
                    artifacts: dict[str, str]) -> None:
    location = details.get("raw_evidence_dir")
    hashes = details.get("raw_files")
    if not isinstance(location, str) or not isinstance(hashes, dict) or set(hashes) != LOCAL_RAW_FILES:
        raise ValueError("local evidence does not contain the required raw result files")
    raw_dir = (root / location).resolve()
    if not raw_dir.is_relative_to(root) or not raw_dir.is_dir():
        raise ValueError("local raw evidence directory is invalid")
    for name in LOCAL_RAW_FILES:
        path = raw_dir / name
        if path.is_symlink() or not path.is_file() or _digest(path) != hashes[name]:
            raise ValueError(f"local raw evidence is missing or changed: {name}")
    result = json.loads((raw_dir / "result.json").read_text(encoding="utf-8"))
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    expected_gates = {
        "static", "tests", "cpu-acceptance", "package", "wheel", "fresh-install",
        "upgrade-boundary",
    }
    gates = result.get("gates")
    if (
        result.get("status") != "SUCCEEDED"
        or result.get("source_sha256") != source_digest
        or result.get("version") != version
        or not isinstance(result.get("artifact_sha256"), dict)
        or {name: digest for name, digest in result["artifact_sha256"].items()
            if name.endswith((".whl", ".tar.gz"))} != {
            name: digest for name, digest in artifacts.items() if name != "uv.lock"
        }
        or result.get("lock_sha256") != artifacts["uv.lock"]
        or not isinstance(gates, list)
        or len(gates) != len(expected_gates)
        or {gate.get("name") for gate in gates if isinstance(gate, dict)} != expected_gates
        or any(
            gate.get("exit_code") != 0
            or Path(gate.get("output", "")).name != f"{gate['name']}.txt"
            for gate in gates
        )
    ):
        raise ValueError("local raw result does not prove all required gates")
    suite = ElementTree.parse(raw_dir / "pytest.xml").getroot()
    suites = [suite] if suite.tag == "testsuite" else list(suite.findall("testsuite"))
    tests = sum(int(item.get("tests", 0)) for item in suites)
    failures = sum(int(item.get("failures", 0)) for item in suites)
    errors = sum(int(item.get("errors", 0)) for item in suites)
    if tests < 100 or failures or errors:
        raise ValueError("local raw test suite is incomplete or failing")
    acceptance = json.loads((raw_dir / "acceptance.json").read_text(encoding="utf-8"))
    expected_cases = {
        "sync-worker_exit", "async-worker_exit", "sync-save_interrupt",
        "async-save_interrupt", "sync-corrupt", "async-corrupt", "sync-hang",
        "omit-rng", "omit-optimizer", "omit-cursor",
    }
    cases = acceptance.get("cases")
    if (
        acceptance.get("status") != "SUCCEEDED"
        or acceptance.get("reference_status") != "VALIDATED"
        or result.get("acceptance") != acceptance
        or not isinstance(cases, list)
        or len(cases) != len(expected_cases)
        or not all(isinstance(case, dict) and isinstance(case.get("name"), str)
                   for case in cases)
        or {case["name"] for case in cases} != expected_cases
        or any(
            case.get("status") != "PASSED"
            or case.get("recovery_count") != 1
            or case.get("fault_attributed") is not True
            or not isinstance(case.get("validation"), dict)
            or case["validation"].get("passed") is not (
                not case["name"].startswith("omit-")
            )
            for case in cases
        )
    ):
        raise ValueError("local raw CPU acceptance matrix is incomplete")
    wheel_name = next(name for name in artifacts if name.endswith(".whl"))
    wheel_result = json.loads((raw_dir / "wheel.txt").read_text(encoding="utf-8").splitlines()[-1])
    install = json.loads((raw_dir / "fresh-install.txt").read_text(encoding="utf-8").splitlines()[-1])
    upgrade = json.loads((raw_dir / "upgrade-boundary.txt").read_text(encoding="utf-8").splitlines()[-1])
    if (
        wheel_result.get("passed") is not True
        or wheel_result.get("version") != version
        or wheel_result.get("source_sha256") != source_digest
        or install.get("version") != version
        or install.get("wheel_sha256") != artifacts[wheel_name]
        or install.get("installed_outside_checkout") is not True
        or install.get("completed_run") is not True
        or install.get("recovered_run_matches_reference") is not True
        or install.get("support_export_checked") is not True
        or install.get("run_data_preserved_after_uninstall") is not True
        or install.get("run_data_files_checked", 0) < 1
        or install.get("attributed_faults") != 1
        or install.get("recoveries") != 1
        or upgrade.get("current_version") != version
        or upgrade.get("current_wheel_sha256") != artifacts[wheel_name]
        or upgrade.get("current_lock_sha256") != artifacts["uv.lock"]
        or upgrade.get("new_version_rejected_interrupted_old_run") is not True
        or upgrade.get("old_locked_environment_resumed_exactly") is not True
        or upgrade.get("old_run_files_unchanged_after_rejection", 0) < 1
    ):
        raise ValueError("local install or upgrade raw evidence is incomplete")


def _receipt(path: Path, gate_id: str, source_digest: str, artifacts: dict[str, str]) -> dict:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{gate_id}: evidence receipt is unreadable") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("gate_id") != gate_id
        or receipt.get("decision") != "PASS"
        or receipt.get("candidate_source_sha256") != source_digest
        or receipt.get("artifacts") != artifacts
        or not isinstance(receipt.get("scope"), str)
        or not receipt["scope"]
        or not isinstance(receipt.get("reviewer_role"), str)
        or not receipt["reviewer_role"]
        or not isinstance(receipt.get("approval_reference"), str)
        or not receipt["approval_reference"]
        or not isinstance(receipt.get("record_reference"), str)
        or not receipt["record_reference"]
        or not isinstance(receipt.get("record_sha256"), str)
        or len(receipt["record_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in receipt["record_sha256"])
        or not isinstance(receipt.get("reviewed_at"), str)
    ):
        raise ValueError(f"{gate_id}: evidence receipt lacks version, scope or approval binding")
    try:
        datetime.fromisoformat(receipt["reviewed_at"])
    except ValueError as exc:
        raise ValueError(f"{gate_id}: evidence review time is invalid") from exc
    if gate_id in {"local_package", "local_cpu"}:
        checks = receipt.get("checks")
        required = (
            {"wheel_identity", "fresh_install", "upgrade_recovery"}
            if gate_id == "local_package" else {"full_suite", "cpu_fault_matrix"}
        )
        if not isinstance(checks, dict) or not all(checks.get(name) is True for name in required):
            raise ValueError(f"{gate_id}: local gate checks are incomplete")
        root = Path(__file__).resolve().parents[1]
        record = (root / receipt["record_reference"]).resolve()
        if not record.is_relative_to(root) or not record.is_file():
            raise ValueError(f"{gate_id}: local evidence record is missing")
        if _digest(record) != receipt["record_sha256"]:
            raise ValueError(f"{gate_id}: local evidence record digest differs")
        details = json.loads(record.read_text(encoding="utf-8"))
        if not isinstance(details, dict) or not isinstance(details.get("checks"), dict):
            raise ValueError(f"{gate_id}: local evidence record has an invalid shape")
        if (
            details.get("schema_version") != 1
            or details.get("status") != "SUCCEEDED"
            or details.get("candidate_source_sha256") != source_digest
            or details.get("artifacts") != artifacts
            or details["checks"].get("full_suite") is not True
            or details["checks"].get("cpu_fault_matrix") is not True
            or details["checks"].get("wheel_identity") is not True
            or details["checks"].get("fresh_install") is not True
            or details["checks"].get("upgrade_recovery") is not True
        ):
            raise ValueError(f"{gate_id}: local evidence record does not prove its checks")
        _local_evidence(root, details, source_digest, artifacts)
    return receipt


def evaluate(manifest: Path, wheel: Path, sdist: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("release gate manifest schema is invalid")
    gates = data.get("gates")
    if (
        not isinstance(gates, list)
        or any(not isinstance(gate, dict) for gate in gates)
        or {gate.get("id") for gate in gates} != REQUIRED_GATES
    ):
        raise ValueError("release gate list is incomplete or duplicated")
    if len(gates) != len(REQUIRED_GATES):
        raise ValueError("release gate list contains duplicates")
    source_digest = package_source_sha256(root)
    if data.get("candidate_source_sha256") != source_digest:
        raise ValueError("release gate manifest does not match current package source")
    for path in (wheel, sdist, root / "uv.lock"):
        if not path.is_file():
            raise ValueError("required release artifact is missing")
    _verify_artifacts(root, wheel, sdist, source_digest)
    artifacts = {wheel.name: _digest(wheel), sdist.name: _digest(sdist),
                 "uv.lock": _digest(root / "uv.lock")}

    checked = []
    for gate in gates:
        state = gate.get("status")
        if state not in {"PASS", "FAIL", "BLOCKED"}:
            raise ValueError("release gate has an invalid status")
        record = {"id": gate["id"], "status": state}
        if state == "PASS":
            location = gate.get("evidence")
            if not isinstance(location, str) or not location:
                raise ValueError(f"{gate['id']}: passing gate lacks evidence")
            raw_path = root / location
            path = raw_path.resolve()
            if not path.is_relative_to(root) or not path.is_file() or raw_path.is_symlink():
                raise ValueError(f"{gate['id']}: evidence path is invalid")
            digest = _digest(path)
            if digest != gate.get("sha256"):
                raise ValueError(f"{gate['id']}: evidence digest differs")
            _receipt(path, gate["id"], source_digest, artifacts)
            record.update(evidence=location, sha256=digest)
        elif not isinstance(gate.get("reason"), str) or not gate["reason"]:
            raise ValueError(f"{gate['id']}: nonpassing gate lacks a reason")
        else:
            record["reason"] = gate["reason"]
        checked.append(record)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip())
    receipts_complete = all(gate["status"] == "PASS" for gate in checked) and not dirty
    evaluation_allowed = (
        not dirty
        and all(gate["status"] == "PASS" for gate in checked
                if gate["id"] in {"local_package", "local_cpu"})
        and not any(gate["status"] == "FAIL" for gate in checked)
    )
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "REVIEW_REQUIRED" if receipts_complete else "BLOCKED",
        "evaluation_allowed": evaluation_allowed,
        "decision_scope": "evidence_manifest_integrity_only",
        "production_release_authorized": False,
        "candidate_source_sha256": source_digest,
        "git_commit": commit,
        "git_dirty": dirty,
        "artifacts": artifacts,
        "gates": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("docs/commercial/release-gates.json"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.wheel, args.sdist)
    except (
        OSError, ValueError, KeyError, TypeError, IndexError, StopIteration,
        ElementTree.ParseError, tarfile.TarError, zipfile.BadZipFile,
    ) as exc:
        report = {"schema_version": 1, "checked_at": datetime.now(UTC).isoformat(), "status": "INVALID",
                  "reason": str(exc)}
    write_json_atomic(args.report, report)
    print(report["status"])
    return {"REVIEW_REQUIRED": 2, "INVALID": 1, "BLOCKED": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
