"""Verify the transferred evaluation bundle's file and evidence integrity.

This checks consistency with the bundled manifest. Authenticity still requires a
trusted copy of the manifest digest or an independently reviewed release record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath


class DeliveryInvalid(ValueError):
    """The evaluation bundle is incomplete or internally inconsistent."""


REQUIRED_GATES = {
    "local_package", "local_cpu", "customer_workload", "persistent_checkpoint",
    "cross_host_fencing", "real_gpu_matrix", "security_operations",
    "commercial_contract", "paid_pilot", "supported_matrix", "sustained_operations",
}
HOSTED_REPOSITORY = "ywutian/trainguard"
SOURCE_MEMBERS = {
    "pyproject.toml": "pyproject.toml",
    "uv.lock": "uv.lock",
    "build-requirements.in": "build-requirements.in",
    "build-constraints.txt": "build-constraints.txt",
    "LICENSE": "LICENSE",
    "SECURITY.md": "SECURITY.md",
    "security-channel-2026-09-27.json": "docs/commercial/security-channel-2026-09-27.json",
    "linux-license-evidence-0.3.6.md": "docs/commercial/linux-license-evidence-0.3.6.md",
    "operations-runbook.md": "docs/commercial/operations-runbook.md",
    "customer-pilot-template.md": "docs/commercial/customer-pilot-template.md",
    "pilot-ledger-template.json": "docs/commercial/pilot-ledger-template.json",
    "market-evidence-2026-09-26.md": "docs/commercial/market-evidence-2026-09-26.md",
    "scripts/calculate_pilot_value.py": "scripts/calculate_pilot_value.py",
    "scripts/verify_delivery_bundle.py": "scripts/verify_delivery_bundle.py",
    "scripts/supply_chain.py": "scripts/supply_chain.py",
    "scripts/supply-chain-tools.in": "scripts/supply-chain-tools.in",
    "scripts/supply-chain-tools.txt": "scripts/supply-chain-tools.txt",
}

SUPPLY_CHAIN_RAW_FILES = {
    "supply-chain.txt", "supply-chain-sbom.json", "supply-chain-licenses.json",
    "supply-chain-audit.json", "supply-chain-installed.json",
    "supply-chain-requirements.txt", "supply-chain-receipt.json",
}
BUNDLE_LOCAL_RAW_FILES = SUPPLY_CHAIN_RAW_FILES | {
    "result.json", "pytest.xml", "acceptance.json", "static.txt", "tests.txt",
    "cpu-acceptance.txt", "package.txt", "wheel.txt", "fresh-install.txt",
    "upgrade-boundary.txt",
}


def _verify_supply_chain(root: Path, raw_dir: Path, expected: dict[str, str]) -> None:
    script = root / "scripts/supply_chain.py"
    spec = importlib.util.spec_from_file_location("delivery_supply_chain", script)
    if spec is None or spec.loader is None:
        raise DeliveryInvalid("bundled supply-chain verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    bytecode_setting = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = bytecode_setting
    try:
        module.verify_candidate_license(root / next(
            name for name in _mapping(root / "readiness-report.json")["artifacts"]
            if name.endswith(".whl")
        ), root / "LICENSE")
        module.verify_supply_chain(raw_dir, expected)
    except ValueError as exc:
        raise DeliveryInvalid(f"bundled supply-chain evidence is invalid: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DeliveryInvalid("bundle file name is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise DeliveryInvalid("bundle file name is unsafe or noncanonical")
    return value


def _mapping(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise DeliveryInvalid(f"bundle record is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise DeliveryInvalid(f"bundle record is not a mapping: {path.name}")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def verify_bundle(root: Path, expected_manifest_sha256: str | None = None) -> dict:
    """Check transferred bytes and the already reviewed evidence references."""
    if root.is_symlink() or not root.is_dir():
        raise DeliveryInvalid("bundle directory is missing or linked")
    manifest_path = root / "delivery-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DeliveryInvalid("delivery manifest is missing or linked")
    if expected_manifest_sha256 is not None and (
        not _is_sha256(expected_manifest_sha256)
        or _sha256(manifest_path) != expected_manifest_sha256
    ):
        raise DeliveryInvalid("delivery manifest differs from the trusted digest")
    manifest = _mapping(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "EVALUATION_ONLY"
        or manifest.get("production_release_authorized") is not False
        or not isinstance(files, dict)
        or not files
        or not _is_sha256(manifest.get("source_sha256"))
        or not _is_sha256(manifest.get("execution_inputs_sha256"))
    ):
        raise DeliveryInvalid("delivery manifest has an invalid evaluation identity")
    expected = {}
    for raw_name, digest in files.items():
        name = _relative_name(raw_name)
        if name == manifest_path.name or not _is_sha256(digest):
            raise DeliveryInvalid("delivery manifest contains an invalid file entry")
        expected[name] = digest
    if "requirements.txt" not in expected:
        raise DeliveryInvalid("delivery requirements are missing")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DeliveryInvalid("bundle contains a symbolic link")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise DeliveryInvalid("bundle contains an unsupported filesystem entry")
    if actual != set(expected) | {manifest_path.name}:
        raise DeliveryInvalid("bundle file set differs from the manifest")
    for name, digest in expected.items():
        if _sha256(root / name) != digest:
            raise DeliveryInvalid(f"bundle file digest differs: {name}")

    report = _mapping(root / "readiness-report.json")
    if (
        report.get("schema_version") != 1
        or report.get("evaluation_allowed") is not True
        or report.get("local_experiment_allowed") is not True
        or report.get("linux_customer_evaluation_allowed") is not True
        or report.get("private_vulnerability_reporting_enabled") is not True
        or report.get("customer_environment_validated") is not False
        or report.get("production_release_authorized") is not False
        or report.get("git_dirty") is not False
        or report.get("status") not in {"BLOCKED", "REVIEW_REQUIRED"}
        or report.get("candidate_source_sha256") != manifest["source_sha256"]
        or report.get("candidate_execution_inputs_sha256") != manifest[
            "execution_inputs_sha256"
        ]
        or report.get("git_commit") != manifest.get("git_commit")
        or report.get("hosted_linux_workflow_run_id") != manifest.get(
            "hosted_linux_workflow_run_id"
        )
        or not isinstance(report.get("previous_release"), dict)
    ):
        raise DeliveryInvalid("readiness report differs from the evaluation identity")
    hosted = report.get("hosted_linux_evidence")
    if (
        type(manifest.get("hosted_linux_workflow_run_id")) is not int
        or manifest["hosted_linux_workflow_run_id"] < 1
        or not isinstance(hosted, dict)
        or hosted.get("status") != "HOSTED_LINUX_EVIDENCE_CONSISTENT"
        or hosted.get("workflow_run_id") != manifest["hosted_linux_workflow_run_id"]
        or hosted.get("candidate_git_commit") != manifest["git_commit"]
        or hosted.get("candidate_source_sha256") != manifest["source_sha256"]
        or hosted.get("candidate_execution_inputs_sha256") != manifest[
            "execution_inputs_sha256"
        ]
        or hosted.get("workflow_and_artifacts_fetched_live") is not True
        or hosted.get("workflow_repository") != HOSTED_REPOSITORY
        or hosted.get("download_origin_authenticated") is not False
        or hosted.get("cryptographic_signature_verified") is not False
        or hosted.get("workflow_event") != "push"
        or not isinstance(hosted.get("workflow_git_commit"), str)
        or len(hosted["workflow_git_commit"]) != 40
        or any(character not in "0123456789abcdef"
               for character in hosted["workflow_git_commit"])
        or hosted.get("customer_environment_validated") is not False
        or hosted.get("production_release_authorized") is not False
        or not isinstance(hosted.get("matrix"), dict)
        or set(hosted["matrix"]) != {"3.11", "3.12"}
    ):
        raise DeliveryInvalid("hosted Linux evidence is missing or differs")
    try:
        version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise DeliveryInvalid("bundle project metadata is invalid") from exc
    if version != manifest.get("version"):
        raise DeliveryInvalid("bundle version differs from project metadata")
    if hosted.get("candidate_version") != version:
        raise DeliveryInvalid("hosted Linux candidate version differs")
    previous = report["previous_release"]
    try:
        previous_parts = tuple(int(part) for part in previous["version"].split("."))
        current_parts = tuple(int(part) for part in version.split("."))
    except (KeyError, AttributeError, ValueError) as exc:
        raise DeliveryInvalid("previous release version is invalid") from exc
    if (
        set(previous) != {"git_commit", "version", "wheel_sha256", "lock_sha256"}
        or not isinstance(previous["git_commit"], str)
        or len(previous["git_commit"]) != 40
        or any(character not in "0123456789abcdef" for character in previous["git_commit"])
        or not _is_sha256(previous["wheel_sha256"])
        or not _is_sha256(previous["lock_sha256"])
        or len(previous_parts) != 3
        or len(current_parts) != 3
        or previous_parts[:2] != current_parts[:2]
        or previous_parts[2] + 1 != current_parts[2]
    ):
        raise DeliveryInvalid("previous release identity is invalid")
    artifacts = report.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or len(artifacts) != 3
        or "uv.lock" not in artifacts
        or sum(name.endswith(".whl") for name in artifacts) != 1
        or sum(name.endswith(".tar.gz") for name in artifacts) != 1
        or any(expected.get(name) != digest for name, digest in artifacts.items())
    ):
        raise DeliveryInvalid("release artifacts differ from the readiness report")
    if (
        hosted.get("lock_sha256") != artifacts["uv.lock"]
        or hosted.get("artifacts") != {
            name: digest for name, digest in artifacts.items() if name != "uv.lock"
        }
        or any(
            not isinstance(lane, dict) or lane.get("artifacts") != hosted["artifacts"]
            for lane in hosted["matrix"].values()
        )
    ):
        raise DeliveryInvalid("hosted Linux package evidence differs")

    sdist_name = next(name for name in artifacts if name.endswith(".tar.gz"))
    try:
        with tarfile.open(root / sdist_name, "r:gz") as archive:
            names = archive.getnames()
            if len(names) != len(set(names)):
                raise DeliveryInvalid("source distribution has duplicate members")
            for staged_name, source_name in SOURCE_MEMBERS.items():
                member = archive.getmember(f"trainguard-{version}/{source_name}")
                source_file = archive.extractfile(member) if member.isfile() else None
                if source_file is None or expected.get(staged_name) != hashlib.sha256(
                    source_file.read()
                ).hexdigest():
                    raise DeliveryInvalid(
                        f"bundle file differs from reviewed source distribution: {staged_name}"
                    )
    except (OSError, KeyError, tarfile.TarError) as exc:
        raise DeliveryInvalid("source distribution cannot verify loose delivery files") from exc

    gate_manifest = _mapping(root / "release-gates.json")
    if (
        gate_manifest.get("schema_version") != 1
        or gate_manifest.get("candidate_source_sha256") != manifest["source_sha256"]
        or gate_manifest.get("candidate_execution_inputs_sha256") != manifest[
            "execution_inputs_sha256"
        ]
        or gate_manifest.get("previous_release") != report.get("previous_release")
    ):
        raise DeliveryInvalid("gate manifest has a different source or prior release identity")
    gates = report.get("gates")
    raw_gates = gate_manifest.get("gates")
    evidence_map = manifest.get("evidence_map")
    if (
        not isinstance(gates, list)
        or not isinstance(raw_gates, list)
        or len(gates) != len(raw_gates)
        or not isinstance(evidence_map, dict)
    ):
        raise DeliveryInvalid("gate records are incomplete")
    seen = set()
    passing = set()
    for gate, raw_gate in zip(gates, raw_gates, strict=True):
        if not isinstance(gate, dict) or not isinstance(raw_gate, dict):
            raise DeliveryInvalid("gate record is invalid")
        gate_id = gate.get("id")
        if not isinstance(gate_id, str) or gate_id in seen or (
            gate_id, gate.get("status")
        ) != (raw_gate.get("id"), raw_gate.get("status")):
            raise DeliveryInvalid("gate identity or status differs")
        seen.add(gate_id)
        if gate["status"] != "PASS":
            if (
                gate["status"] not in {"FAIL", "BLOCKED"}
                or not isinstance(gate.get("reason"), str)
                or not gate["reason"]
                or gate["reason"] != raw_gate.get("reason")
            ):
                raise DeliveryInvalid(f"nonpassing gate reason differs: {gate_id}")
            continue
        passing.add(gate_id)
        location = _relative_name(gate.get("evidence"))
        if (
            evidence_map.get(gate_id) != location
            or raw_gate.get("evidence") != location
            or raw_gate.get("sha256") != gate.get("sha256")
            or expected.get(location) != gate.get("sha256")
        ):
            raise DeliveryInvalid(f"passing gate evidence differs: {gate_id}")
        receipt = _mapping(root / location)
        if (
            receipt.get("gate_id") != gate_id
            or receipt.get("decision") != "PASS"
            or receipt.get("candidate_source_sha256") != manifest["source_sha256"]
            or (gate_id in {"local_package", "local_cpu"} and
                receipt.get("execution_inputs_sha256") != manifest[
                    "execution_inputs_sha256"
                ])
            or receipt.get("artifacts") != artifacts
        ):
            raise DeliveryInvalid(f"passing gate receipt differs: {gate_id}")
        if gate_id not in {"local_package", "local_cpu"}:
            continue
        record_name = _relative_name(receipt.get("record_reference"))
        if expected.get(record_name) != receipt.get("record_sha256"):
            raise DeliveryInvalid(f"local gate record differs: {gate_id}")
        record = _mapping(root / record_name)
        if (
            record.get("execution_inputs_sha256") != manifest["execution_inputs_sha256"]
            or record.get("execution_commit") != receipt.get("execution_commit")
        ):
            raise DeliveryInvalid(f"local gate execution identity differs: {gate_id}")
        raw_dir = _relative_name(record.get("raw_evidence_dir"))
        raw_files = record.get("raw_files")
        if not isinstance(raw_files, dict) or set(raw_files) != BUNDLE_LOCAL_RAW_FILES:
            raise DeliveryInvalid(f"local gate raw file list is missing: {gate_id}")
        for raw_name, digest in raw_files.items():
            if expected.get(f"{raw_dir}/{_relative_name(raw_name)}") != digest:
                raise DeliveryInvalid(f"local gate raw evidence differs: {gate_id}")
        wheel_name = next(name for name in artifacts if name.endswith(".whl"))
        _verify_supply_chain(root, root / raw_dir, {
            "candidate_version": version,
            "source_sha256": manifest["source_sha256"],
            "execution_inputs_sha256": manifest["execution_inputs_sha256"],
            "wheel_sha256": artifacts[wheel_name],
            "lock_sha256": artifacts["uv.lock"],
            "tool_lock_sha256": expected["scripts/supply-chain-tools.txt"],
            "security_policy_sha256": expected["SECURITY.md"],
            "first_party_license_sha256": expected["LICENSE"],
            "security_channel_record_sha256": expected[
                "security-channel-2026-09-27.json"
            ],
        })
        supply_result = _mapping(root / raw_dir / "supply-chain-receipt.json")
        if supply_result.get("status") != "PASS":
            raise DeliveryInvalid("bundled supply-chain gate is not clear")
        if gate_id == "local_package" and (
            root / "requirements.txt"
        ).read_bytes() != (root / raw_dir / "supply-chain-requirements.txt").read_bytes():
            raise DeliveryInvalid("delivery requirements differ from scanned candidate inputs")
    if seen != REQUIRED_GATES or set(evidence_map) != passing or not {
        "local_package", "local_cpu"
    } <= passing:
        raise DeliveryInvalid("evaluation gate evidence map is incomplete")
    return {
        "status": "INTEGRITY_CHECKED_EVALUATION_BUNDLE",
        "version": version,
        "files_checked": len(expected),
        "manifest_sha256": _sha256(manifest_path),
        "production_release_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        result = verify_bundle(args.bundle, args.expected_manifest_sha256)
    except DeliveryInvalid as exc:
        print(json.dumps({"status": "INVALID", "reason": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
