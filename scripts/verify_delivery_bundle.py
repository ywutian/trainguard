"""Verify the transferred evaluation bundle's file and evidence integrity.

This checks consistency with the bundled manifest. Authenticity still requires a
trusted copy of the manifest digest or an independently reviewed release record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
SOURCE_MEMBERS = {
    "pyproject.toml": "pyproject.toml",
    "uv.lock": "uv.lock",
    "LICENSE": "LICENSE",
    "operations-runbook.md": "docs/commercial/operations-runbook.md",
    "customer-pilot-template.md": "docs/commercial/customer-pilot-template.md",
    "pilot-ledger-template.json": "docs/commercial/pilot-ledger-template.json",
    "market-evidence-2026-09-26.md": "docs/commercial/market-evidence-2026-09-26.md",
    "scripts/calculate_pilot_value.py": "scripts/calculate_pilot_value.py",
    "scripts/verify_delivery_bundle.py": "scripts/verify_delivery_bundle.py",
}


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


def verify_bundle(root: Path) -> dict:
    """Check transferred bytes and the already reviewed evidence references."""
    if root.is_symlink() or not root.is_dir():
        raise DeliveryInvalid("bundle directory is missing or linked")
    manifest_path = root / "delivery-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DeliveryInvalid("delivery manifest is missing or linked")
    manifest = _mapping(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "EVALUATION_ONLY"
        or manifest.get("production_release_authorized") is not False
        or not isinstance(files, dict)
        or not files
        or not _is_sha256(manifest.get("source_sha256"))
    ):
        raise DeliveryInvalid("delivery manifest has an invalid evaluation identity")
    expected = {}
    for raw_name, digest in files.items():
        name = _relative_name(raw_name)
        if name == manifest_path.name or not _is_sha256(digest):
            raise DeliveryInvalid("delivery manifest contains an invalid file entry")
        expected[name] = digest
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
        or report.get("production_release_authorized") is not False
        or report.get("git_dirty") is not False
        or report.get("status") not in {"BLOCKED", "REVIEW_REQUIRED"}
        or report.get("candidate_source_sha256") != manifest["source_sha256"]
        or report.get("git_commit") != manifest.get("git_commit")
    ):
        raise DeliveryInvalid("readiness report differs from the evaluation identity")
    try:
        version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise DeliveryInvalid("bundle project metadata is invalid") from exc
    if version != manifest.get("version"):
        raise DeliveryInvalid("bundle version differs from project metadata")
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
    ):
        raise DeliveryInvalid("gate manifest has a different source identity")
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
            or receipt.get("artifacts") != artifacts
        ):
            raise DeliveryInvalid(f"passing gate receipt differs: {gate_id}")
        if gate_id not in {"local_package", "local_cpu"}:
            continue
        record_name = _relative_name(receipt.get("record_reference"))
        if expected.get(record_name) != receipt.get("record_sha256"):
            raise DeliveryInvalid(f"local gate record differs: {gate_id}")
        record = _mapping(root / record_name)
        raw_dir = _relative_name(record.get("raw_evidence_dir"))
        raw_files = record.get("raw_files")
        if not isinstance(raw_files, dict):
            raise DeliveryInvalid(f"local gate raw file list is missing: {gate_id}")
        for raw_name, digest in raw_files.items():
            if expected.get(f"{raw_dir}/{_relative_name(raw_name)}") != digest:
                raise DeliveryInvalid(f"local gate raw evidence differs: {gate_id}")
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
    args = parser.parse_args()
    try:
        result = verify_bundle(args.bundle)
    except DeliveryInvalid as exc:
        print(json.dumps({"status": "INVALID", "reason": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
