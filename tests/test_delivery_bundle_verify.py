"""Transferred evaluation files must match their reviewed artifact references."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from test_supply_chain import supply_fixture


def _module():
    source = Path(__file__).parents[1] / "scripts" / "verify_delivery_bundle.py"
    spec = importlib.util.spec_from_file_location("verify_delivery_bundle", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _seal(root: Path) -> None:
    path = root / "delivery-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = {
        entry.relative_to(root).as_posix(): _sha(entry)
        for entry in root.rglob("*") if entry.is_file() and entry != path
    }
    _write(path, manifest)


def _bundle(root: Path, module) -> Path:
    root.mkdir()
    version = "0.3.2"
    source = "a" * 64
    inputs = "f" * 64
    commit = "b" * 40
    previous = {
        "git_commit": "c" * 40,
        "version": "0.3.1",
        "wheel_sha256": "d" * 64,
        "lock_sha256": "e" * 64,
    }
    wheel = f"trainguard-{version}-py3-none-any.whl"
    sdist = f"trainguard-{version}.tar.gz"
    (root / "uv.lock").write_bytes(b"lock bytes")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "trainguard"\nversion = "{version}"\n', encoding="utf-8"
    )
    for staged_name in module.SOURCE_MEMBERS:
        path = root / staged_name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if staged_name == "scripts/supply_chain.py":
                path.write_bytes((Path(__file__).parents[1] / staged_name).read_bytes())
            else:
                path.write_bytes(f"reviewed {staged_name}".encode())
    with zipfile.ZipFile(root / wheel, "w") as archive:
        prefix = f"trainguard-{version}.dist-info/"
        archive.writestr(prefix + "METADATA", (
            f"Name: trainguard\nVersion: {version}\nLicense-Expression: MIT\n"
            "License-File: LICENSE\n"
        ))
        archive.write(root / "LICENSE", prefix + "licenses/LICENSE")
    with tarfile.open(root / sdist, "w:gz") as archive:
        for staged_name, source_name in module.SOURCE_MEMBERS.items():
            content = (root / staged_name).read_bytes()
            member = tarfile.TarInfo(f"trainguard-{version}/{source_name}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    artifacts = {name: _sha(root / name) for name in (wheel, sdist, "uv.lock")}
    raw_dir = "docs/commercial/evidence/local-0.3.2"
    raw_name = f"{raw_dir}/result.json"
    _write(root / raw_name, {"status": "SUCCEEDED"})
    supply_expected = supply_fixture(root / raw_dir)
    for name in ("supply-chain-installed.json", "supply-chain-sbom.json",
                 "supply-chain-licenses.json", "supply-chain-audit.json"):
        path = root / raw_dir / name
        data = json.loads(path.read_text())
        rows = data if isinstance(data, list) else data.get(
            "components", data.get("dependencies", [])
        )
        for row in rows:
            if row.get("name") == "trainguard":
                row["version"] = version
                if "bom-ref" in row:
                    row["bom-ref"] = f"pkg:pypi/trainguard@{version}"
        if name == "supply-chain-sbom.json":
            data["dependencies"][0]["ref"] = f"pkg:pypi/trainguard@{version}"
        _write(path, data)
    supply_expected.update({
        "candidate_version": version,
        "source_sha256": source, "execution_inputs_sha256": inputs,
        "wheel_sha256": artifacts[wheel], "lock_sha256": artifacts["uv.lock"],
        "tool_lock_sha256": _sha(root / "scripts/supply-chain-tools.txt"),
        "security_policy_sha256": _sha(root / "SECURITY.md"),
        "security_channel_record_sha256": _sha(root / "security-channel-2026-09-27.json"),
        "first_party_license_sha256": _sha(root / "LICENSE"),
    })
    supply_receipt = root / raw_dir / "supply-chain-receipt.json"
    report = json.loads(supply_receipt.read_text())
    report.update(supply_expected)
    report["files"] = {
        name: _sha(root / raw_dir / name) for name in report["files"]
    }
    _write(supply_receipt, report)
    _write(root / raw_dir / "supply-chain.txt", {
        "status": "PASS", "wheel_sha256": artifacts[wheel]
    })
    (root / "requirements.txt").write_bytes(
        (root / raw_dir / "supply-chain-requirements.txt").read_bytes()
    )
    for name in module.BUNDLE_LOCAL_RAW_FILES - {
        "result.json", "supply-chain.txt", "supply-chain-sbom.json",
        "supply-chain-licenses.json", "supply-chain-audit.json",
        "supply-chain-installed.json", "supply-chain-requirements.txt",
        "supply-chain-receipt.json",
    }:
        (root / raw_dir / name).write_text("synthetic raw result\n", encoding="utf-8")
    record_name = "docs/commercial/evidence/local-validation-0.3.2.json"
    _write(root / record_name, {
        "status": "SUCCEEDED", "candidate_source_sha256": source,
        "execution_inputs_sha256": inputs, "execution_commit": commit,
        "artifacts": artifacts, "raw_evidence_dir": raw_dir,
        "raw_files": {name: _sha(root / raw_dir / name)
                      for name in sorted(module.BUNDLE_LOCAL_RAW_FILES)},
    })
    gates = []
    evidence_map = {}
    for gate_id in sorted(module.REQUIRED_GATES):
        if gate_id in {"local_package", "local_cpu"}:
            name = f"docs/commercial/evidence/{gate_id}-0.3.2.json"
            _write(root / name, {
                "gate_id": gate_id, "decision": "PASS",
                "candidate_source_sha256": source, "artifacts": artifacts,
                "execution_inputs_sha256": inputs, "execution_commit": commit,
                "record_reference": record_name,
                "record_sha256": _sha(root / record_name),
            })
            gates.append({"id": gate_id, "status": "PASS", "evidence": name,
                          "sha256": _sha(root / name)})
            evidence_map[gate_id] = name
        else:
            gates.append({"id": gate_id, "status": "BLOCKED", "reason": "pending"})
    _write(root / "release-gates.json", {
        "schema_version": 1, "candidate_source_sha256": source,
        "candidate_execution_inputs_sha256": inputs,
        "previous_release": previous, "gates": gates,
    })
    hosted_artifacts = {name: digest for name, digest in artifacts.items() if name != "uv.lock"}
    hosted = {
        "status": "HOSTED_LINUX_EVIDENCE_CONSISTENT",
        "workflow_run_id": 123,
        "candidate_git_commit": commit,
        "workflow_git_commit": commit,
        "workflow_event": "push",
        "candidate_version": version,
        "candidate_source_sha256": source,
        "candidate_execution_inputs_sha256": inputs,
        "lock_sha256": artifacts["uv.lock"],
        "artifacts": hosted_artifacts,
        "workflow_and_artifacts_fetched_live": True,
        "workflow_repository": "ywutian/trainguard",
        "download_origin_authenticated": False,
        "cryptographic_signature_verified": False,
        "customer_environment_validated": False,
        "production_release_authorized": False,
        "matrix": {lane: {"artifacts": hosted_artifacts} for lane in ("3.11", "3.12")},
    }
    _write(root / "readiness-report.json", {
        "schema_version": 1, "status": "BLOCKED", "evaluation_allowed": True,
        "local_experiment_allowed": True,
        "linux_customer_evaluation_allowed": True,
        "private_vulnerability_reporting_enabled": True,
        "customer_environment_validated": False,
        "hosted_linux_workflow_run_id": 123, "hosted_linux_evidence": hosted,
        "production_release_authorized": False, "git_dirty": False,
        "git_commit": commit, "candidate_source_sha256": source,
        "candidate_execution_inputs_sha256": inputs,
        "artifacts": artifacts, "previous_release": previous, "gates": gates,
    })
    _write(root / "delivery-manifest.json", {
        "schema_version": 1, "version": version, "status": "EVALUATION_ONLY",
        "production_release_authorized": False, "git_commit": commit,
        "source_sha256": source, "execution_inputs_sha256": inputs,
        "hosted_linux_workflow_run_id": 123,
        "evidence_map": evidence_map, "files": {},
    })
    _seal(root)
    return root


def test_transferred_bundle_checks_files_and_evidence_bindings(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    assert module.verify_bundle(root)["files_checked"] == len(
        json.loads((root / "delivery-manifest.json").read_text())["files"]
    )
    (root / "trainguard-0.3.2-py3-none-any.whl").write_bytes(b"changed wheel")
    with pytest.raises(module.DeliveryInvalid, match="file digest differs"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_self_consistent_file_list_with_stale_artifact(
    tmp_path: Path,
) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    report_path = root / "readiness-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifacts"]["uv.lock"] = "0" * 64
    _write(report_path, report)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="release artifacts differ"):
        module.verify_bundle(root)


def test_transferred_bundle_requires_bound_hosted_linux_evidence(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    report_path = root / "readiness-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["private_vulnerability_reporting_enabled"] = False
    _write(report_path, report)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="evaluation identity"):
        module.verify_bundle(root)

    report["private_vulnerability_reporting_enabled"] = True
    report["linux_customer_evaluation_allowed"] = False
    _write(report_path, report)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="evaluation identity"):
        module.verify_bundle(root)

    report["linux_customer_evaluation_allowed"] = True
    report["hosted_linux_evidence"]["cryptographic_signature_verified"] = True
    _write(report_path, report)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="hosted Linux evidence"):
        module.verify_bundle(root)

    report["hosted_linux_evidence"]["cryptographic_signature_verified"] = False
    report["hosted_linux_evidence"]["workflow_repository"] = "another/repository"
    _write(report_path, report)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="hosted Linux evidence"):
        module.verify_bundle(root)

    report["hosted_linux_evidence"]["workflow_repository"] = "ywutian/trainguard"
    report["hosted_linux_evidence"]["artifacts"][next(
        name for name in report["artifacts"] if name.endswith(".whl")
    )] = "0" * 64
    _write(report_path, report)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="hosted Linux package"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_extra_files_and_links(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    extra = root / "unreviewed.txt"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(module.DeliveryInvalid, match="file set differs"):
        module.verify_bundle(root)
    extra.unlink()
    (root / "linked.txt").symlink_to(root / "uv.lock")
    with pytest.raises(module.DeliveryInvalid, match="symbolic link"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_gate_manifest_drift(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    gate_path = root / "release-gates.json"
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    gates["gates"][0]["reason"] = "a different blocker"
    _write(gate_path, gates)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="reason differs"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_prior_release_pin_drift(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    gate_path = root / "release-gates.json"
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    gates["previous_release"]["lock_sha256"] = "0" * 64
    _write(gate_path, gates)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="prior release identity"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_missing_supply_chain_report(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    (root / "docs/commercial/evidence/local-0.3.2/supply-chain-sbom.json").unlink()
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="local gate raw evidence differs"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_changed_supply_chain_report(tmp_path: Path) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    report = root / "docs/commercial/evidence/local-0.3.2/supply-chain-audit.json"
    changed = json.loads(report.read_text())
    changed["dependencies"][1]["vulns"] = [{"id": "TEST-1"}]
    _write(report, changed)
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="local gate raw evidence differs"):
        module.verify_bundle(root)


def test_transferred_bundle_rejects_loose_file_rewritten_after_source_review(
    tmp_path: Path,
) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    (root / "LICENSE").write_bytes(b"changed after source review")
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="differs from reviewed source distribution"):
        module.verify_bundle(root)
