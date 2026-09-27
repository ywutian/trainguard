"""Transferred evaluation files must match their reviewed artifact references."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


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
    (root / wheel).write_bytes(b"wheel bytes")
    (root / "uv.lock").write_bytes(b"lock bytes")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "trainguard"\nversion = "{version}"\n', encoding="utf-8"
    )
    for staged_name in module.SOURCE_MEMBERS:
        path = root / staged_name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"reviewed {staged_name}".encode())
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
    record_name = "docs/commercial/evidence/local-validation-0.3.2.json"
    _write(root / record_name, {
        "status": "SUCCEEDED", "candidate_source_sha256": source,
        "execution_inputs_sha256": inputs, "execution_commit": commit,
        "artifacts": artifacts, "raw_evidence_dir": raw_dir,
        "raw_files": {"result.json": _sha(root / raw_name)},
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
    assert module.verify_bundle(root)["files_checked"] == 17
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


def test_transferred_bundle_rejects_loose_file_rewritten_after_source_review(
    tmp_path: Path,
) -> None:
    module = _module()
    root = _bundle(tmp_path / "bundle", module)
    (root / "LICENSE").write_bytes(b"changed after source review")
    _seal(root)
    with pytest.raises(module.DeliveryInvalid, match="differs from reviewed source distribution"):
        module.verify_bundle(root)
