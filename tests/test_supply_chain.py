"""Supply-chain receipts reject absent, changed, and incomplete scanner evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


def _module():
    source = Path(__file__).parents[1] / "scripts" / "supply_chain.py"
    spec = importlib.util.spec_from_file_location("supply_chain", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def supply_fixture(root: Path) -> dict[str, str]:
    """Small captured-output shape with a candidate wheel and one dependency."""
    root.mkdir(parents=True, exist_ok=True)
    components = [
        {"name": name, "version": version, "bom-ref": f"pkg:pypi/{name}@{version}",
         "licenses": [{"license": {"id": "MIT"}}]}
        for name, version in (("trainguard", "0.3.6"), ("dependency", "1.0"))
    ]
    _write(root / "supply-chain-installed.json", [
        {"name": item["name"], "version": item["version"]} for item in components
    ])
    _write(root / "supply-chain-sbom.json", {
        "bomFormat": "CycloneDX", "specVersion": "1.6", "components": components,
        "dependencies": [
            {"ref": item["bom-ref"], "dependsOn": []} for item in components
        ],
    })
    _write(root / "supply-chain-licenses.json", {
        "schema_version": 1, "source": "CycloneDX declared package metadata",
        "components": [
            {"name": item["name"], "version": item["version"],
             "licenses": item["licenses"]} for item in components
        ],
    })
    _write(root / "supply-chain-audit.json", {
        "dependencies": [
            {"name": "trainguard", "version": "0.3.6", "vulns": [],
             "skip_reason": "candidate is not published on PyPI"},
            {"name": "dependency", "version": "1.0", "vulns": []},
        ],
        "fixes": [],
    })
    (root / "supply-chain-requirements.txt").write_text(
        "dependency==1.0 --hash=sha256:" + "f" * 64 + "\n", encoding="utf-8"
    )
    expected = {
        "candidate_version": "0.3.6",
        "source_sha256": "a" * 64,
        "execution_inputs_sha256": "b" * 64,
        "wheel_sha256": "c" * 64,
        "lock_sha256": "d" * 64,
        "tool_lock_sha256": "e" * 64,
        "security_policy_sha256": "f" * 64,
        "security_channel_record_sha256": "1" * 64,
        "first_party_license_sha256": "2" * 64,
    }
    _write(root / "supply-chain-receipt.json", {
        "schema_version": 1, "status": "PASS", "audit_service": "pypi",
        "audit_exit_code": 0, "database_snapshot_available": False,
        "uv_version": "uv 0.9.10 (test)",
        "generated_at_utc": "2026-09-27T00:00:00+00:00",
        "sbom_tool_version": "7.4.0", "audit_tool_version": "2.10.1",
        "component_count": 2, "third_party_count": 1,
        "known_vulnerability_count": 0, "unscanned_third_party": [],
        "unlicensed_third_party": [], **expected,
        "files": {name: _sha(root / name) for name in sorted(_module().RAW_FILES)},
    })
    return expected


def _reseal(root: Path) -> None:
    receipt = json.loads((root / "supply-chain-receipt.json").read_text(encoding="utf-8"))
    receipt["files"] = {name: _sha(root / name) for name in sorted(_module().RAW_FILES)}
    _write(root / "supply-chain-receipt.json", receipt)


def test_supply_chain_receipt_rejects_missing_and_changed_files(tmp_path: Path) -> None:
    module = _module()
    expected = supply_fixture(tmp_path / "evidence")
    root = tmp_path / "evidence"
    assert module.verify_supply_chain(root, expected)["third_party_count"] == 1
    (root / "supply-chain-licenses.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(module.SupplyChainInvalid, match="evidence differs"):
        module.verify_supply_chain(root, expected)
    (root / "supply-chain-licenses.json").unlink()
    with pytest.raises(module.SupplyChainInvalid, match="evidence differs"):
        module.verify_supply_chain(root, expected)


@pytest.mark.parametrize("change, message", [
    ("hidden_vulnerability", "not clear"),
    ("omitted_dependency", "omitted installed packages"),
    ("missing_license", "not clear"),
    ("local_path", "private URL"),
])
def test_supply_chain_rejects_rehashed_false_pass(
    tmp_path: Path, change: str, message: str,
) -> None:
    module = _module()
    expected = supply_fixture(tmp_path / "evidence")
    root = tmp_path / "evidence"
    if change in {"hidden_vulnerability", "omitted_dependency"}:
        path = root / "supply-chain-audit.json"
        audit = json.loads(path.read_text())
        if change == "hidden_vulnerability":
            audit["dependencies"][1]["vulns"] = [{"id": "TEST-1"}]
        else:
            audit["dependencies"].pop()
        _write(path, audit)
    elif change == "missing_license":
        for name in ("supply-chain-sbom.json", "supply-chain-licenses.json"):
            path = root / name
            report = json.loads(path.read_text())
            report["components"][1]["licenses"] = []
            _write(path, report)
    else:
        path = root / "supply-chain-sbom.json"
        report = json.loads(path.read_text())
        report["components"][1]["externalReferences"] = [
            {"type": "distribution", "url": "file:///private/customer"}
        ]
        _write(path, report)
    _reseal(root)
    with pytest.raises(module.SupplyChainInvalid, match=message):
        module.verify_supply_chain(root, expected)


def test_supply_chain_rejects_candidate_identity_change(tmp_path: Path) -> None:
    module = _module()
    expected = supply_fixture(tmp_path / "evidence")
    expected["wheel_sha256"] = "0" * 64
    with pytest.raises(module.SupplyChainInvalid, match="wheel_sha256 differs"):
        module.verify_supply_chain(tmp_path / "evidence", expected)


@pytest.mark.parametrize("name, secret", [
    ("supply-chain-requirements.txt", "--index-url https://user:pass@example.com/simple"),
    ("supply-chain-audit.json", "/private/customer/checkpoints"),
    ("supply-chain-installed.json", "C:\\Users\\customer\\secret"),
    ("supply-chain-sbom.json", "https://repo.example/?access_token=secret"),
    ("supply-chain-sbom.json", "https://repo.example/?X-Amz-Signature=test"),
    ("supply-chain-sbom.json", "https://repo.example/?X-Goog-Credential=test"),
    ("supply-chain-sbom.json", "https://repo.example/?sv=1&sp=r&sr=b&sig=test"),
    ("supply-chain-sbom.json", "https://repo.example/?signature=test"),
    ("supply-chain-receipt.json", "file:///private/customer/evidence"),
], ids=[
    "requirements-url", "audit-posix-path", "installed-windows-path",
    "sbom-secret-url", "sbom-aws-signature", "sbom-google-credential",
    "sbom-azure-signature", "sbom-generic-signature", "receipt-local-url",
])
def test_supply_chain_rejects_private_output_after_rehash(
    tmp_path: Path, name: str, secret: str,
) -> None:
    module = _module()
    expected = supply_fixture(tmp_path / "evidence")
    root = tmp_path / "evidence"
    path = root / name
    if path.suffix == ".txt":
        path.write_text(path.read_text() + secret + "\n", encoding="utf-8")
    else:
        report = json.loads(path.read_text())
        if isinstance(report, list):
            report[0]["extra_observation"] = secret
        else:
            report["extra_observation"] = secret
        _write(path, report)
    if name != "supply-chain-receipt.json":
        _reseal(root)
    with pytest.raises(module.SupplyChainInvalid, match="local path|private URL"):
        module.verify_supply_chain(root, expected)


def test_sbom_reference_redaction_and_public_urls(tmp_path: Path) -> None:
    module = _module()
    expected = supply_fixture(tmp_path / "evidence")
    root = tmp_path / "evidence"
    path = root / "supply-chain-sbom.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    public = [
        "https://pypi.org/project/dependency/?source=docs",
        "https://storage.example/blob?sv=1&sp=r&sr=b",
    ]
    private = [
        "https://download.example/blob?X-Amz-Signature=test&X-Amz-Credential=test",
        "https://download.example/blob?X-Goog-Signature=test",
        "https://download.example/blob?sv=1&sp=r&sr=b&sig=test",
        "https://download.example/blob?signature=test",
        "https://user:password@download.example/blob",
        "file:///private/customer/blob",
    ]
    report["components"][1]["externalReferences"] = [
        {"type": "distribution", "url": url} for url in public + private
    ]
    assert module._remove_private_locations(report) == len(private)
    assert [row["url"] for row in report["components"][1]["externalReferences"]] == public
    _write(path, report)
    _reseal(root)
    assert module.verify_supply_chain(root, expected)["third_party_count"] == 1


@pytest.mark.parametrize("name, secret", [
    ("supply-chain-sbom.json", "https://download.example/?X-Amz-Signature=test"),
    ("supply-chain-audit.json", "/private/customer/checkpoints"),
])
def test_blocked_scan_private_output_is_not_published(
    tmp_path: Path, name: str, secret: str,
) -> None:
    module = _module()
    stage = tmp_path / "stage"
    supply_fixture(stage)
    receipt_path = stage / "supply-chain-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "BLOCKED"
    _write(receipt_path, receipt)
    path = stage / name
    report = json.loads(path.read_text(encoding="utf-8"))
    report["private_observation"] = secret
    _write(path, report)
    _reseal(stage)
    published = tmp_path / "published"
    with pytest.raises(module.SupplyChainInvalid, match="local path|private URL"):
        module._publish_scan_output(stage, published)
    assert not published.exists()
    assert all((stage / name).is_file() for name in module.RAW_FILES | {module.RECEIPT})


def test_safe_blocked_scan_keeps_public_failure_evidence(tmp_path: Path) -> None:
    module = _module()
    stage = tmp_path / "stage"
    supply_fixture(stage)
    receipt_path = stage / "supply-chain-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "BLOCKED"
    _write(receipt_path, receipt)
    published = tmp_path / "published"
    module._publish_scan_output(stage, published)
    assert {path.name for path in published.iterdir()} == (
        module.RAW_FILES | {module.RECEIPT, module.PRIVACY_MARKER}
    )
    assert json.loads((published / module.RECEIPT).read_text())["status"] == "BLOCKED"
    with pytest.raises(module.SupplyChainInvalid, match="not clear"):
        module.verify_supply_chain(published, {
            "candidate_version": "0.3.6", "source_sha256": "a" * 64,
        })


def test_scan_exception_leaves_no_published_raw_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    output = tmp_path / "simulation"
    output.mkdir()
    (output / "report.md").write_text("safe failure summary\n", encoding="utf-8")

    def interrupted(_root: Path, _wheel: Path, stage: Path) -> dict:
        (stage / "supply-chain-audit.json").write_text(
            'https://download.example/?X-Goog-Signature=test\n', encoding="utf-8"
        )
        raise RuntimeError("scanner stopped")

    monkeypatch.setattr(module, "_generate_stage", interrupted)
    with pytest.raises(RuntimeError, match="scanner stopped"):
        module.generate(tmp_path, tmp_path / "candidate.whl", output)
    assert [path.name for path in output.iterdir()] == ["report.md"]
    assert not list(tmp_path.glob(".supply-chain-stage-*"))


def test_publish_error_leaves_no_privacy_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    stage = tmp_path / "stage"
    supply_fixture(stage)
    output = tmp_path / "simulation"
    output.mkdir()
    original_replace = module.os.replace
    moves = 0

    def interrupted(source: Path, destination: Path) -> None:
        nonlocal moves
        moves += 1
        if moves == 2:
            raise OSError("publication stopped")
        original_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", interrupted)
    with pytest.raises(OSError, match="publication stopped"):
        module._publish_scan_output(stage, output)
    assert not list(output.iterdir())


@pytest.mark.parametrize("license_bytes, expression, accepted", [
    (b"reviewed license", "MIT", True),
    (b"changed license", "MIT", False),
    (b"reviewed license", "NOASSERTION", False),
])
def test_candidate_wheel_requires_reviewed_license_and_metadata(
    tmp_path: Path, license_bytes: bytes, expression: str, accepted: bool,
) -> None:
    module = _module()
    license_file = tmp_path / "LICENSE"
    license_file.write_bytes(b"reviewed license")
    wheel = tmp_path / "trainguard-0.3.6-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trainguard-0.3.6.dist-info/METADATA", (
            f"Name: trainguard\nVersion: 0.3.6\nLicense-Expression: {expression}\n"
            "License-File: LICENSE\n"
        ))
        archive.writestr("trainguard-0.3.6.dist-info/licenses/LICENSE", license_bytes)
    if accepted:
        assert module.verify_candidate_license(wheel, license_file) == _sha(license_file)
    else:
        with pytest.raises(module.SupplyChainInvalid, match="LICENSE differs|metadata differs"):
            module.verify_candidate_license(wheel, license_file)
