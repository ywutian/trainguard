"""Evidence archiving rejects stale runs and removes test-host private text."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def archive_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    sys.modules.pop("archive_local_evidence", None)
    import archive_local_evidence as archiver

    root = tmp_path / "checkout"
    evidence = root / "docs/commercial/evidence"
    evidence.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nversion = "0.3.6"\n')
    (root / "uv.lock").write_bytes(b"fixed-lock")
    for name in ("SECURITY.md", "LICENSE", "scripts/supply-chain-tools.txt",
                 "docs/commercial/security-channel-2026-09-27.json"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("fixed-policy")
    (evidence.parent / "release-gates.json").write_text(
        json.dumps({"previous_release": {"git_commit": "b" * 40}})
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "pyproject.toml", "uv.lock"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Evidence Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "Prepare fixture"], cwd=root, check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source = root / "runs/simulation-123456789abc"
    source.mkdir(parents=True)
    (source / "dist").mkdir()
    wheel = source / "dist/trainguard-0.3.6-py3-none-any.whl"
    sdist = source / "dist/trainguard-0.3.6.tar.gz"
    wheel.write_bytes(b"wheel-fixture")
    sdist.write_bytes(b"source-fixture")
    digest = lambda data: hashlib.sha256(data).hexdigest()
    artifacts = {wheel.name: digest(wheel.read_bytes()), sdist.name: digest(sdist.read_bytes())}
    names = sorted(archiver.EXPECTED_CASES)
    acceptance = {
        "status": "SUCCEEDED", "reference_status": "VALIDATED",
        "environment": {"git_commit": commit, "host_path": "/Users/private/secret.key"},
        "cases": [
            {
                "name": name, "mode": "sync", "fault": "worker_exit",
                "omit_state": "none", "expected_exact": not name.startswith("omit-"),
                "status": "PASSED", "recovery_count": 1, "fault_attributed": True,
                "run_dir": "/Users/private/test-artifacts/customer-sample.key",
                "reason": "TOPSECRET-KEY-MATERIAL",
                "metrics": {"recovery_rto_seconds": 1.5, "host_path": "/Users/private"},
                "validation": {"passed": not name.startswith("omit-"),
                               "differences": ["TOPSECRET-KEY-MATERIAL"]},
            }
            for name in names
        ],
    }
    original_acceptance = source / "acceptance/acceptance-123/acceptance.json"
    original_acceptance.parent.mkdir(parents=True)
    original_acceptance.write_text(json.dumps(acceptance))
    gates = [
        {
            "name": name, "command": ["uv", name, "/Users/private/test-artifacts"],
            "output": str(source / f"{name}.txt"), "exit_code": 0, "timed_out": False,
            "execution_inputs_before_sha256": "2" * 64,
            "execution_inputs_after_sha256": "2" * 64,
        }
        for name in sorted(archiver.SUPPORTED_GATE_NAMES | archiver.OPTIONAL_GATE_NAMES)
    ]
    result = {
        "schema_version": 1, "status": "SUCCEEDED", "started_at": "2026-09-27T00:00:00+00:00",
        "finished_at": "2026-09-27T00:01:00+00:00", "version": "0.3.6",
        "source_sha256": "1" * 64, "execution_inputs_sha256": "2" * 64,
        "execution_inputs_after_sha256": "2" * 64,
        "execution_commit": commit, "execution_commit_after": commit,
        "python": "3.12.12", "torch": "2.14.0", "gates": gates,
        "artifact_sha256": artifacts,
        "lock_sha256": digest((root / "uv.lock").read_bytes()),
        "acceptance_path": str(original_acceptance), "acceptance": acceptance,
    }
    (source / "result.json").write_text(json.dumps(result))
    (source / "pytest.xml").write_text(
        '<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" '
        'tests="1" time="0.01" timestamp="2026-09-27" hostname="private-host">'
        '<testcase classname="tests.test_sample" name="test_one" time="0.01">'
        '<system-out>TOPSECRET-KEY-MATERIAL /Users/private</system-out>'
        '</testcase></testsuite></testsuites>'
    )
    (source / "tests.txt").write_text(
        "Trace at /Users/private/test-artifacts/customer-sample.key: "
        "TOPSECRET-KEY-MATERIAL\n1 passed, 0 skipped in 0.01s\n"
    )
    (source / "static.txt").write_text("All checks passed!\n")
    (source / "cpu-acceptance.txt").write_text("Acceptance report: /Users/private/secret\n")
    (source / "package.txt").write_text("Built packages\n")
    (source / "wheel.txt").write_text(json.dumps({
        "passed": True, "version": "0.3.6", "source_sha256": "1" * 64,
        "private_path": "/Users/private/secret.key",
    }) + "\n")
    (source / "fresh-install.txt").write_text(json.dumps({
        "version": "0.3.6", "wheel_sha256": artifacts[wheel.name],
        "installed_outside_checkout": True, "completed_run": True,
        "recovered_run_matches_reference": True, "support_export_checked": True,
        "run_data_preserved_after_uninstall": True, "run_data_files_checked": 1,
        "attributed_faults": 1, "recoveries": 1,
    }) + "\n")
    (source / "upgrade-boundary.txt").write_text(json.dumps({
        "current_version": "0.3.6", "current_wheel_sha256": artifacts[wheel.name],
        "current_lock_sha256": result["lock_sha256"], "previous_commit_sha": "b" * 40,
        "previous_version": "0.3.5", "previous_wheel_sha256": "c" * 64,
        "previous_lock_sha256": "d" * 64,
        "new_version_rejected_interrupted_old_run": True,
        "old_locked_environment_resumed_exactly": True,
        "old_run_files_unchanged_after_rejection": 1,
    }) + "\n")
    from supply_chain import AUDIT_TOOL_VERSION, RAW_FILES, SBOM_TOOL_VERSION

    installed = [{"name": "trainguard", "version": "0.3.6"},
                 {"name": "example", "version": "1.0"}]
    refs = {row["name"]: f"pkg:pypi/{row['name']}@{row['version']}" for row in installed}
    component_licenses = [{"license": {"id": "MIT"}}]
    reports = {
        "supply-chain-installed.json": installed,
        "supply-chain-sbom.json": {
            "bomFormat": "CycloneDX", "specVersion": "1.6",
            "components": [{**row, "bom-ref": refs[row["name"]],
                            "licenses": component_licenses} for row in installed],
            "dependencies": [
                {"ref": refs["trainguard"], "dependsOn": [refs["example"]]},
                {"ref": refs["example"], "dependsOn": []},
            ],
        },
        "supply-chain-licenses.json": {
            "schema_version": 1, "source": "CycloneDX declared package metadata",
            "components": [{**row, "licenses": component_licenses} for row in installed],
        },
        "supply-chain-audit.json": {
            "dependencies": [
                {"name": "trainguard", "skip_reason": "first-party package"},
                {"name": "example", "version": "1.0", "vulns": []},
            ], "fixes": [],
        },
    }
    for name, report in reports.items():
        (source / name).write_text(json.dumps(report))
    (source / "supply-chain-requirements.txt").write_text(
        "example==1.0 --hash=sha256:abc\n"
    )
    chain_expected = {
        "source_sha256": "1" * 64, "execution_inputs_sha256": "2" * 64,
        "wheel_sha256": artifacts[wheel.name], "lock_sha256": result["lock_sha256"],
        "tool_lock_sha256": digest((root / "scripts/supply-chain-tools.txt").read_bytes()),
        "security_policy_sha256": digest((root / "SECURITY.md").read_bytes()),
        "first_party_license_sha256": digest((root / "LICENSE").read_bytes()),
        "security_channel_record_sha256": digest(
            (root / "docs/commercial/security-channel-2026-09-27.json").read_bytes()
        ),
    }
    receipt = {
        "schema_version": 1, "status": "PASS", "generated_at_utc": "2026-09-27T00:00:00+00:00",
        **chain_expected, "python_version": "3.12.12", "platform": "Darwin",
        "machine": "arm64", "audit_service": "pypi", "audit_exit_code": 0,
        "audit_tool_version": AUDIT_TOOL_VERSION, "uv_version": "uv 0.9.1",
        "sbom_tool_version": SBOM_TOOL_VERSION,
        "sbom_private_references_removed": 0, "database_snapshot_available": False,
        "candidate_version": "0.3.6", "component_count": 2,
        "third_party_count": 1, "known_vulnerability_count": 0,
        "unscanned_third_party": [], "unlicensed_third_party": [],
        "files": {name: digest((source / name).read_bytes()) for name in RAW_FILES},
    }
    (source / "supply-chain-receipt.json").write_text(json.dumps(receipt))
    (source / "supply-chain.txt").write_text(json.dumps({
        key: receipt[key] for key in (
            "status", "component_count", "third_party_count", "known_vulnerability_count",
            "unscanned_third_party", "unlicensed_third_party", "wheel_sha256",
        )
    }) + "\n")
    monkeypatch.setattr(archiver, "package_source_sha256", lambda current: "1" * 64)
    monkeypatch.setattr(archiver, "execution_inputs_sha256", lambda current: "2" * 64)
    monkeypatch.setattr(archiver, "require_evidence_only_descendant", lambda *args: None)
    monkeypatch.setattr(archiver, "_local_evidence", lambda *args: None)
    monkeypatch.setattr(archiver, "_receipt", lambda *args: None)
    return archiver, root, source


def test_archive_keeps_original_hashes_and_removes_private_test_text(archive_fixture) -> None:
    archiver, root, source = archive_fixture
    original_hash = hashlib.sha256((source / "result.json").read_bytes()).hexdigest()
    output = archiver.archive(source, "0.3.6-r1", root)
    record = json.loads((output / "local-validation.json").read_text())
    assert record["unredacted_source_sha256"]["result.json"] == original_hash
    assert record["full_original_result_reference"] == "simulation:simulation-123456789abc"
    assert record["human_approval"] is False
    for name in ("local_package", "local_cpu"):
        receipt = json.loads((output / f"{name}.json").read_text())
        assert receipt["reviewer_role"] == "automated_local_evidence_verification"
        assert receipt["human_approval"] is False
        assert receipt["record_sha256"] == hashlib.sha256(
            (output / "local-validation.json").read_bytes()
        ).hexdigest()
    contents = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert b"TOPSECRET-KEY-MATERIAL" not in contents
    assert b"/Users/private" not in contents
    assert b"customer-sample.key" not in contents
    assert b"private-host" not in contents
    from supply_chain import verify_supply_chain

    chain_record = json.loads((output / "raw/supply-chain-receipt.json").read_text())
    verify_supply_chain(output / "raw", {
        field: chain_record[field] for field in (
            "candidate_version", "source_sha256", "execution_inputs_sha256",
            "wheel_sha256", "lock_sha256", "tool_lock_sha256",
            "security_policy_sha256", "first_party_license_sha256",
            "security_channel_record_sha256",
        )
    })


@pytest.mark.parametrize("damage", ["source", "inputs", "missing", "acceptance", "failed"])
def test_archive_rejects_stale_incomplete_or_unsuccessful_runs(
    archive_fixture, damage: str
) -> None:
    archiver, root, source = archive_fixture
    result_path = source / "result.json"
    result = json.loads(result_path.read_text())
    if damage == "source":
        result["source_sha256"] = "f" * 64
    elif damage == "inputs":
        result["execution_inputs_sha256"] = "f" * 64
    elif damage == "missing":
        (source / "wheel.txt").unlink()
    elif damage == "acceptance":
        Path(result["acceptance_path"]).unlink()
    else:
        result["status"] = "FAILED"
    result_path.write_text(json.dumps(result))
    with pytest.raises((ValueError, TypeError), match="source|input|missing|status"):
        archiver.archive(source, "0.3.6-r1", root)
    assert not (root / "docs/commercial/evidence/local-0.3.6-r1").exists()


def test_archive_rejects_same_simulation_under_another_label(archive_fixture) -> None:
    archiver, root, source = archive_fixture
    archiver.archive(source, "0.3.6-r1", root)
    with pytest.raises(ValueError, match="already been archived"):
        archiver.archive(source, "0.3.6-r2", root)


def test_archive_rejects_source_mutation_during_verification(
    archive_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, root, source = archive_fixture
    monkeypatch.setattr(
        archiver, "_local_evidence",
        lambda *args: (source / "static.txt").write_text("changed mid-archive"),
    )
    with pytest.raises(ValueError, match="changed while archiving"):
        archiver.archive(source, "0.3.6-r1", root)
    assert not (root / "docs/commercial/evidence/local-0.3.6-r1").exists()


def test_supply_chain_projection_preserves_graph_without_private_references(
    archive_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    archiver, root, source = archive_fixture
    for name in ("SECURITY.md", "LICENSE", "scripts/supply-chain-tools.txt",
                 "docs/commercial/security-channel-2026-09-27.json"):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("fixed-policy")
    original_ref = "pkg:pypi/example@1.0"
    components = [{
        "name": "example", "version": "1.0", "bom-ref": original_ref,
        "licenses": [{"license": {"id": "MIT", "url": "https://example.invalid/license"}}],
        "externalReferences": [{"url": "https://example.invalid/source"}],
    }]
    reports = {
        "supply-chain-sbom.json": {
            "bomFormat": "CycloneDX", "specVersion": "1.6", "components": components,
            "dependencies": [{"ref": original_ref, "dependsOn": []}],
        },
        "supply-chain-licenses.json": {
            "schema_version": 1, "source": "CycloneDX declared package metadata",
            "components": [{"name": "example", "version": "1.0",
                            "licenses": components[0]["licenses"]}],
        },
        "supply-chain-audit.json": {
            "dependencies": [{"name": "example", "version": "1.0", "vulns": []}],
            "fixes": [],
        },
        "supply-chain-installed.json": [{"name": "example", "version": "1.0",
                                         "location": "/Users/private/key"}],
        "supply-chain-receipt.json": {
            "schema_version": 1, "status": "PASS", "generated_at_utc": "2026-09-27T00:00:00+00:00",
            **{key: "a" * 64 for key in (
                "source_sha256", "execution_inputs_sha256", "wheel_sha256",
                "lock_sha256", "tool_lock_sha256", "security_policy_sha256",
                "first_party_license_sha256", "security_channel_record_sha256",
            )},
            "python_version": "3.12.12", "platform": "Darwin", "machine": "arm64",
            "audit_service": "pypi", "audit_exit_code": 0, "audit_tool_version": "2.10.1",
            "uv_version": "uv 0.9.1", "sbom_tool_version": "7.4.0",
            "sbom_private_references_removed": 0, "database_snapshot_available": False,
            "candidate_version": "0.3.6", "component_count": 1, "third_party_count": 1,
            "known_vulnerability_count": 0, "unscanned_third_party": [],
            "unlicensed_third_party": [],
        },
    }
    for name, value in reports.items():
        (source / name).write_text(json.dumps(value))
    (source / "supply-chain-requirements.txt").write_text(
        "example==1.0 --hash=sha256:abc /Users/private/key\n"
    )
    (source / "supply-chain.txt").write_text(json.dumps({
        "status": "PASS", "component_count": 1, "third_party_count": 1,
        "known_vulnerability_count": 0, "unscanned_third_party": [],
        "unlicensed_third_party": [], "wheel_sha256": "a" * 64,
    }))
    chain_files = set(reports) | {"supply-chain-requirements.txt"}
    monkeypatch.setattr(archiver, "LOCAL_RAW_FILES", chain_files | {"supply-chain.txt"})
    verified = []
    monkeypatch.setitem(sys.modules, "supply_chain", SimpleNamespace(
        RAW_FILES=chain_files - {"supply-chain-receipt.json"},
        RECEIPT="supply-chain-receipt.json",
        verify_supply_chain=lambda *args: verified.append(args),
    ))
    original_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in chain_files | {"supply-chain.txt"}
    }
    output = archiver._safe_supply_chain(
        source, root, "1" * 64, "2" * 64,
        {"trainguard-0.3.6-py3-none-any.whl": "a" * 64, "uv.lock": "b" * 64},
        original_hashes,
    )
    assert len(verified) == 1
    assert set(output) == chain_files | {"supply-chain.txt"}
    all_bytes = b"".join(output.values())
    assert b"/Users/private" not in all_bytes
    assert b"externalReferences" not in all_bytes
    assert original_ref.encode() not in all_bytes
    projected = json.loads(output["supply-chain-sbom.json"])
    assert projected["dependencies"][0]["ref"] == projected["components"][0]["bom-ref"]
