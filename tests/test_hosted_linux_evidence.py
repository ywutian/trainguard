"""Downloaded matrix evidence must agree with one current candidate."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest
from test_supply_chain import supply_fixture


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _acceptance(source: str, python: str, commit: str, platform: str) -> dict:
    cases = [
        {"name": f"{mode}-{fault}", "mode": mode, "fault": fault,
         "omit_state": "none", "expected_exact": True, "status": "PASSED",
         "recovery_count": 1, "fault_attributed": True,
         "validation": {"passed": True, "differences": [],
                        "independent_reference": True,
                        "comparison_kind": "INDEPENDENT_REFERENCE"}}
        for mode, fault in (
            ("sync", "worker_exit"), ("async", "worker_exit"),
            ("sync", "save_interrupt"), ("async", "save_interrupt"),
            ("sync", "corrupt"), ("async", "corrupt"), ("sync", "hang"),
        )
    ]
    cases.extend(
        {"name": f"omit-{state}", "mode": "sync", "fault": "worker_exit",
         "omit_state": state, "expected_exact": False, "status": "PASSED",
         "recovery_count": 1, "fault_attributed": True,
         "validation": {"passed": False, "independent_reference": True,
                        "comparison_kind": "INDEPENDENT_REFERENCE", "differences": [
             "final model_sha256 differs",
             *(
                 [
                     f"rank {rank} {kind} differs"
                     for rank in range(2)
                     for kind in ("effective sample sequence", "consumed batch sequence")
                 ] if state == "cursor" else []
             ),
         ]}}
        for state in ("rng", "optimizer", "cursor")
    )
    return {
        "status": "SUCCEEDED", "reference_status": "VALIDATED", "cases": cases,
        "config": {"run": {"device": "cpu", "backend": "gloo", "world_size": 2}},
        "environment": {"source_sha256": source, "python": python,
                        "git_commit": commit, "platform": platform},
    }


def _junit(path: Path, identities: set[tuple[str, str]]) -> None:
    suite = ElementTree.Element(
        "testsuite", tests=str(len(identities)), failures="0", errors="0", skipped="4"
    )
    for classname, name in sorted(identities):
        case = ElementTree.SubElement(suite, "testcase", classname=classname, name=name)
        if classname == "tests.test_gpu_acceptance":
            ElementTree.SubElement(case, "skipped", message="requires two actual CUDA devices")
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    verifier = importlib.import_module("check_hosted_linux_evidence")
    release = importlib.import_module("check_release_readiness")
    version, commit, source = "0.3.5", "a" * 40, "b" * 64
    inputs = "f" * 64
    platform = "Linux-6.8.0-x86_64-with-glibc2.39"
    previous = {
        "git_commit": "c" * 40, "version": "0.3.4",
        "wheel_sha256": "d" * 64, "lock_sha256": "e" * 64,
    }
    wheel = tmp_path / f"trainguard-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("trainguard/__init__.py", '__version__ = "0.3.5"\n')
        prefix = f"trainguard-{version}.dist-info/"
        archive.writestr(prefix + "METADATA", (
            f"Name: trainguard\nVersion: {version}\nLicense-Expression: MIT\n"
            "License-File: LICENSE\n"
        ))
        archive.write(root / "LICENSE", prefix + "licenses/LICENSE")
    sdist = tmp_path / f"trainguard-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        payload = b"reviewed source"
        member = tarfile.TarInfo(f"trainguard-{version}/README.md")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    artifacts = {
        wheel.name: hashlib.sha256(wheel.read_bytes()).hexdigest(),
        sdist.name: hashlib.sha256(sdist.read_bytes()).hexdigest(),
    }
    lock_sha = hashlib.sha256(b"locked dependencies").hexdigest()
    metadata = {
        "databaseId": 12345, "headSha": commit, "conclusion": "success",
        "event": "push", "workflowName": "Verify recovery package",
        "jobs": [
            {"databaseId": 1100 + int(lane[-2:]), "name": f"cpu-and-package ({lane})",
             "status": "completed", "conclusion": "success"}
            for lane in verifier.PYTHON_LANES
        ],
    }
    identities = release._collected_test_identities(root)
    summary, packages = {}, {}
    for lane in verifier.PYTHON_LANES:
        py = f"{lane}.12"
        run = tmp_path / f"summary-{lane}" / "simulation-run"
        run.mkdir(parents=True)
        _junit(run / "pytest.xml", identities)
        pack = tmp_path / f"packages-{lane}" / "simulation-run" / "dist"
        pack.mkdir(parents=True)
        shutil.copy2(wheel, pack / wheel.name)
        shutil.copy2(sdist, pack / sdist.name)
        gates = [
            {"name": name, "exit_code": 0, "timed_out": False,
             "output": f"/runner/verification/simulation-run/{name}.txt",
             "execution_inputs_before_sha256": inputs,
             "execution_inputs_after_sha256": inputs}
            for name in sorted(verifier.REQUIRED_GATES)
        ]
        for gate in gates:
            (run / f"{gate['name']}.txt").write_text("completed\n", encoding="utf-8")
        supply_expected = supply_fixture(run)
        for name in ("supply-chain-installed.json", "supply-chain-sbom.json",
                     "supply-chain-licenses.json", "supply-chain-audit.json"):
            path = run / name
            report = json.loads(path.read_text())
            rows = report if isinstance(report, list) else report.get(
                "components", report.get("dependencies", [])
            )
            for row in rows:
                if row.get("name") == "trainguard":
                    row["version"] = version
                    if "bom-ref" in row:
                        row["bom-ref"] = f"pkg:pypi/trainguard@{version}"
            if name == "supply-chain-sbom.json":
                report["dependencies"][0]["ref"] = f"pkg:pypi/trainguard@{version}"
            _write_json(path, report)
        supply_expected.update({
            "candidate_version": version,
            "source_sha256": source,
            "execution_inputs_sha256": inputs,
            "wheel_sha256": artifacts[wheel.name],
            "lock_sha256": lock_sha,
            "tool_lock_sha256": hashlib.sha256(
                (root / "scripts/supply-chain-tools.txt").read_bytes()
            ).hexdigest(),
            "first_party_license_sha256": hashlib.sha256(
                (root / "LICENSE").read_bytes()
            ).hexdigest(),
            "security_policy_sha256": hashlib.sha256(
                (root / "SECURITY.md").read_bytes()
            ).hexdigest(),
            "security_channel_record_sha256": hashlib.sha256(
                (root / "docs/commercial/security-channel-2026-09-27.json").read_bytes()
            ).hexdigest(),
        })
        supply_receipt_path = run / "supply-chain-receipt.json"
        supply_receipt = json.loads(supply_receipt_path.read_text())
        supply_receipt.update(supply_expected)
        supply_receipt.update(python_version=py, platform="Linux", machine="x86_64")
        supply_receipt["files"] = {
            name: hashlib.sha256((run / name).read_bytes()).hexdigest()
            for name in supply_receipt["files"]
        }
        _write_json(supply_receipt_path, supply_receipt)
        _write_json(run / "supply-chain.txt", {
            "status": "PASS", "wheel_sha256": artifacts[wheel.name],
            "component_count": 2, "known_vulnerability_count": 0,
            "unscanned_third_party": [], "unlicensed_third_party": [],
        })
        (run / "wheel.txt").write_text(json.dumps({
            "passed": True, "version": version, "source_sha256": source,
            "sdist_members": 90,
        }) + "\n", encoding="utf-8")
        (run / "fresh-install.txt").write_text(json.dumps({
            "version": version, "wheel_sha256": artifacts[wheel.name],
            "installed_outside_checkout": True, "completed_run": True,
            "recovered_run_matches_reference": True, "support_export_checked": True,
            "run_data_preserved_after_uninstall": True, "attributed_faults": 1,
            "recoveries": 1, "run_data_files_checked": 12,
        }) + "\n", encoding="utf-8")
        (run / "upgrade-boundary.txt").write_text(json.dumps({
            "current_version": version, "current_wheel_sha256": artifacts[wheel.name],
            "current_lock_sha256": lock_sha, "previous_commit_sha": previous["git_commit"],
            "previous_version": previous["version"],
            "previous_wheel_sha256": previous["wheel_sha256"],
            "previous_lock_sha256": previous["lock_sha256"],
            "new_version_rejected_interrupted_old_run": True,
            "old_locked_environment_resumed_exactly": True,
            "old_run_files_unchanged_after_rejection": 12,
        }) + "\n", encoding="utf-8")
        _write_json(run / "result.json", {
            "status": "SUCCEEDED", "version": version, "source_sha256": source,
            "execution_commit": commit, "execution_commit_after": commit,
            "execution_inputs_sha256": inputs, "execution_inputs_after_sha256": inputs,
            "lock_sha256": lock_sha, "artifact_sha256": artifacts,
            "python": py, "platform": platform, "gates": gates,
            "acceptance": _acceptance(source, py, commit, platform),
        })
        summary[lane] = run.parent
        packages[lane] = pack.parent.parent
    return verifier, metadata, summary, packages, wheel, sdist, {
        "commit": commit, "version": version, "source_sha256": source,
        "execution_inputs_sha256": inputs, "lock_sha256": lock_sha,
        "previous_release": previous,
    }


def _verify(evidence):
    verifier, metadata, summary, packages, wheel, sdist, candidate = evidence
    return verifier.verify_evidence(
        metadata, summary, packages, wheel, sdist, **candidate
    )


def test_hosted_linux_evidence_requires_both_exact_candidate_lanes(evidence) -> None:
    result = _verify(evidence)
    assert result["status"] == "HOSTED_LINUX_EVIDENCE_CONSISTENT"
    assert set(result["matrix"]) == {"3.11", "3.12"}
    assert result["workflow_git_commit"] == result["candidate_git_commit"]
    assert result["customer_environment_validated"] is False
    assert result["production_release_authorized"] is False


def test_hosted_linux_evidence_rejects_stale_commit(evidence) -> None:
    evidence[1]["headSha"] = "f" * 40
    with pytest.raises(ValueError, match="current-commit"):
        _verify(evidence)


def test_hosted_linux_evidence_requires_push_checkout_identity(evidence) -> None:
    evidence[1]["event"] = "pull_request"
    with pytest.raises(ValueError, match="current-commit"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_missing_python_lane(evidence) -> None:
    evidence[1]["jobs"].pop()
    with pytest.raises(ValueError, match="not both successful"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_failed_python_job(evidence) -> None:
    evidence[1]["jobs"][0]["conclusion"] = "failure"
    with pytest.raises(ValueError, match="not both successful"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_green_job_with_failed_raw_result(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["status"] = "FAILED"
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="incomplete or mismatched gates"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_missing_critical_junit_case(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    path = run / "pytest.xml"
    suite = ElementTree.parse(path).getroot()
    target = next(case for case in suite.findall("testcase") if (
        case.get("classname"), case.get("name")
    ) == ("tests.test_campaign", "test_cpu_campaign_closes_recovery_matrix"))
    target.set("name", "test_unrelated_replacement")
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        _verify(evidence)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_hosted_linux_evidence_rejects_package_byte_drift(evidence, kind: str) -> None:
    local = evidence[4] if kind == "wheel" else evidence[5]
    hosted = evidence[3]["3.11"] / "simulation-run" / "dist" / local.name
    hosted.write_bytes(hosted.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="package bytes differ"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_missing_cpu_negative_control(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["acceptance"]["cases"] = [
        case for case in result["acceptance"]["cases"] if case["name"] != "omit-rng"
    ]
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="CPU fault matrix is incomplete"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_missing_cpu_negative_difference(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    case = next(case for case in result["acceptance"]["cases"]
                if case["name"] == "omit-cursor")
    case["validation"]["differences"] = ["final model_sha256 differs"]
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="CPU fault matrix is incomplete"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_cpu_self_comparison(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    case = next(case for case in result["acceptance"]["cases"]
                if case["name"] == "sync-worker_exit")
    case["validation"]["independent_reference"] = False
    case["validation"]["comparison_kind"] = "SELF_CHECK"
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="CPU fault matrix is incomplete"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_acceptance_from_another_commit(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["acceptance"]["environment"]["git_commit"] = "f" * 40
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="acceptance used another environment"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_result_from_another_commit(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["execution_commit_after"] = "0" * 40
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="incomplete or mismatched gates"):
        _verify(evidence)


@pytest.mark.parametrize("platform", ["Linux-6.8.0-aarch64", "notLinux-6.8.0-x86_64"])
def test_hosted_linux_evidence_rejects_wrong_runtime_architecture(
    evidence, platform: str
) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["platform"] = platform
    result["acceptance"]["environment"]["platform"] = platform
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="incomplete or mismatched gates"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_acceptance_platform_drift(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["acceptance"]["environment"]["platform"] = "Linux-6.8.0-aarch64"
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="acceptance used another environment"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_stale_execution_input_digest(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    result = json.loads((run / "result.json").read_text())
    result["gates"][0]["execution_inputs_after_sha256"] = "0" * 64
    _write_json(run / "result.json", result)
    with pytest.raises(ValueError, match="incomplete or mismatched gates"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_previous_release_pin_drift(evidence) -> None:
    evidence[6]["previous_release"]["lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="package or upgrade evidence is incomplete"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_missing_supply_chain_report(evidence) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    (run / "supply-chain-sbom.json").unlink()
    with pytest.raises(ValueError, match="supply-chain evidence differs"):
        _verify(evidence)


def test_hosted_linux_evidence_rejects_changed_supply_chain_scan(evidence) -> None:
    run = evidence[2]["3.12"] / "simulation-run"
    path = run / "supply-chain-audit.json"
    report = json.loads(path.read_text())
    report["dependencies"][1]["vulns"] = [{"id": "TEST-1"}]
    _write_json(path, report)
    with pytest.raises(ValueError, match="supply-chain evidence differs"):
        _verify(evidence)


@pytest.mark.parametrize("field", ["wheel_sha256", "lock_sha256"])
def test_hosted_linux_evidence_rejects_unbound_supply_chain_receipt(
    evidence, field: str,
) -> None:
    run = evidence[2]["3.11"] / "simulation-run"
    path = run / "supply-chain-receipt.json"
    receipt = json.loads(path.read_text())
    receipt[field] = "0" * 64
    _write_json(path, receipt)
    with pytest.raises(ValueError, match=f"receipt {field} differs"):
        _verify(evidence)
