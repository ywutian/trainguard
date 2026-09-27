import hashlib
import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "check_release_readiness.py"
    spec = importlib.util.spec_from_file_location("check_release_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_gate_rejects_arbitrary_artifact_bytes(tmp_path: Path) -> None:
    module = _module()
    root = Path(__file__).parents[1]
    wheel, sdist = tmp_path / "candidate.whl", tmp_path / "candidate.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"source")
    with pytest.raises(ValueError, match="archive formats"):
        module._verify_artifacts(root, wheel, sdist, module.package_source_sha256(root))


def test_local_evaluation_rejects_self_asserted_checks_without_raw_results() -> None:
    module = _module()
    root = Path(__file__).parents[1]
    with pytest.raises(ValueError, match="required raw result files"):
        module._local_evidence(
            root, {"checks": {"full_suite": True, "cpu_fault_matrix": True}},
            module.package_source_sha256(root), {"uv.lock": "0" * 64},
        )


def test_local_test_gate_rejects_skip_inflation(tmp_path: Path) -> None:
    module = _module()
    suite = ElementTree.Element(
        "testsuite", tests="160", failures="0", errors="0", skipped="4"
    )
    for number in range(156):
        ElementTree.SubElement(
            suite, "testcase", classname="tests.test_cpu", name=f"test_cpu_{number}"
        )
    for name in sorted(module.GPU_TESTS):
        case = ElementTree.SubElement(
            suite, "testcase", classname="tests.test_gpu_acceptance", name=name
        )
        ElementTree.SubElement(case, "skipped", message="requires two actual CUDA devices")
    path = tmp_path / "pytest.xml"
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    module._validate_pytest_result(
        tmp_path, {"tests": {"passed": 156, "skipped_no_gpu": 4}}
    )

    for case in suite.findall("testcase")[1:156]:
        case.set("name", "test_cpu_0")
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        module._validate_pytest_result(
            tmp_path, {"tests": {"passed": 156, "skipped_no_gpu": 4}}
        )

    for case in suite.findall("testcase")[:156]:
        ElementTree.SubElement(case, "skipped")
    suite.set("skipped", "160")
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="over-skipped"):
        module._validate_pytest_result(
            tmp_path, {"tests": {"passed": 0, "skipped_no_gpu": 160}}
        )


def test_local_test_gate_accepts_device_preflight_skip_on_gpu_host(tmp_path: Path) -> None:
    module = _module()
    suite = ElementTree.Element("testsuite", tests="157", failures="0", errors="0", skipped="1")
    for number in range(156):
        ElementTree.SubElement(
            suite, "testcase", classname="tests.test_cpu", name=f"test_cpu_{number}"
        )
    case = ElementTree.SubElement(
        suite, "testcase", classname=module.DEVICE_PREFLIGHT_SKIP[0],
        name=module.DEVICE_PREFLIGHT_SKIP[1],
    )
    ElementTree.SubElement(
        case, "skipped", message="this test verifies the unavailable-device preflight"
    )
    path = tmp_path / "pytest.xml"
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    details = {"tests": {"passed": 156, "skipped_no_gpu": 0,
                         "skipped_device_preflight": 1}}
    module._validate_pytest_result(tmp_path, details)
    case.find("skipped").set("message", "an unreviewed reason")
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="over-skipped"):
        module._validate_pytest_result(tmp_path, details)


def test_release_gate_blocks_missing_external_evidence_and_detects_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    root = Path(__file__).parents[1]
    monkeypatch.setattr(module, "_verify_artifacts", lambda *args: None)
    wheel, sdist = tmp_path / "candidate.whl", tmp_path / "candidate.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"source")
    gates = [
        {"id": identity, "status": "BLOCKED", "reason": "evidence pending"}
        for identity in sorted(module.REQUIRED_GATES)
    ]
    manifest = tmp_path / "gates.json"
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "gates": gates}))
    result = module.evaluate(manifest, wheel, sdist)
    assert result["status"] == "BLOCKED"
    assert len(result["gates"]) == len(module.REQUIRED_GATES)
    gates[0] = {"id": gates[0]["id"], "status": "PASS", "evidence": "README.md",
                "sha256": hashlib.sha256((root / "README.md").read_bytes()).hexdigest()}
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "gates": gates}))
    with pytest.raises(ValueError, match="evidence receipt"):
        module.evaluate(manifest, wheel, sdist)
    monkeypatch.setattr(module, "_receipt", lambda *args: {})
    assert module.evaluate(manifest, wheel, sdist)["status"] == "BLOCKED"
    gates[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "gates": gates}))
    with pytest.raises(ValueError, match="digest differs"):
        module.evaluate(manifest, wheel, sdist)
