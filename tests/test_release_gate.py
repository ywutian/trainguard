import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
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


def _collected_suite(module, *, gpu_host: bool = False):
    identities = sorted(module._collected_test_identities(Path(__file__).parents[1]))
    suite = ElementTree.Element(
        "testsuite", tests=str(len(identities)), failures="0", errors="0",
        skipped="1" if gpu_host else str(len(module.GPU_TESTS)),
    )
    for classname, name in identities:
        case = ElementTree.SubElement(suite, "testcase", classname=classname, name=name)
        if classname == "tests.test_gpu_acceptance" and not gpu_host:
            ElementTree.SubElement(case, "skipped", message="requires two actual CUDA devices")
        elif (classname, name) == module.DEVICE_PREFLIGHT_SKIP and gpu_host:
            ElementTree.SubElement(
                case, "skipped", message="this test verifies the unavailable-device preflight"
            )
    details = {
        "tests": {
            "passed": len(identities) - (1 if gpu_host else len(module.GPU_TESTS)),
            "skipped_no_gpu": 0 if gpu_host else len(module.GPU_TESTS),
            "skipped_device_preflight": 1 if gpu_host else 0,
        }
    }
    return suite, details


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
    previous = json.loads((root / "docs/commercial/release-gates.json").read_text())[
        "previous_release"
    ]
    with pytest.raises(ValueError, match="required raw result files"):
        module._local_evidence(
            root, {"checks": {"full_suite": True, "cpu_fault_matrix": True}},
            module.package_source_sha256(root), {"uv.lock": "0" * 64}, previous,
        )


def test_local_test_gate_rejects_skip_inflation(tmp_path: Path) -> None:
    module = _module()
    suite, details = _collected_suite(module)
    path = tmp_path / "pytest.xml"
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    module._validate_pytest_result(tmp_path, details)

    target = next(case for case in suite.findall("testcase") if (
        case.get("classname"), case.get("name")
    ) == ("tests.test_campaign", "test_cpu_campaign_closes_recovery_matrix"))
    target.set("name", "test_replacement_that_only_checks_true")
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        module._validate_pytest_result(tmp_path, details)

    target.set("name", "test_cpu_campaign_closes_recovery_matrix")
    for case in suite.findall("testcase"):
        if case.find("skipped") is not None:
            continue
        ElementTree.SubElement(case, "skipped")
    suite.set("skipped", suite.get("tests"))
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    with pytest.raises(ValueError, match="over-skipped"):
        module._validate_pytest_result(tmp_path, {
            "tests": {"passed": 0, "skipped_no_gpu": len(module.GPU_TESTS),
                      "skipped_device_preflight": 0}
        })


def test_local_test_gate_rejects_fabricated_count_preserving_suite(tmp_path: Path) -> None:
    module = _module()
    suite = ElementTree.Element("testsuite", tests="160", failures="0", errors="0", skipped="4")
    for number in range(156):
        ElementTree.SubElement(
            suite, "testcase", classname="tests.test_cpu", name=f"test_cpu_{number}"
        )
    for name in sorted(module.GPU_TESTS):
        case = ElementTree.SubElement(
            suite, "testcase", classname="tests.test_gpu_acceptance", name=name
        )
        ElementTree.SubElement(case, "skipped", message="requires two actual CUDA devices")
    ElementTree.ElementTree(suite).write(tmp_path / "pytest.xml", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        module._validate_pytest_result(
            tmp_path, {"tests": {"passed": 156, "skipped_no_gpu": 4,
                                  "skipped_device_preflight": 0}}
        )


def test_local_test_gate_accepts_device_preflight_skip_on_gpu_host(tmp_path: Path) -> None:
    module = _module()
    suite, details = _collected_suite(module, gpu_host=True)
    case = next(case for case in suite.findall("testcase") if (
        case.get("classname"), case.get("name")
    ) == module.DEVICE_PREFLIGHT_SKIP)
    path = tmp_path / "pytest.xml"
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
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
    previous = json.loads((root / "docs/commercial/release-gates.json").read_text())[
        "previous_release"
    ]
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
                                    "previous_release": previous,
                                    "gates": gates}))
    result = module.evaluate(manifest, wheel, sdist)
    assert result["status"] == "BLOCKED"
    assert len(result["gates"]) == len(module.REQUIRED_GATES)
    gates[0] = {"id": gates[0]["id"], "status": "PASS", "evidence": "README.md",
                "sha256": hashlib.sha256((root / "README.md").read_bytes()).hexdigest()}
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "previous_release": previous,
                                    "gates": gates}))
    with pytest.raises(ValueError, match="evidence receipt"):
        module.evaluate(manifest, wheel, sdist)
    monkeypatch.setattr(module, "_receipt", lambda *args: {})
    assert module.evaluate(manifest, wheel, sdist)["status"] == "BLOCKED"
    gates[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "previous_release": previous,
                                    "gates": gates}))
    with pytest.raises(ValueError, match="digest differs"):
        module.evaluate(manifest, wheel, sdist)


def test_upgrade_rehearsal_rejects_a_different_same_version_commit() -> None:
    root = Path(__file__).parents[1]
    pin = json.loads((root / "docs/commercial/release-gates.json").read_text())[
        "previous_release"
    ]
    other = subprocess.run(
        ["git", "rev-parse", f"{pin['git_commit']}^"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    metadata = subprocess.run(
        ["git", "show", f"{other}:pyproject.toml"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    assert f'version = "{pin["version"]}"' in metadata
    command = [
        sys.executable, str(root / "scripts/verify_upgrade_boundary.py"),
        "--previous-ref", other,
        "--expected-previous-commit", pin["git_commit"],
        "--expected-previous-wheel-sha256", pin["wheel_sha256"],
        "--expected-previous-lock-sha256", pin["lock_sha256"],
        "--current-wheel", str(root / "does-not-need-to-exist.whl"),
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "previous ref differs from the approved release commit" in result.stderr


def test_release_gate_rejects_previous_release_lock_tamper() -> None:
    module = _module()
    root = Path(__file__).parents[1]
    pin = json.loads((root / "docs/commercial/release-gates.json").read_text())[
        "previous_release"
    ]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    assert module._previous_release(root, pin, version) == pin
    tampered = pin | {"lock_sha256": "0" * 64}
    with pytest.raises(ValueError, match="does not match its commit"):
        module._previous_release(root, tampered, version)
