import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
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

    # The critical privacy and retention cases are pinned even when the
    # collected suite and its JUnit report could otherwise shrink together.
    assert {
        ("tests.test_capacity", "test_guarded_completion_rejects_only_one_verified_candidate"),
        ("tests.test_capacity", "test_guarded_audit_counts_uncommitted_candidate_bytes"),
        ("tests.test_privacy", "test_guarded_missing_wrong_key_and_sample_tampering_fail_closed"),
        ("tests.test_privacy", "test_guarded_recovery_matches_uninterrupted_reference"),
    } <= module.REQUIRED_TEST_IDENTITIES

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
                                    "candidate_execution_inputs_sha256":
                                        module.execution_inputs_sha256(root),
                                    "previous_release": previous,
                                    "gates": gates}))
    result = module.evaluate(manifest, wheel, sdist)
    assert result["status"] == "BLOCKED"
    assert len(result["gates"]) == len(module.REQUIRED_GATES)
    gates[0] = {"id": gates[0]["id"], "status": "PASS", "evidence": "README.md",
                "sha256": hashlib.sha256((root / "README.md").read_bytes()).hexdigest()}
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "candidate_execution_inputs_sha256":
                                        module.execution_inputs_sha256(root),
                                    "previous_release": previous,
                                    "gates": gates}))
    with pytest.raises(ValueError, match="evidence receipt"):
        module.evaluate(manifest, wheel, sdist)
    monkeypatch.setattr(module, "_receipt", lambda *args: {})
    assert module.evaluate(manifest, wheel, sdist)["status"] == "BLOCKED"
    gates[0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "candidate_source_sha256": module.package_source_sha256(root),
                                    "candidate_execution_inputs_sha256":
                                        module.execution_inputs_sha256(root),
                                    "previous_release": previous,
                                    "gates": gates}))
    with pytest.raises(ValueError, match="digest differs"):
        module.evaluate(manifest, wheel, sdist)


def test_local_experiment_does_not_grant_linux_customer_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    root = Path(__file__).parents[1]
    original_run = subprocess.run

    def clean_status(command, **kwargs):
        if command == ["git", "status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        return original_run(command, **kwargs)

    monkeypatch.setattr(
        module, "subprocess",
        SimpleNamespace(run=clean_status, TimeoutExpired=subprocess.TimeoutExpired),
    )
    monkeypatch.setattr(module, "_verify_artifacts", lambda *args: None)
    monkeypatch.setattr(module, "_previous_release", lambda *args: {})
    commit = original_run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    monkeypatch.setattr(module, "_receipt", lambda *args: {"execution_commit": commit})
    wheel, sdist = tmp_path / "candidate.whl", tmp_path / "candidate.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"source")
    readme = root / "README.md"
    gates = [
        ({"id": gate_id, "status": "PASS", "evidence": "README.md",
          "sha256": hashlib.sha256(readme.read_bytes()).hexdigest()}
         if gate_id in {"local_package", "local_cpu"}
         else {"id": gate_id, "status": "BLOCKED", "reason": "pending"})
        for gate_id in sorted(module.REQUIRED_GATES)
    ]
    manifest = tmp_path / "gates.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "candidate_source_sha256": module.package_source_sha256(root),
        "candidate_execution_inputs_sha256": module.execution_inputs_sha256(root),
        "previous_release": {}, "gates": gates,
    }), encoding="utf-8")
    local = module.evaluate(manifest, wheel, sdist)
    assert local["local_experiment_allowed"] is True
    assert local["evaluation_allowed"] is False
    assert local["linux_customer_evaluation_allowed"] is False
    assert local["decision_scope"] == "local experiments only"
    assert local["hosted_linux_evidence"] is None

    def hosted(*args, **kwargs):
        assert args[1] == 123
        return {"workflow_run_id": 123, "status": "HOSTED_LINUX_EVIDENCE_CONSISTENT"}

    monkeypatch.setattr(module, "_hosted_linux_from_run", hosted)
    monkeypatch.setattr(module, "_private_reporting_enabled", lambda root: True)
    scoped = module.evaluate(manifest, wheel, sdist, 123)
    assert scoped["evaluation_allowed"] is True
    assert scoped["linux_customer_evaluation_allowed"] is True
    assert scoped["decision_scope"] == "hosted Linux candidate evaluation"

    monkeypatch.setattr(module, "_private_reporting_enabled", lambda root: False)
    disabled = module.evaluate(manifest, wheel, sdist, 123)
    assert disabled["linux_customer_evaluation_allowed"] is False
    assert disabled["private_vulnerability_reporting_enabled"] is False
    assert disabled["linux_customer_evaluation_reason"] == (
        "private vulnerability reporting is disabled"
    )
    monkeypatch.setattr(module, "_private_reporting_enabled", lambda root: True)

    original_input_digest = module.execution_inputs_sha256

    def hosted_with_candidate_drift(*args, **kwargs):
        monkeypatch.setattr(module, "execution_inputs_sha256", lambda root: "0" * 64)
        return hosted(*args, **kwargs)

    monkeypatch.setattr(module, "_hosted_linux_from_run", hosted_with_candidate_drift)
    with pytest.raises(ValueError, match="changed during release evidence review"):
        module.evaluate(manifest, wheel, sdist, 123)
    monkeypatch.setattr(module, "execution_inputs_sha256", original_input_digest)
    monkeypatch.setattr(
        module, "_hosted_linux_from_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("workflow evidence differs")),
    )
    rejected = module.evaluate(manifest, wheel, sdist, 123)
    assert rejected["linux_customer_evaluation_allowed"] is False
    assert rejected["linux_customer_evaluation_reason"] == "workflow evidence differs"


def test_hosted_evidence_fetch_uses_authenticated_pinned_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    commands = []
    commit, source, inputs = "a" * 40, "b" * 64, "c" * 64
    wheel = tmp_path / "candidate.whl"
    sdist = tmp_path / "candidate.tar.gz"
    artifacts = {wheel.name: "d" * 64, sdist.name: "e" * 64, "uv.lock": "f" * 64}

    def fetched(command, **kwargs):
        commands.append(command)
        if command[:3] == ["gh", "run", "download"]:
            Path(command[command.index("--dir") + 1]).mkdir()
        elif command[0] == sys.executable:
            report_path = Path(command[command.index("--report") + 1])
            report_path.write_text(json.dumps({
                "status": "HOSTED_LINUX_EVIDENCE_CONSISTENT",
                "workflow_run_id": 123,
                "candidate_git_commit": commit,
                "candidate_version": "0.3.5",
                "candidate_source_sha256": source,
                "candidate_execution_inputs_sha256": inputs,
                "lock_sha256": artifacts["uv.lock"],
                "artifacts": {wheel.name: artifacts[wheel.name], sdist.name: artifacts[sdist.name]},
                "matrix": {"3.11": {}, "3.12": {}},
                "customer_environment_validated": False,
                "production_release_authorized": False,
            }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="{}")

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=fetched))
    report = module._hosted_linux_from_run(
        Path(__file__).parents[1], 123, wheel, sdist,
        commit=commit, version="0.3.5", source_digest=source,
        input_digest=inputs, artifacts=artifacts,
    )
    assert commands[0] == ["gh", "auth", "status", "--active", "--hostname", "github.com"]
    assert len([command for command in commands if command[:2] == ["gh", "run"]]) == 5
    assert all(
        command[-2:] == ["--repo", module.HOSTED_REPOSITORY]
        for command in commands if command[:2] == ["gh", "run"]
    )
    assert report["workflow_repository"] == module.HOSTED_REPOSITORY
    assert report["cryptographic_signature_verified"] is False

    monkeypatch.setattr(
        module, "subprocess",
        SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(returncode=1)),
    )
    with pytest.raises(ValueError, match="not authenticated"):
        module._hosted_linux_from_run(
            Path(__file__).parents[1], 123, wheel, sdist,
            commit=commit, version="0.3.5", source_digest=source,
            input_digest=inputs, artifacts=artifacts,
        )


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
