"""Bound local release gates and retain a machine-readable failure record."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree


def _closure_module():
    source = Path(__file__).parents[1] / "scripts/run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failed_report(tmp_path: Path, module) -> str:
    module._persist(tmp_path, {
        "status": "FAILED", "version": "0.3.6", "source_sha256": "0" * 64,
        "gates": [
            {"name": "static", "exit_code": 0, "output": str(tmp_path / "static.txt")},
            {"name": "tests", "exit_code": 1, "output": str(tmp_path / "tests.txt")},
        ],
    })
    return (tmp_path / "report.md").read_text(encoding="utf-8")


def test_early_failure_report_only_shows_source_defined_test_identity(tmp_path: Path) -> None:
    module = _closure_module()
    root = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(root, "testsuite")
    case = ElementTree.SubElement(suite, "testcase", {
        "classname": "tests.test_simulation_gate",
        "name": (
            "test_success_status_without_complete_cpu_matrix_is_rejected"
            "[https://private.example/?accessToken=customer-secret]"
        ),
    })
    ElementTree.SubElement(case, "failure", {
        "message": "private assertion /Users/customer/input.json",
    }).text = "sample value: customer-secret"
    ElementTree.ElementTree(root).write(tmp_path / "pytest.xml", encoding="utf-8")
    report = _failed_report(tmp_path, module)
    assert (
        "test_simulation_gate.py::"
        "test_success_status_without_complete_cpu_matrix_is_rejected"
    ) in report
    for private in ("accessToken", "customer-secret", "private.example", "/Users/", "sample value"):
        assert private not in report


def test_early_failure_report_rejects_unsafe_or_unparseable_xml(tmp_path: Path) -> None:
    module = _closure_module()
    path = tmp_path / "pytest.xml"
    path.write_text("<testsuites><testsuite><testcase", encoding="utf-8")
    report = _failed_report(tmp_path, module)
    assert "测试结果 XML 不可解析" in report
    assert "<testcase" not in report

    root = ElementTree.Element("testsuites")
    suite = ElementTree.SubElement(root, "testsuite")
    case = ElementTree.SubElement(suite, "testcase", {
        "classname": "tests.test_simulation_gate.private/customer",
        "name": "test_success_status_without_complete_cpu_matrix_is_rejected",
    })
    ElementTree.SubElement(case, "failure").text = "customer-secret"
    ElementTree.ElementTree(root).write(path, encoding="utf-8")
    report = _failed_report(tmp_path, module)
    assert "测试结果 XML 不可解析" in report
    assert "customer-secret" not in report
    assert "test_success_status_without_complete_cpu_matrix_is_rejected" not in report


def test_early_failure_report_rejects_oversized_or_entity_xml(tmp_path: Path) -> None:
    module = _closure_module()
    path = tmp_path / "pytest.xml"
    path.write_bytes(b"x" * (module.MAX_JUNIT_SUMMARY_BYTES + 1))
    assert "测试结果 XML 不可解析" in _failed_report(tmp_path, module)
    path.write_text(
        '<!DOCTYPE testsuites [<!ENTITY private "customer-secret">]>'
        '<testsuites><testsuite><testcase classname="tests.test_simulation_gate" '
        'name="test_success_status_without_complete_cpu_matrix_is_rejected">'
        '<failure>&private;</failure></testcase></testsuite></testsuites>',
        encoding="utf-8",
    )
    report = _failed_report(tmp_path, module)
    assert "测试结果 XML 不可解析" in report
    assert "customer-secret" not in report
    assert "test_success_status_without_complete_cpu_matrix_is_rejected" not in report


def test_real_pytest_xml_failure_extracts_only_source_identity(tmp_path: Path) -> None:
    module = _closure_module()
    fixture = tmp_path / "fixture"
    tests_root = fixture / "tests"
    tests_root.mkdir(parents=True)
    (tests_root / "__init__.py").write_text("", encoding="utf-8")
    (tests_root / "test_probe.py").write_text(
        'def test_failure():\n    assert False, "customer-secret /Users/private/input"\n',
        encoding="utf-8",
    )
    xml = tmp_path / "pytest.xml"
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--rootdir", str(fixture),
         "-o", "addopts=",
         f"--junitxml={xml}", str(tests_root / "test_probe.py")],
        cwd=fixture, env=environment, capture_output=True, text=True,
        check=False, timeout=30,
    )
    assert completed.returncode == 1
    assert b"customer-secret" in xml.read_bytes()
    assert module._failed_test_identities(xml, tests_root=tests_root) == (
        ("test_probe.py::test_failure", "失败"),
    )


def test_safe_failure_identity_requires_source_defined_class_and_function(tmp_path: Path) -> None:
    module = _closure_module()
    (tmp_path / "test_local.py").write_text(
        "class TestRecovery:\n    def test_resume(self):\n        pass\n",
        encoding="utf-8",
    )
    case = ElementTree.Element("testcase", {
        "classname": "tests.test_local.TestRecovery",
        "name": "test_resume[private parameter]",
    })
    assert module._safe_failed_test_identity(case, tmp_path) == (
        "test_local.py::TestRecovery::test_resume"
    )
    case.set("name", "test_unknown[private parameter]")
    try:
        module._safe_failed_test_identity(case, tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown test function was accepted")


def test_timeout_report_uses_durable_source_identity_without_junit(
    tmp_path: Path, monkeypatch,
) -> None:
    module = _closure_module()
    fixture = tmp_path / "fixture"
    tests_root = fixture / "tests"
    tests_root.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tests_root / "conftest.py")
    (tests_root / "test_probe.py").write_text(
        "import time\nimport pytest\n"
        "@pytest.mark.parametrize('payload', ['https://private.example/?accessToken=secret'])\n"
        "def test_hangs(payload):\n    time.sleep(60)\n",
        encoding="utf-8",
    )
    progress = tmp_path / "pytest-progress.json"
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    module.GATE_TIMEOUT_SECONDS["tests"] = 5
    gate = module._run(
        tmp_path, "tests",
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=",
         f"--safe-progress-file={progress}", str(tests_root / "test_probe.py")],
    )
    assert gate["exit_code"] == 124 and gate["timed_out"] is True
    assert not (tmp_path / "pytest.xml").exists()
    assert module._last_started_test(progress, tests_root=tests_root) == (
        "test_probe.py::test_hangs"
    )
    module._persist(tmp_path, {
        "status": "FAILED", "version": "0.3.6", "source_sha256": "0" * 64,
        "gates": [gate],
    }, tests_root=tests_root)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "test_probe.py::test_hangs" in report
    for private in ("accessToken", "private.example", "secret", str(fixture), "time.sleep"):
        assert private not in report


def test_timeout_report_rejects_forged_or_unsafe_progress(tmp_path: Path) -> None:
    module = _closure_module()
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_probe.py").write_text(
        "def test_known():\n    pass\n", encoding="utf-8",
    )
    progress = tmp_path / "pytest-progress.json"
    gate = {"name": "tests", "exit_code": 124, "timed_out": True,
            "output": str(tmp_path / "tests.txt")}
    result = {"status": "FAILED", "version": "0.3.6", "source_sha256": "0" * 64,
              "gates": [gate]}
    unsafe = (
        {"schema_version": 1, "identity": "test_probe.py::test_unknown"},
        {"schema_version": 1, "identity": "test_probe.py::test_known[private]"},
        {"schema_version": 1, "identity": "test_other.py::test_known"},
        {"schema_version": 1, "identity": "test_probe.py::test_known", "secret": "private"},
        {"schema_version": True, "identity": "test_probe.py::test_known"},
    )
    for item in unsafe:
        progress.write_text(json.dumps(item), encoding="utf-8")
        module._persist(tmp_path, result, tests_root=tests_root)
        report = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "没有可核验的测试标识" in report
        assert "test_probe.py::test_known" not in report
        assert "private" not in report
    progress.write_bytes(b"x" * (module.MAX_TEST_PROGRESS_BYTES + 1))
    assert module._last_started_test(progress, tests_root=tests_root) is None
    progress.unlink()
    progress.symlink_to(tests_root / "test_probe.py")
    assert module._last_started_test(progress, tests_root=tests_root) is None


def test_success_status_without_complete_cpu_matrix_is_rejected() -> None:
    path = Path(__file__).parents[1] / "scripts/run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module._acceptance_complete({
        "status": "SUCCEEDED", "reference_status": "VALIDATED", "cases": [],
    })


def test_cpu_matrix_rejects_missing_or_extra_negative_control_differences() -> None:
    source = Path(__file__).parents[1] / "scripts/run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = (
        "sync-worker_exit", "async-worker_exit", "sync-save_interrupt",
        "async-save_interrupt", "sync-corrupt", "async-corrupt", "sync-hang",
        "omit-rng", "omit-optimizer", "omit-cursor",
    )
    cases = []
    for name in names:
        if name.startswith("omit-"):
            mode, fault, omitted, exact = "sync", "worker_exit", name.removeprefix("omit-"), False
            differences = ["final model_sha256 differs"]
            if omitted == "cursor":
                differences.extend(
                    f"rank {rank} {kind} differs"
                    for rank in range(2)
                    for kind in ("effective sample sequence", "consumed batch sequence")
                )
        else:
            mode, fault = name.split("-", 1)
            omitted, exact, differences = "none", True, []
        cases.append({
            "name": name, "mode": mode, "fault": fault, "omit_state": omitted,
            "expected_exact": exact, "status": "PASSED", "recovery_count": 1,
            "fault_attributed": True,
            "validation": {
                "passed": exact, "differences": differences,
                "independent_reference": True,
                "comparison_kind": "INDEPENDENT_REFERENCE",
            },
        })
    acceptance = {
        "status": "SUCCEEDED", "reference_status": "VALIDATED",
        "config": {"run": {"device": "cpu", "backend": "gloo", "world_size": 2}},
        "cases": cases,
    }
    assert module._acceptance_complete(acceptance)
    for name, differences in (
        ("omit-rng", []),
        ("omit-optimizer", ["final model_sha256 differs", "unrelated difference"]),
        ("omit-cursor", ["final model_sha256 differs"]),
        ("sync-worker_exit", ["final model_sha256 differs"]),
    ):
        changed = json.loads(json.dumps(acceptance))
        next(case for case in changed["cases"] if case["name"] == name)["validation"][
            "differences"
        ] = differences
        assert not module._acceptance_complete(changed), name
    changed = json.loads(json.dumps(acceptance))
    changed["config"]["run"]["world_size"] = 1
    assert not module._acceptance_complete(changed)
    for name in ("sync-worker_exit", "omit-rng"):
        for marker in (None, False):
            changed = json.loads(json.dumps(acceptance))
            case = next(case for case in changed["cases"] if case["name"] == name)
            if marker is None:
                case["validation"].pop("independent_reference")
            else:
                case["validation"]["independent_reference"] = marker
            assert not module._acceptance_complete(changed)
        changed = json.loads(json.dumps(acceptance))
        case = next(case for case in changed["cases"] if case["name"] == name)
        case["validation"]["comparison_kind"] = "SELF_CHECK"
        assert not module._acceptance_complete(changed)


def test_gate_timeout_is_recorded_and_process_stops(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "scripts" / "run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GATE_TIMEOUT_SECONDS["static"] = 0.05
    started = time.monotonic()
    result = module._run(tmp_path, "static", [sys.executable, "-c", "import time; time.sleep(60)"])
    assert result["exit_code"] == 124
    assert result["timed_out"] is True
    assert "Gate timed out" in (tmp_path / "static.txt").read_text()
    assert time.monotonic() - started < 10


def test_gate_timeout_stops_child_in_a_separate_session(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "scripts" / "run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GATE_TIMEOUT_SECONDS["static"] = 1
    module.GATE_TERMINATION_GRACE_SECONDS = 0.2
    child_path = tmp_path / "child.pid"
    command = [
        sys.executable, "-c",
        (
            "import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', "
            "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'], "
            "start_new_session=True); "
            "open(sys.argv[1], 'w').write(str(child.pid)); time.sleep(60)"
        ),
        str(child_path),
    ]
    child_pid = None
    try:
        result = module._run(tmp_path, "static", command)
        child_pid = int(child_path.read_text(encoding="utf-8"))
        assert result["exit_code"] == 124 and result["timed_out"] is True
        state = subprocess.run(
            ["ps", "-p", str(child_pid), "-o", "stat="],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        assert not state or state.startswith("Z"), state
    finally:
        if child_pid is None and child_path.exists():
            child_pid = int(child_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
