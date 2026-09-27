"""Execution evidence remains valid only for the bytes that were tested."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from trainguard.evidence_lineage import require_evidence_only_descendant
from trainguard.execution_inputs import execution_inputs_sha256


@pytest.fixture
def candidate(tmp_path: Path) -> Path:
    for name in ("src/trainguard", "tests", "scripts", "configs", "docs/evidence"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "src/trainguard/__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests/test_recovery.py").write_text(
        "def test_recovery():\n    assert VALUE == 1\n", encoding="utf-8"
    )
    (tmp_path / "scripts/gate.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    (tmp_path / "configs/cpu.yaml").write_text("world_size: 2\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.3.5"\n', encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("path,replacement", [
    ("tests/test_recovery.py", "def test_recovery():\n    assert True\n"),
    ("scripts/gate.py", "raise SystemExit(1)\n"),
    ("configs/cpu.yaml", "world_size: 1\n"),
    ("src/trainguard/__init__.py", "VALUE = 2\n"),
    ("pyproject.toml", '[project]\nversion = "0.3.6"\n'),
    ("uv.lock", "version = 2\n"),
])
def test_execution_digest_changes_for_each_gate_input(
    candidate: Path, path: str, replacement: str
) -> None:
    original = execution_inputs_sha256(candidate)
    (candidate / path).write_text(replacement, encoding="utf-8")
    assert execution_inputs_sha256(candidate) != original


def test_successful_command_does_not_pass_if_test_changes_during_gate(candidate: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/run_simulation_closure.py"
    spec = importlib.util.spec_from_file_location("run_simulation_closure", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = execution_inputs_sha256(candidate)
    output = candidate / "verification"
    output.mkdir()
    test_file = candidate / "tests/test_recovery.py"
    mutation = (
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text("
        "'def test_recovery():\\n    assert True\\n')"
    )
    gate = module._run_bound(
        candidate, output, "static",
        [sys.executable, "-c", mutation, str(test_file)],
        expected,
    )
    assert gate["command_exit_code"] == 0
    assert gate["exit_code"] == 1
    assert gate["execution_inputs_before_sha256"] == expected
    assert gate["execution_inputs_after_sha256"] != expected
    assert "execution inputs changed during static" in (output / "static.txt").read_text()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Evidence Tester", "-c", "user.email=test@example.invalid",
         "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


def test_evidence_only_descendant_keeps_receipt_but_same_name_assertion_invalidates_it(
    candidate: Path,
) -> None:
    _git(candidate, "init", "-q")
    execution_commit = _commit(candidate, "Record executable candidate")
    tested_digest = execution_inputs_sha256(candidate)
    (candidate / "docs/evidence/receipt.json").write_text("{}\n", encoding="utf-8")
    evidence_commit = _commit(candidate, "Record validation evidence")
    require_evidence_only_descendant(candidate, execution_commit, evidence_commit)
    assert execution_inputs_sha256(candidate) == tested_digest

    (candidate / "tests/test_recovery.py").write_text(
        "def test_recovery():\n    assert True\n", encoding="utf-8"
    )
    changed_commit = _commit(candidate, "Change recovery assertion")
    with pytest.raises(ValueError, match="executable or unreviewed file changed"):
        require_evidence_only_descendant(candidate, execution_commit, changed_commit)
    assert execution_inputs_sha256(candidate) != tested_digest
