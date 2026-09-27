"""Bound local release gates and retain a machine-readable failure record."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


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
            "validation": {"passed": exact, "differences": differences},
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
