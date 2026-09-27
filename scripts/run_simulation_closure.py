"""Run and preserve the full locally executable recovery acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

from trainguard import __version__
from trainguard.environment import source_sha256
from trainguard.events import utc_now, write_json_atomic
from trainguard.execution_inputs import execution_inputs_sha256

GATE_TIMEOUT_SECONDS = {
    "static": 180,
    "tests": 1800,
    "cpu-acceptance": 1200,
    "package": 300,
    "wheel": 300,
    "supply-chain": 1200,
    "fresh-install": 1200,
    "upgrade-boundary": 1200,
}
GATE_TERMINATION_GRACE_SECONDS = 5


def _process_table() -> dict[int, tuple[int, int, str]]:
    """Read parent, group and state before a timed-out gate loses its children."""
    result = subprocess.run(
        ["ps", "axww", "-o", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "stat="],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("cannot inspect timed-out gate processes")
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4:
            continue
        try:
            pid, parent, group = map(int, fields[:3])
        except ValueError:
            continue
        rows[pid] = (parent, group, fields[3])
    return rows


def _descendant_groups(root_pid: int) -> set[int]:
    rows = _process_table()
    descendants = {root_pid}
    while True:
        found = {pid for pid, (parent, _, _) in rows.items() if parent in descendants}
        if found <= descendants:
            break
        descendants.update(found)
    # A child can start a fresh session. Signal only groups led by this gate's
    # descendants, never a group that a child happened to join elsewhere.
    groups = {root_pid}
    for pid in descendants:
        entry = rows.get(pid)
        if entry is not None and entry[1] in descendants:
            groups.add(entry[1])
    return groups


def _live_groups(groups: set[int]) -> set[int]:
    return {
        current_group for _, current_group, state in _process_table().values()
        if current_group in groups and not state.startswith("Z")
    }


def _signal_groups(groups: set[int], signum: signal.Signals) -> None:
    for group in sorted(groups):
        if group == os.getpgrp():
            continue
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass


def _run(directory: Path, name: str, command: list[str]) -> dict:
    output = directory / f"{name}.txt"
    with output.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        timed_out = False
        try:
            exit_code = process.wait(timeout=GATE_TIMEOUT_SECONDS[name])
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                groups = _descendant_groups(process.pid)
            except RuntimeError:
                groups = {process.pid}
                stream.write("Could not inspect gate descendants before cleanup.\n")
            _signal_groups(groups, signal.SIGTERM)
            deadline = time.monotonic() + GATE_TERMINATION_GRACE_SECONDS
            while time.monotonic() < deadline:
                try:
                    if not _live_groups(groups):
                        break
                except RuntimeError:
                    break
                time.sleep(0.1)
            try:
                remaining = _live_groups(groups)
            except RuntimeError:
                remaining = groups
            _signal_groups(remaining, signal.SIGKILL)
            process.wait(timeout=5)
            stream.write(f"\nGate timed out after {GATE_TIMEOUT_SECONDS[name]} seconds.\n")
            exit_code = 124
    return {
        "name": name,
        "command": command,
        "exit_code": exit_code,
        "output": str(output),
        "timed_out": timed_out,
    }


def _run_bound(
    root: Path, directory: Path, name: str, command: list[str], expected: str
) -> dict:
    """Record the tested input bytes on both sides of every gate."""
    output = directory / f"{name}.txt"
    try:
        before = execution_inputs_sha256(root)
    except (OSError, ValueError) as exc:
        before = None
        error = f"cannot inspect execution inputs before {name}: {exc}"
    else:
        error = None if before == expected else f"execution inputs changed before {name}"
    if error is not None:
        output.write_text(error + "\n", encoding="utf-8")
        gate = {"name": name, "command": command, "exit_code": 1,
                "output": str(output), "timed_out": False}
    elif name == "package":
        try:
            from verify_sdist import verify_build_inputs

            verify_build_inputs(root)
        except ValueError as exc:
            output.write_text(f"source package input preflight failed: {exc}\n", encoding="utf-8")
            gate = {"name": name, "command": command, "exit_code": 1,
                    "output": str(output), "timed_out": False}
        else:
            gate = _run(directory, name, command)
    else:
        gate = _run(directory, name, command)
    try:
        after = execution_inputs_sha256(root)
    except (OSError, ValueError) as exc:
        after = None
        error = f"cannot inspect execution inputs after {name}: {exc}"
    else:
        if after != expected:
            error = f"execution inputs changed during {name}"
    gate["execution_inputs_before_sha256"] = before
    gate["execution_inputs_after_sha256"] = after
    if error is not None:
        gate["command_exit_code"] = gate["exit_code"]
        gate["exit_code"] = 1
        gate["input_integrity_error"] = error
        with output.open("a", encoding="utf-8") as stream:
            stream.write(error + "\n")
    return gate


def _persist(directory: Path, result: dict) -> None:
    write_json_atomic(directory / "result.json", result)
    lines = [
        "# 本机训练恢复模拟闭环",
        "",
        f"状态：{result['status']}",
        f"版本：{result['version']}",
        f"源码指纹：`{result['source_sha256']}`",
        "",
        "| 门槛 | 退出码 | 原始输出 |",
        "| --- | ---: | --- |",
    ]
    for gate in result["gates"]:
        lines.append(
            f"| {gate['name']} | {gate['exit_code']} | [{gate['name']}]({Path(gate['output']).name}) |"
        )
    if (directory / "test-artifacts").is_dir():
        lines.extend(["", "[测试故障与恢复原始目录](test-artifacts/)保留在本次结果目录中。"])
    acceptance = result.get("acceptance")
    if acceptance:
        passed = sum(case["status"] == "PASSED" for case in acceptance["cases"])
        lines.extend(
            [
                "",
                f"真实 CPU 验收：{acceptance['status']}；{passed}/{len(acceptance['cases'])} 个故障/负控案例通过。",
                f"[验收原始记录]({Path(result['acceptance_path']).relative_to(directory)})。",
            ]
        )
    lines.extend(
        [
            "",
            "本机矩阵包含真实 CPU 双进程恢复、双启动器拓扑、提交切点硬退出、",
            "异步协调与事件耐久顺序，以及对象条件提交和 epoch 接管的协议模拟。",
            "真实 CUDA、跨主机隔离、远程对象服务和主机断电仍需对应环境实测。",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_differences_complete(case: dict) -> bool:
    validation = case.get("validation")
    if not isinstance(validation, dict):
        return False
    differences = validation.get("differences")
    if not isinstance(differences, list) or any(type(item) is not str for item in differences):
        return False
    observed = set(differences)
    if len(observed) != len(differences):
        return False
    if not case["name"].startswith("omit-"):
        return not observed
    allowed = {"final model_sha256 differs", "final optimizer_sha256 differs"}
    required = {"final model_sha256 differs"}
    if case["name"] == "omit-cursor":
        sequences = {
            f"rank {rank} {name} differs"
            for rank in range(2)
            for name in ("effective sample sequence", "consumed batch sequence")
        }
        allowed |= sequences
        required |= sequences
    return required <= observed <= allowed


def _acceptance_complete(acceptance: dict) -> bool:
    if not isinstance(acceptance, dict):
        return False
    expected = {
        f"{mode}-{fault}": (mode, fault, "none", True)
        for mode, fault in [
            ("sync", "worker_exit"), ("async", "worker_exit"),
            ("sync", "save_interrupt"), ("async", "save_interrupt"),
            ("sync", "corrupt"), ("async", "corrupt"), ("sync", "hang"),
        ]
    }
    expected.update({
        f"omit-{state}": ("sync", "worker_exit", state, False)
        for state in ("rng", "optimizer", "cursor")
    })
    config = acceptance.get("config")
    settings = config.get("run") if isinstance(config, dict) else None
    cases = acceptance.get("cases")
    if (
        acceptance.get("status") != "SUCCEEDED"
        or acceptance.get("reference_status") != "VALIDATED"
        or not isinstance(settings, dict)
        or settings.get("device") != "cpu"
        or settings.get("backend") != "gloo"
        or type(settings.get("world_size")) is not int
        or settings["world_size"] != 2
        or not isinstance(cases, list)
        or len(cases) != len(expected)
        or not all(isinstance(case, dict) and isinstance(case.get("name"), str)
                   for case in cases)
        or {case["name"] for case in cases} != set(expected)
    ):
        return False
    return all(
        case.get("status") == "PASSED"
        and (case.get("mode"), case.get("fault"), case.get("omit_state")) == (
            expected[case["name"]][:3]
        )
        and case.get("expected_exact") is expected[case["name"]][3]
        and type(case.get("recovery_count")) is int
        and case["recovery_count"] == 1
        and case.get("fault_attributed") is True
        and isinstance(case.get("validation"), dict)
        and case["validation"].get("passed") is case["expected_exact"]
        and _case_differences_complete(case)
        for case in cases
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--previous-ref")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    release_manifest = json.loads(Path("docs/commercial/release-gates.json").read_text())
    previous = release_manifest["previous_release"]
    previous_ref = args.previous_ref or previous["git_commit"]
    directory = (args.output_root / f"simulation-{uuid.uuid4().hex[:12]}").resolve()
    directory.mkdir(parents=True)
    inputs_sha256 = execution_inputs_sha256(root)
    execution_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": utc_now(),
        "version": __version__,
        "source_sha256": source_sha256(),
        "execution_commit": execution_commit,
        "execution_inputs_sha256": inputs_sha256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "gates": [],
    }
    _persist(directory, result)
    commands = [
        ("static", ["uv", "run", "ruff", "check", "."]),
        (
            "tests",
            [
                "uv", "run", "pytest", "-q",
                f"--junitxml={directory / 'pytest.xml'}",
                f"--basetemp={directory / 'test-artifacts'}",
            ],
        ),
        (
            "cpu-acceptance",
            [
                "uv", "run", "trainguard", "acceptance", "--config",
                "configs/cpu_demo.yaml", "--output-root", str(directory / "acceptance"),
            ],
        ),
        ("package", ["uv", "build", "--wheel", "--sdist", "--build-constraints",
                     "build-constraints.txt", "--require-hashes", "--out-dir",
                     str(directory / "dist")]),
    ]
    class GateStopped(Exception):
        pass

    try:
        for name, command in commands:
            gate = _run_bound(root, directory, name, command, inputs_sha256)
            result["gates"].append(gate)
            _persist(directory, result)
            if gate["exit_code"]:
                result["status"] = "FAILED"
                raise GateStopped
        acceptance_paths = list((directory / "acceptance").glob("*/acceptance.json"))
        if len(acceptance_paths) != 1:
            result["status"] = "FAILED"
            result["reason"] = "acceptance record missing or ambiguous"
            raise GateStopped
        result["acceptance_path"] = str(acceptance_paths[0])
        result["acceptance"] = json.loads(acceptance_paths[0].read_text(encoding="utf-8"))
        if not _acceptance_complete(result["acceptance"]):
            result["status"] = "FAILED"
            result["reason"] = "acceptance matrix incomplete or inconsistent"
            raise GateStopped
        wheels = list((directory / "dist").glob("trainguard-*.whl"))
        if len(wheels) != 1:
            result["status"] = "FAILED"
            result["reason"] = "built wheel missing or ambiguous"
            raise GateStopped
        gate = _run_bound(
            root, directory,
            "wheel",
            ["uv", "run", "python", "scripts/verify_wheel.py", str(wheels[0])],
            inputs_sha256,
        )
        result["gates"].append(gate)
        if gate["exit_code"]:
            result["status"] = "FAILED"
            raise GateStopped
        result["artifact_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((directory / "dist").iterdir())
            if path.is_file() and (path.name.endswith(".whl") or path.name.endswith(".tar.gz"))
        }
        result["lock_sha256"] = hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest()
        gate = _run_bound(
            root, directory,
            "supply-chain",
            ["uv", "run", "python", "scripts/supply_chain.py", "--wheel", str(wheels[0]),
             "--output-dir", str(directory)],
            inputs_sha256,
        )
        result["gates"].append(gate)
        if gate["exit_code"]:
            result["status"] = "FAILED"
            raise GateStopped
        gate = _run_bound(
            root, directory,
            "fresh-install",
            ["uv", "run", "python", "scripts/verify_install.py", str(wheels[0])],
            inputs_sha256,
        )
        result["gates"].append(gate)
        if gate["exit_code"]:
            result["status"] = "FAILED"
            raise GateStopped
        gate = _run_bound(
            root, directory,
            "upgrade-boundary",
            ["uv", "run", "python", "scripts/verify_upgrade_boundary.py",
             "--previous-ref", previous_ref,
             "--expected-previous-commit", previous["git_commit"],
             "--expected-previous-wheel-sha256", previous["wheel_sha256"],
             "--expected-previous-lock-sha256", previous["lock_sha256"],
             "--current-wheel", str(wheels[0])],
            inputs_sha256,
        )
        result["gates"].append(gate)
        result["status"] = "SUCCEEDED" if gate["exit_code"] == 0 else "FAILED"
    except GateStopped:
        pass
    except Exception as exc:  # noqa: BLE001 - retain the failed gate record
        result["status"] = "FAILED"
        result["reason"] = f"gate execution failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            result["execution_commit_after"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            result["execution_commit_after"] = None
            result["reason"] = f"cannot inspect final execution commit: {exc}"
            result["status"] = "FAILED"
        else:
            if result["execution_commit_after"] != execution_commit:
                result["reason"] = "candidate commit changed during verification"
                result["status"] = "FAILED"
        try:
            result["execution_inputs_after_sha256"] = execution_inputs_sha256(root)
        except (OSError, ValueError) as exc:
            result["execution_inputs_after_sha256"] = None
            result["reason"] = f"cannot inspect final execution inputs: {exc}"
            result["status"] = "FAILED"
        else:
            if result["execution_inputs_after_sha256"] != inputs_sha256:
                result["reason"] = "execution inputs changed during verification"
                result["status"] = "FAILED"
        result["finished_at"] = utc_now()
        _persist(directory, result)
        print(directory)
    return 0 if result["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
