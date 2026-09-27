"""Run and preserve the full locally executable recovery acceptance gate."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import uuid
from pathlib import Path

import torch

from trainguard import __version__
from trainguard.environment import source_sha256
from trainguard.events import utc_now, write_json_atomic


def _run(directory: Path, name: str, command: list[str]) -> dict:
    output = directory / f"{name}.txt"
    with output.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "output": str(output),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    directory = (args.output_root / f"simulation-{uuid.uuid4().hex[:12]}").resolve()
    directory.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": utc_now(),
        "version": __version__,
        "source_sha256": source_sha256(),
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
        ("package", ["uv", "build", "--wheel", "--sdist", "--out-dir", str(directory / "dist")]),
    ]
    try:
        for name, command in commands:
            gate = _run(directory, name, command)
            result["gates"].append(gate)
            _persist(directory, result)
            if gate["exit_code"]:
                result["status"] = "FAILED"
                return 1
        acceptance_paths = list((directory / "acceptance").glob("*/acceptance.json"))
        if len(acceptance_paths) != 1:
            result["status"] = "FAILED"
            result["reason"] = "acceptance record missing or ambiguous"
            return 1
        result["acceptance_path"] = str(acceptance_paths[0])
        result["acceptance"] = json.loads(acceptance_paths[0].read_text(encoding="utf-8"))
        if result["acceptance"]["status"] != "SUCCEEDED":
            result["status"] = "FAILED"
            return 1
        wheels = list((directory / "dist").glob("trainguard-*.whl"))
        if len(wheels) != 1:
            result["status"] = "FAILED"
            result["reason"] = "built wheel missing or ambiguous"
            return 1
        gate = _run(
            directory,
            "wheel",
            ["uv", "run", "python", "scripts/verify_wheel.py", str(wheels[0])],
        )
        result["gates"].append(gate)
        result["status"] = "SUCCEEDED" if gate["exit_code"] == 0 else "FAILED"
        return 0 if result["status"] == "SUCCEEDED" else 1
    finally:
        result["finished_at"] = utc_now()
        _persist(directory, result)
        print(directory)


if __name__ == "__main__":
    sys.exit(main())
