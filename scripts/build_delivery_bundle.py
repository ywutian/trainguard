"""Package a source-bound evaluation or release candidate with locked dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from check_release_readiness import LOCAL_RAW_FILES, evaluate

from trainguard.events import write_json_atomic


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--gate-manifest", type=Path, default=Path("docs/commercial/release-gates.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads(args.readiness_report.read_text(encoding="utf-8"))
    fresh = evaluate(args.gate_manifest, args.wheel, args.sdist)
    if set(report) != set(fresh) or any(
        report[field] != fresh[field] for field in fresh if field != "checked_at"
    ):
        raise ValueError("readiness report is stale or inconsistent with gate evidence")
    if not fresh["evaluation_allowed"]:
        raise ValueError("local package and CPU gates must pass before a customer evaluation bundle")

    output = args.output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="delivery-stage-", dir=output.parent) as temporary:
        stage = Path(temporary)
        files = [
            args.wheel, args.sdist, root / "pyproject.toml", root / "uv.lock",
            root / "LICENSE", root / "docs/commercial/operations-runbook.md",
            root / "docs/commercial/customer-pilot-template.md",
            root / "docs/commercial/pilot-ledger-template.json",
            root / "docs/commercial/market-evidence-2026-09-26.md",
            args.gate_manifest,
        ]
        if len({path.name for path in files}) != len(files) or any(
            path.name in {"readiness-report.json", "delivery-manifest.json", "requirements.txt"}
            for path in files
        ):
            raise ValueError("delivery inputs contain conflicting file names")
        for path in files:
            shutil.copy2(path, stage / path.name)
        write_json_atomic(stage / "readiness-report.json", fresh)
        scripts = stage / "scripts"
        scripts.mkdir()
        shutil.copy2(root / "scripts/calculate_pilot_value.py", scripts / "calculate_pilot_value.py")
        evidence_map = {}
        for gate in fresh["gates"]:
            if gate["status"] == "PASS":
                source = root / gate["evidence"]
                target = stage / gate["evidence"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                evidence_map[gate["id"]] = target.relative_to(stage).as_posix()
                if gate["id"] in {"local_package", "local_cpu"}:
                    receipt = json.loads(source.read_text(encoding="utf-8"))
                    record = root / receipt["record_reference"]
                    record_target = stage / receipt["record_reference"]
                    record_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(record, record_target)
                    raw_dir = json.loads(record.read_text(encoding="utf-8"))["raw_evidence_dir"]
                    raw_target = stage / raw_dir
                    raw_target.mkdir(parents=True, exist_ok=True)
                    for name in LOCAL_RAW_FILES:
                        shutil.copy2(root / raw_dir / name, raw_target / name)
        requirements = stage / "requirements.txt"
        exported = subprocess.run(
            ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--format",
             "requirements.txt"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        requirements.write_text(exported.stdout, encoding="utf-8")
        version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
        manifest = {
            "schema_version": 1,
            "version": version,
            "status": "EVALUATION_ONLY",
            "production_release_authorized": False,
            "source_sha256": fresh["candidate_source_sha256"],
            "git_commit": report["git_commit"],
            "evidence_map": evidence_map,
            "files": {path.relative_to(stage).as_posix(): _digest(path)
                      for path in sorted(stage.rglob("*")) if path.is_file()},
        }
        write_json_atomic(stage / "delivery-manifest.json", manifest)
        stage.rename(output)
    print(json.dumps({"status": manifest["status"], "version": version,
                      "files": len(manifest["files"])}, sort_keys=True))


if __name__ == "__main__":
    main()
