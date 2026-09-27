"""Archive a successful local gate as private-data-free, source-bound evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from check_release_readiness import (
    LOCAL_RAW_FILES,
    _local_evidence,
    _receipt,
    package_source_sha256,
)

from trainguard.evidence_lineage import require_evidence_only_descendant
from trainguard.execution_inputs import execution_inputs_sha256

SUPPORTED_GATE_NAMES = {
    "static", "tests", "cpu-acceptance", "package", "wheel", "fresh-install",
    "upgrade-boundary",
}
OPTIONAL_GATE_NAMES = {"supply-chain"}
SIMULATION_NAME = re.compile(r"simulation-[0-9a-f]{12}\Z")
CHECKS = {
    "full_suite": True, "cpu_fault_matrix": True, "wheel_identity": True,
    "fresh_install": True, "upgrade_recovery": True,
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}\Z")
CASE_NAME = re.compile(r"[a-z0-9_-]+\Z")
CASE_FIELD = re.compile(r"[A-Za-z0-9_-]+\Z")
EXPECTED_CASES = {
    "sync-worker_exit", "async-worker_exit", "sync-save_interrupt",
    "async-save_interrupt", "sync-corrupt", "async-corrupt", "sync-hang",
    "omit-rng", "omit-optimizer", "omit-cursor",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required original evidence is missing or linked: {path.name}")
    return path.read_bytes()


def _json_file(path: Path) -> dict:
    try:
        value = json.loads(_read_file(path))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"original evidence is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"original evidence is not a mapping: {path.name}")
    return value


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _last_json(content: bytes, label: str) -> dict:
    try:
        line = content.decode("utf-8").splitlines()[-1]
        value = json.loads(line)
    except (IndexError, UnicodeError, ValueError) as exc:
        raise ValueError(f"original gate summary is unreadable: {label}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"original gate summary is not a mapping: {label}")
    return value


def _selected(value: dict, names: tuple[str, ...], label: str) -> dict:
    missing = set(names) - set(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")
    selected = {name: value[name] for name in names}
    try:
        _json_bytes(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-JSON values") from exc
    return selected


def _numeric_metrics(value: object) -> dict:
    if not isinstance(value, dict):
        return {}
    kept = {}
    for name, amount in value.items():
        if not isinstance(name, str) or not CASE_NAME.fullmatch(name):
            continue
        if type(amount) in (int, float) and math.isfinite(amount):
            kept[name] = amount
    return kept


def _safe_acceptance(acceptance: dict) -> dict:
    if (
        acceptance.get("status") != "SUCCEEDED"
        or acceptance.get("reference_status") != "VALIDATED"
        or not isinstance(acceptance.get("cases"), list)
        or len(acceptance["cases"]) != len(EXPECTED_CASES)
        or not isinstance(acceptance.get("environment"), dict)
    ):
        raise ValueError("original CPU acceptance result is incomplete")
    cases = []
    for original in acceptance["cases"]:
        if not isinstance(original, dict) or original.get("name") not in EXPECTED_CASES:
            raise ValueError("original CPU case identity is invalid")
        validation = original.get("validation")
        if not isinstance(validation, dict) or type(validation.get("passed")) is not bool:
            raise ValueError("original CPU case validation is missing")
        case = _selected(
            original,
            ("name", "mode", "fault", "omit_state", "expected_exact", "status",
             "recovery_count", "fault_attributed"),
            "CPU case",
        )
        if any(
            not isinstance(case[field], str) or not CASE_FIELD.fullmatch(case[field])
            for field in ("name", "mode", "fault", "omit_state", "status")
        ):
            raise ValueError("original CPU case labels are unsafe")
        differences = validation.get("differences", [])
        if not isinstance(differences, list) or any(not isinstance(item, str) for item in differences):
            raise ValueError("original CPU difference evidence is invalid")
        case["validation"] = {
            "passed": validation["passed"],
            "difference_sha256": [hashlib.sha256(item.encode()).hexdigest() for item in differences],
        }
        case["metrics"] = _numeric_metrics(original.get("metrics"))
        cases.append(case)
    if {case["name"] for case in cases} != EXPECTED_CASES:
        raise ValueError("original CPU cases are duplicated or incomplete")
    commit = acceptance["environment"].get("git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("original CPU environment commit is invalid")
    return {
        "status": "SUCCEEDED", "reference_status": "VALIDATED",
        "environment": {"git_commit": commit}, "cases": cases,
    }


def _safe_junit(original: bytes) -> tuple[bytes, dict]:
    try:
        root = ElementTree.fromstring(original)
    except ElementTree.ParseError as exc:
        raise ValueError("original test XML is unreadable") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("original test XML has no suite")
    output = ElementTree.Element("testsuites", {"name": "pytest tests"})
    total = skipped = gpu = preflight = 0
    for suite in suites:
        attrs = {field: suite.get(field, "") for field in (
            "errors", "failures", "skipped", "tests"
        )}
        if any(not value.isdecimal() for value in attrs.values()):
            raise ValueError("original test XML suite counts are invalid")
        attrs["name"] = "pytest"
        attrs["hostname"] = "redacted-host"
        attrs["time"] = "0"
        attrs["timestamp"] = "redacted"
        target = ElementTree.SubElement(output, "testsuite", attrs)
        for case in suite.findall("testcase"):
            classname, name = case.get("classname"), case.get("name")
            if (
                not isinstance(classname, str) or not isinstance(name, str)
                or any(mark in classname + name for mark in ("/", "\\", "@"))
            ):
                raise ValueError("test identity contains a host path or unsafe text")
            duration = case.get("time", "0")
            if re.fullmatch(r"\d+(?:\.\d+)?", duration) is None:
                raise ValueError("original test duration is invalid")
            projected = ElementTree.SubElement(target, "testcase", {
                "classname": classname, "name": name, "time": duration,
            })
            if case.find("failure") is not None or case.find("error") is not None:
                raise ValueError("original test XML contains a failure")
            marker = case.find("skipped")
            if marker is not None:
                message = marker.get("message")
                if message not in {
                    "requires two actual CUDA devices",
                    "this test verifies the unavailable-device preflight",
                }:
                    raise ValueError("original test XML has an unapproved skip")
                ElementTree.SubElement(projected, "skipped", {"message": message})
                skipped += 1
                gpu += message == "requires two actual CUDA devices"
                preflight += message == "this test verifies the unavailable-device preflight"
            total += 1
    tests = {"passed": total - skipped, "skipped_no_gpu": gpu}
    if preflight:
        tests["skipped_device_preflight"] = preflight
    return ElementTree.tostring(output, encoding="utf-8", xml_declaration=True), tests


def _safe_result(result: dict, acceptance: dict) -> dict:
    fields = (
        "schema_version", "status", "started_at", "finished_at", "version",
        "source_sha256", "execution_commit", "execution_commit_after",
        "execution_inputs_sha256", "execution_inputs_after_sha256", "python", "torch",
        "artifact_sha256", "lock_sha256",
    )
    output = _selected(result, fields, "local result")
    for field in ("python", "torch", "version"):
        value = output[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9][0-9A-Za-z.+-]{0,63}", value) is None:
            raise ValueError("original local runtime identity is unsafe")
    for field in ("started_at", "finished_at"):
        value = output[field]
        if not isinstance(value, str):
            raise TypeError("original local time is invalid")
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("original local time is invalid") from exc
    output["acceptance"] = acceptance
    output["acceptance_path"] = "raw/acceptance.json"
    gates = result.get("gates")
    gate_names = {
        name.removesuffix(".txt") for name in LOCAL_RAW_FILES
        if name.endswith(".txt") and name != "supply-chain-requirements.txt"
    }
    if gate_names not in (SUPPORTED_GATE_NAMES, SUPPORTED_GATE_NAMES | OPTIONAL_GATE_NAMES):
        raise ValueError("the release gate contains an unsupported original output")
    if not isinstance(gates, list) or len(gates) != len(gate_names):
        raise ValueError("original local gate list is incomplete")
    output["gates"] = []
    for gate in gates:
        if not isinstance(gate, dict) or gate.get("name") not in gate_names:
            raise ValueError("original local gate identity is invalid")
        row = _selected(
            gate,
            ("name", "exit_code", "timed_out", "execution_inputs_before_sha256",
             "execution_inputs_after_sha256"),
            "local gate",
        )
        row["output"] = f"raw/{gate['name']}.txt"
        command = gate.get("command")
        if not isinstance(command, list) or any(not isinstance(part, str) for part in command):
            raise ValueError("original local gate command is invalid")
        row["original_command_sha256"] = hashlib.sha256(_json_bytes({"command": command})).hexdigest()
        output["gates"].append(row)
    if {gate["name"] for gate in output["gates"]} != gate_names:
        raise ValueError("original local gates are duplicated")
    return output


def _original_reference(source: Path) -> str:
    # A runner-generated ID plus the original digest identifies the source
    # without publishing a parent path that may include a username.
    return f"simulation:{source.name}"


def _existing_original(evidence_root: Path, original_result_sha256: str) -> bool:
    for path in evidence_root.rglob("local-validation*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        originals = record.get("unredacted_source_sha256", {}) if isinstance(record, dict) else {}
        if isinstance(originals, dict) and originals.get("result.json") == original_result_sha256:
            return True
    return False


def _safe_license_rows(rows: object) -> list[dict]:
    if not isinstance(rows, list):
        raise TypeError("original license list is invalid")
    output = []
    for entry in rows:
        if not isinstance(entry, dict):
            raise TypeError("original license entry is invalid")
        data = entry.get("license")
        if isinstance(data, dict):
            identity = data.get("id") or data.get("name")
            key = "id" if data.get("id") else "name"
            if isinstance(identity, str) and re.fullmatch(r"[A-Za-z0-9 .+():_&|-]{1,128}", identity):
                output.append({"license": {key: identity}})
            else:
                output.append({"license": {"name": "redacted-sha256:" + hashlib.sha256(
                    _json_bytes(entry)
                ).hexdigest()}})
        elif isinstance(entry.get("expression"), str):
            expression = entry["expression"]
            if re.fullmatch(r"[A-Za-z0-9 .+():_&|-]{1,128}", expression):
                output.append({"expression": expression})
            else:
                output.append({"expression": "redacted-sha256:" + hashlib.sha256(
                    _json_bytes(entry)
                ).hexdigest()})
        else:
            raise TypeError("original license entry has no identity")
    return output


def _safe_supply_chain(source: Path, root: Path, source_digest: str,
                       inputs_digest: str, artifacts: dict[str, str],
                       original_hashes: dict[str, str]) -> dict[str, bytes]:
    from supply_chain import RAW_FILES, RECEIPT, verify_supply_chain

    names = {"supply-chain.txt", *RAW_FILES, RECEIPT}
    if not names.issubset(LOCAL_RAW_FILES):
        raise ValueError("the release gate omits required supply-chain files")
    wheel_name = next(name for name in artifacts if name.endswith(".whl"))
    expected = {
        "candidate_version": tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"],
        "source_sha256": source_digest,
        "execution_inputs_sha256": inputs_digest,
        "wheel_sha256": artifacts[wheel_name],
        "lock_sha256": artifacts["uv.lock"],
        "tool_lock_sha256": _digest(root / "scripts/supply-chain-tools.txt"),
        "security_policy_sha256": _digest(root / "SECURITY.md"),
        "first_party_license_sha256": _digest(root / "LICENSE"),
        "security_channel_record_sha256": _digest(
            root / "docs/commercial/security-channel-2026-09-27.json"
        ),
    }
    verify_supply_chain(source, expected)
    sbom = _json_file(source / "supply-chain-sbom.json")
    licenses = _json_file(source / "supply-chain-licenses.json")
    audit = _json_file(source / "supply-chain-audit.json")
    installed = json.loads(_read_file(source / "supply-chain-installed.json"))
    receipt = _json_file(source / RECEIPT)
    if not isinstance(installed, list):
        raise TypeError("original installed package inventory is invalid")

    def package(row: object) -> dict:
        if not isinstance(row, dict):
            raise TypeError("original package entry is invalid")
        identity = _selected(row, ("name", "version"), "installed package")
        for value in identity.values():
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._+-]{1,128}", value):
                raise ValueError("original package identity is unsafe")
        return identity

    references = {
        component["bom-ref"]: "sha256:" + hashlib.sha256(
            component["bom-ref"].encode()
        ).hexdigest()
        for component in sbom["components"]
    }
    safe_components = []
    for component in sbom["components"]:
        safe_components.append({
            **package(component), "bom-ref": references[component["bom-ref"]],
            "licenses": _safe_license_rows(component.get("licenses", [])),
        })
    safe_sbom = {
        "bomFormat": sbom["bomFormat"], "specVersion": sbom["specVersion"],
        "components": safe_components,
        "dependencies": [
            {"ref": references[row["ref"]],
             "dependsOn": [references[target] for target in row.get("dependsOn", [])]}
            for row in sbom["dependencies"]
        ],
    }
    safe_licenses = {
        "schema_version": licenses["schema_version"], "source": licenses["source"],
        "components": [
            {**package(row), "licenses": _safe_license_rows(row.get("licenses", []))}
            for row in licenses["components"]
        ],
    }
    safe_audit = {"dependencies": [], "fixes": []}
    for row in audit["dependencies"]:
        if "skip_reason" in row:
            safe_audit["dependencies"].append({
                "name": row["name"], "skip_reason": "first-party package",
            })
        else:
            safe_audit["dependencies"].append({
                **package(row), "vulns": [
                    {"id": "sha256:" + hashlib.sha256(item["id"].encode()).hexdigest()}
                    for item in row["vulns"]
                ],
            })
    safe = {
        "supply-chain-sbom.json": _json_bytes(safe_sbom),
        "supply-chain-licenses.json": _json_bytes(safe_licenses),
        "supply-chain-audit.json": _json_bytes(safe_audit),
        "supply-chain-requirements.txt": (
            "Original locked requirements SHA-256: "
            + original_hashes["supply-chain-requirements.txt"] + "\n"
        ).encode(),
    }
    # The scanner treats the installed inventory as a top-level array.
    safe["supply-chain-installed.json"] = (
        json.dumps([package(row) for row in installed], indent=2, sort_keys=True) + "\n"
    ).encode()
    receipt_fields = (
        "schema_version", "status", "generated_at_utc", "source_sha256",
        "execution_inputs_sha256", "wheel_sha256", "lock_sha256", "tool_lock_sha256",
        "security_policy_sha256", "first_party_license_sha256",
        "security_channel_record_sha256", "python_version", "platform", "machine",
        "audit_service", "audit_exit_code", "audit_tool_version", "uv_version",
        "sbom_tool_version", "sbom_private_references_removed",
        "database_snapshot_available", "candidate_version", "component_count",
        "third_party_count", "known_vulnerability_count", "unscanned_third_party",
        "unlicensed_third_party",
    )
    safe_receipt = _selected(receipt, receipt_fields, "supply-chain receipt")
    safe_receipt["files"] = {
        name: hashlib.sha256(safe[name]).hexdigest() for name in RAW_FILES
    }
    safe[RECEIPT] = _json_bytes(safe_receipt)
    safe["supply-chain.txt"] = _json_bytes(_selected(
        _last_json(_read_file(source / "supply-chain.txt"), "supply-chain.txt"),
        ("status", "component_count", "third_party_count", "known_vulnerability_count",
         "unscanned_third_party", "unlicensed_third_party", "wheel_sha256"),
        "supply-chain gate",
    ))
    return safe


def archive(source: Path, label: str, root: Path) -> Path:
    """Publish one immutable machine-verified archive; human approval stays separate."""
    root = root.resolve()
    source = source.resolve()
    if not LABEL.fullmatch(label) or label in {".", ".."}:
        raise ValueError("archive label must be a short path-safe name")
    evidence_root = root / "docs/commercial/evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        raise ValueError("local evidence destination is missing or linked")
    final = evidence_root / f"local-{label}"
    if final.exists() or final.is_symlink():
        raise FileExistsError("local evidence label already exists")
    if not SIMULATION_NAME.fullmatch(source.name) or not source.is_dir():
        raise ValueError("source must be a simulation output directory")
    result = _json_file(source / "result.json")
    acceptance_path = Path(result.get("acceptance_path", ""))
    if (
        not acceptance_path.is_absolute()
        or not acceptance_path.resolve().is_relative_to(source)
        or acceptance_path.is_symlink()
        or not acceptance_path.is_file()
    ):
        raise ValueError("original acceptance reference is missing or unsafe")
    originals = {
        name: _read_file(acceptance_path if name == "acceptance.json" else source / name)
        for name in LOCAL_RAW_FILES
    }
    original_hashes = {name: hashlib.sha256(content).hexdigest()
                       for name, content in originals.items()}
    if json.loads(originals["result.json"]) != result:
        raise ValueError("original result changed while archiving")
    if _existing_original(evidence_root, original_hashes["result.json"]):
        raise ValueError("this simulation result has already been archived")
    source_digest = package_source_sha256(root)
    inputs_digest = execution_inputs_sha256(root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    if (
        result.get("status") != "SUCCEEDED"
        or result.get("version") != version
        or result.get("source_sha256") != source_digest
        or result.get("execution_inputs_sha256") != inputs_digest
        or result.get("execution_inputs_after_sha256") != inputs_digest
        or result.get("execution_commit") != result.get("execution_commit_after")
        or not isinstance(result.get("execution_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", result["execution_commit"]) is None
    ):
        raise ValueError("simulation status, source, input or commit identity differs")
    require_evidence_only_descendant(root, result["execution_commit"], commit)
    acceptance = json.loads(originals["acceptance.json"])
    if not isinstance(acceptance, dict):
        raise TypeError("original acceptance result is not a mapping")
    if result.get("acceptance") != acceptance:
        raise ValueError("original acceptance result differs from its reference")
    if not isinstance(result.get("gates"), list):
        raise TypeError("original gate list is missing")
    for gate in result["gates"]:
        if (
            not isinstance(gate, dict)
            or gate.get("name") not in (SUPPORTED_GATE_NAMES | OPTIONAL_GATE_NAMES)
            or Path(gate.get("output", "")).resolve() != (source / f"{gate['name']}.txt")
        ):
            raise ValueError("original gate output reference differs")
    wheels = list((source / "dist").glob(f"trainguard-{version}-*.whl"))
    sdists = list((source / "dist").glob(f"trainguard-{version}.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("original package artifacts are missing or ambiguous")
    if wheels[0].is_symlink() or sdists[0].is_symlink():
        raise ValueError("original package artifacts cannot be linked")
    artifacts = {
        wheels[0].name: _digest(wheels[0]), sdists[0].name: _digest(sdists[0]),
        "uv.lock": _digest(root / "uv.lock"),
    }
    if (
        result.get("artifact_sha256") != {name: digest for name, digest in artifacts.items()
                                           if name != "uv.lock"}
        or result.get("lock_sha256") != artifacts["uv.lock"]
    ):
        raise ValueError("original package artifact hashes differ")
    previous = _json_file(root / "docs/commercial/release-gates.json")["previous_release"]
    safe_acceptance = _safe_acceptance(acceptance)
    safe_result = _safe_result(result, safe_acceptance)
    junit, tests = _safe_junit(originals["pytest.xml"])
    summary = next(
        (line for line in reversed(originals["tests.txt"].decode("utf-8").splitlines())
         if re.fullmatch(
             r"\d+ passed(?:, \d+ skipped)?(?:, \d+ warnings)? in [0-9:.s() ]+",
             line,
         )),
        None,
    )
    if summary is None:
        raise ValueError("original test summary is missing")
    safe_files = {
        "result.json": _json_bytes(safe_result),
        "acceptance.json": _json_bytes(safe_acceptance),
        "pytest.xml": junit,
        "tests.txt": (summary + "\n").encode(),
        "static.txt": b"Static check succeeded; original output hash is in local-validation.json.\n",
        "cpu-acceptance.txt": (
            f"CPU acceptance: SUCCEEDED; {len(safe_acceptance['cases'])} cases recorded.\n"
        ).encode(),
        "package.txt": ("Artifacts: " + ", ".join(sorted(artifacts)) + "\n").encode(),
    }
    summaries = {
        "wheel.txt": ("passed", "version", "source_sha256"),
        "fresh-install.txt": (
            "version", "wheel_sha256", "installed_outside_checkout", "completed_run",
            "recovered_run_matches_reference", "support_export_checked",
            "run_data_preserved_after_uninstall", "run_data_files_checked",
            "attributed_faults", "recoveries",
        ),
        "upgrade-boundary.txt": (
            "current_version", "current_wheel_sha256", "current_lock_sha256",
            "previous_commit_sha", "previous_version", "previous_wheel_sha256",
            "previous_lock_sha256", "new_version_rejected_interrupted_old_run",
            "old_locked_environment_resumed_exactly", "old_run_files_unchanged_after_rejection",
        ),
    }
    for name, fields in summaries.items():
        safe_files[name] = (json.dumps(
            _selected(_last_json(originals[name], name), fields, name), sort_keys=True,
            allow_nan=False,
        ) + "\n").encode()
    has_supply_chain = "supply-chain" in {gate["name"] for gate in safe_result["gates"]}
    if has_supply_chain:
        safe_files.update(_safe_supply_chain(
            source, root, source_digest, inputs_digest, artifacts, original_hashes
        ))
    if set(safe_files) != LOCAL_RAW_FILES:
        raise ValueError("sanitized evidence file set is incomplete")
    final_relative = final.relative_to(root).as_posix()
    scope = f"local CPU/Gloo; Python {result['python']}; PyTorch {result['torch']}"
    if any(character in scope for character in ("/Users/", "/home/", "@", "\\")):
        raise ValueError("local scope contains a host identity")
    with tempfile.TemporaryDirectory(prefix=".local-evidence-stage-", dir=evidence_root) as name:
        stage = Path(name)
        raw = stage / "raw"
        raw.mkdir()
        for filename, content in safe_files.items():
            (raw / filename).write_bytes(content)
        raw_hashes = {filename: _digest(raw / filename) for filename in LOCAL_RAW_FILES}
        checks = {**CHECKS, **({"supply_chain": True} if has_supply_chain else {})}
        record = {
            "schema_version": 1, "status": "SUCCEEDED", "scope": scope,
            "candidate_source_sha256": source_digest,
            "execution_inputs_sha256": inputs_digest,
            "execution_commit": result["execution_commit"],
            "artifacts": artifacts, "checks": checks,
            "tests": tests,
            "cpu_cases": {case["name"]: case["status"] for case in safe_acceptance["cases"]},
            "raw_evidence_dir": (raw.relative_to(root)).as_posix(),
            "raw_files": raw_hashes,
            "unredacted_source_sha256": original_hashes,
            "full_original_result_reference": _original_reference(source),
            "redaction": "Structured summaries only; original gate files remain at the recorded simulation reference.",
            "human_approval": False,
        }
        _local_evidence(root, record, source_digest, artifacts, previous)
        record["raw_evidence_dir"] = f"{final_relative}/raw"
        record_path = stage / "local-validation.json"
        record_path.write_bytes(_json_bytes(record))
        record_hash = _digest(record_path)
        reviewed_at = datetime.now(UTC).isoformat()
        for gate_id, receipt_checks in (
            ("local_package", ("wheel_identity", "fresh_install", "upgrade_recovery",
                               *(["supply_chain"] if has_supply_chain else []))),
            ("local_cpu", ("full_suite", "cpu_fault_matrix")),
        ):
            receipt = {
                "schema_version": 1, "gate_id": gate_id, "decision": "PASS",
                "candidate_source_sha256": source_digest,
                "execution_commit": result["execution_commit"],
                "execution_inputs_sha256": inputs_digest,
                "artifacts": artifacts, "scope": scope,
                "checks": {field: True for field in receipt_checks},
                "record_reference": f"{final_relative}/local-validation.json",
                "record_sha256": record_hash,
                "reviewer_role": "automated_local_evidence_verification",
                "approval_reference": f"machine-record-sha256:{record_hash}",
                "reviewed_at": reviewed_at, "human_approval": False,
            }
            (stage / f"{gate_id}.json").write_bytes(_json_bytes(receipt))
        for filename, original_digest in original_hashes.items():
            path = acceptance_path if filename == "acceptance.json" else source / filename
            if _digest(path) != original_digest:
                raise ValueError("original evidence changed while archiving")
        if final.exists() or final.is_symlink():
            raise FileExistsError("local evidence label already exists")
        stage.rename(final)
        try:
            for gate_id in ("local_package", "local_cpu"):
                _receipt(final / f"{gate_id}.json", gate_id, source_digest,
                         artifacts, previous)
        except BaseException:
            shutil.rmtree(final)
            raise
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulation_dir", type=Path)
    parser.add_argument("--label", required=True)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = archive(arguments.simulation_dir, arguments.label, root)
    receipts = {
        name: _digest(output / f"{name}.json") for name in ("local_package", "local_cpu")
    }
    print(json.dumps({"evidence_dir": output.relative_to(root).as_posix(),
                      "receipts": receipts}, sort_keys=True))


if __name__ == "__main__":
    main()
