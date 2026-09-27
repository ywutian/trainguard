"""Build and verify source-bound Python supply-chain evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

RAW_FILES = {
    "supply-chain-sbom.json",
    "supply-chain-licenses.json",
    "supply-chain-audit.json",
    "supply-chain-installed.json",
    "supply-chain-requirements.txt",
}
RECEIPT = "supply-chain-receipt.json"
PRIVACY_MARKER = "supply-chain-privacy-checked.txt"
FIRST_PARTY = "trainguard"
SBOM_TOOL_VERSION = "7.4.0"
AUDIT_TOOL_VERSION = "2.10.1"
URL_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"'<>]+")
ABSOLUTE_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9:/\\])/(?!/)[A-Za-z0-9_.~%-]+"
    r"(?:/[A-Za-z0-9_.~%-]+)*(?=$|[\s\"',;)}\]])"
)
ABSOLUTE_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>]+")
SECRET_URL_KEYS = {
    "token", "access_token", "api_key", "apikey", "key", "password",
    "passwd", "secret", "credential", "authorization", "signature", "sig",
    "auth", "auth_token", "access_key", "secret_key", "client_secret",
    "awsaccesskeyid", "googleaccessid",
}
SECRET_URL_KEY_PREFIXES = ("x-amz-", "x-goog-")


class SupplyChainInvalid(ValueError):
    """Candidate supply-chain evidence is absent, changed, or incomplete."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_candidate_license(wheel: Path, license_file: Path) -> str:
    """Require the candidate wheel to carry the reviewed MIT license bytes."""
    try:
        expected = license_file.read_bytes()
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise SupplyChainInvalid("candidate wheel has duplicate archive members")
            wheel_parts = wheel.name.split("-")
            if len(wheel_parts) < 5 or wheel_parts[0] != FIRST_PARTY:
                raise SupplyChainInvalid("candidate wheel file name is invalid")
            expected_metadata = f"trainguard-{wheel_parts[1]}.dist-info/METADATA"
            metadata = [name for name in members if name.endswith(".dist-info/METADATA")]
            if metadata != [expected_metadata]:
                raise SupplyChainInvalid("candidate wheel metadata is missing or ambiguous")
            prefix = metadata[0].removesuffix("METADATA")
            license_member = f"{prefix}licenses/LICENSE"
            if license_member not in members or archive.read(license_member) != expected:
                raise SupplyChainInvalid("candidate wheel LICENSE differs from reviewed source")
            headers = Parser().parsestr(archive.read(metadata[0]).decode("utf-8"), headersonly=True)
            if (
                headers.get_all("Name") != [FIRST_PARTY]
                or headers.get_all("Version") != [wheel_parts[1]]
                or headers.get_all("License-Expression") != ["MIT"]
                or headers.get_all("License-File") != ["LICENSE"]
            ):
                raise SupplyChainInvalid("candidate wheel license metadata differs")
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise SupplyChainInvalid("candidate wheel license cannot be verified") from exc
    return hashlib.sha256(expected).hexdigest()


def _name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SupplyChainInvalid("package name is missing")
    return re.sub(r"[-_.]+", "-", value).lower()


def _load(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise SupplyChainInvalid(f"supply-chain evidence is missing or linked: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SupplyChainInvalid(f"supply-chain evidence is unreadable: {path.name}") from exc


def _private_url(url: str) -> bool:
    """Classify links that can disclose local locations or access credentials."""
    try:
        parsed = urlsplit(url)
    except ValueError:
        return True
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        return True
    return any(
        key.lower() in SECRET_URL_KEYS
        or key.lower().startswith(SECRET_URL_KEY_PREFIXES)
        for part in (parsed.query, parsed.fragment)
        for key, _ in parse_qsl(part, keep_blank_values=True)
    )


def _verify_output_privacy(directory: Path) -> None:
    """Reject local paths or credential-bearing links in every shipped scanner file."""
    for name in sorted(RAW_FILES | {RECEIPT}):
        payload = (directory / name).read_text(encoding="utf-8")
        if ABSOLUTE_POSIX_PATH.search(payload) or ABSOLUTE_WINDOWS_PATH.search(payload):
            raise SupplyChainInvalid(f"supply-chain output contains a local path: {name}")
        for match in URL_PATTERN.finditer(payload):
            if _private_url(match.group().rstrip(".,;)]}")):
                raise SupplyChainInvalid(f"supply-chain output contains a private URL: {name}")


def _packages(rows: object, *, kind: str) -> dict[str, str]:
    if not isinstance(rows, list) or not rows:
        raise SupplyChainInvalid(f"{kind} package list is missing")
    found = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SupplyChainInvalid(f"{kind} package entry is invalid")
        name = _name(row.get("name"))
        version = row.get("version")
        if name in found or not isinstance(version, str) or not version:
            raise SupplyChainInvalid(f"{kind} package identity is ambiguous")
        found[name] = version
    return found


def _analysis(directory: Path) -> dict:
    installed = _load(directory / "supply-chain-installed.json")
    bom = _load(directory / "supply-chain-sbom.json")
    licenses = _load(directory / "supply-chain-licenses.json")
    audit = _load(directory / "supply-chain-audit.json")
    installed_names = _packages(installed, kind="installed")
    if (
        not isinstance(bom, dict)
        or bom.get("bomFormat") != "CycloneDX"
        or bom.get("specVersion") != "1.6"
        or not isinstance(bom.get("components"), list)
        or not isinstance(bom.get("dependencies"), list)
    ):
        raise SupplyChainInvalid("CycloneDX SBOM is invalid")
    bom_names = _packages(bom["components"], kind="SBOM")
    if bom_names != installed_names:
        raise SupplyChainInvalid("SBOM differs from the installed candidate environment")
    refs = set()
    for component in bom["components"]:
        reference = component.get("bom-ref")
        if not isinstance(reference, str) or not reference or reference in refs:
            raise SupplyChainInvalid("SBOM component reference is invalid")
        refs.add(reference)
    dependency_refs = set()
    for dependency in bom["dependencies"]:
        if not isinstance(dependency, dict) or not isinstance(dependency.get("ref"), str):
            raise SupplyChainInvalid("SBOM dependency relation is invalid")
        reference = dependency["ref"]
        targets = dependency.get("dependsOn", [])
        if reference in dependency_refs or not isinstance(targets, list) or any(
            not isinstance(target, str) or target not in refs for target in targets
        ):
            raise SupplyChainInvalid("SBOM dependency relation is invalid")
        dependency_refs.add(reference)
    if dependency_refs != refs or "file://" in json.dumps(bom):
        raise SupplyChainInvalid("SBOM dependency graph or local-path redaction is incomplete")
    if (
        not isinstance(licenses, dict)
        or licenses.get("schema_version") != 1
        or licenses.get("source") != "CycloneDX declared package metadata"
    ):
        raise SupplyChainInvalid("license inventory is invalid")
    license_rows = licenses.get("components")
    license_names = _packages(license_rows, kind="license")
    if license_names != installed_names:
        raise SupplyChainInvalid("license inventory differs from installed packages")
    declared = {
        (_name(component["name"]), component["version"]): component.get("licenses", [])
        for component in bom["components"]
    }
    unlicensed = []
    for row in license_rows:
        key = (_name(row["name"]), row["version"])
        if row.get("licenses") != declared[key]:
            raise SupplyChainInvalid("license inventory differs from declared SBOM metadata")
        if key[0] != FIRST_PARTY and not row["licenses"]:
            unlicensed.append(key[0])
    if not isinstance(audit, dict) or not isinstance(audit.get("dependencies"), list):
        raise SupplyChainInvalid("known-vulnerability scan output is invalid")
    audited = {}
    skipped = []
    vulnerabilities = []
    for row in audit["dependencies"]:
        if not isinstance(row, dict):
            raise SupplyChainInvalid("known-vulnerability scan entry is invalid")
        name = _name(row.get("name"))
        if name in audited or name not in installed_names:
            raise SupplyChainInvalid("known-vulnerability scan coverage is ambiguous")
        if isinstance(row.get("skip_reason"), str):
            if name != FIRST_PARTY:
                skipped.append(name)
            audited[name] = installed_names[name]
            continue
        if row.get("version") != installed_names[name] or not isinstance(row.get("vulns"), list):
            raise SupplyChainInvalid("known-vulnerability scan package identity differs")
        audited[name] = row["version"]
        for item in row["vulns"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise SupplyChainInvalid("known-vulnerability result is invalid")
            vulnerabilities.append({"name": name, "id": item["id"]})
    if audited != installed_names:
        raise SupplyChainInvalid("known-vulnerability scan omitted installed packages")
    if not isinstance(audit.get("fixes"), list) or audit["fixes"]:
        raise SupplyChainInvalid("known-vulnerability scan attempted a package change")
    if installed_names.get(FIRST_PARTY) is None:
        raise SupplyChainInvalid("candidate wheel is absent from the installed environment")
    return {
        "candidate_version": installed_names[FIRST_PARTY],
        "component_count": len(installed_names),
        "third_party_count": len(installed_names) - 1,
        "known_vulnerability_count": len(vulnerabilities),
        "unscanned_third_party": sorted(skipped),
        "unlicensed_third_party": sorted(unlicensed),
    }


def verify_supply_chain(directory: Path, expected: dict[str, str]) -> dict:
    """Verify bundled reports against their own installation and release identity."""
    directory = directory.resolve()
    receipt = _load(directory / RECEIPT)
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise SupplyChainInvalid("supply-chain receipt is invalid")
    try:
        scan_time = datetime.fromisoformat(receipt["generated_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SupplyChainInvalid("supply-chain scan time is missing") from exc
    if scan_time.utcoffset() is None:
        raise SupplyChainInvalid("supply-chain scan time lacks a timezone")
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise SupplyChainInvalid(f"supply-chain receipt {key} differs")
    hashes = receipt.get("files")
    if not isinstance(hashes, dict) or set(hashes) != RAW_FILES:
        raise SupplyChainInvalid("supply-chain receipt file list is incomplete")
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SupplyChainInvalid("supply-chain receipt digest is invalid")
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
            raise SupplyChainInvalid(f"supply-chain evidence differs: {name}")
    _verify_output_privacy(directory)
    analysis = _analysis(directory)
    if (
        receipt.get("status") != "PASS"
        or receipt.get("audit_service") != "pypi"
        or receipt.get("audit_exit_code") != 0
        or receipt.get("database_snapshot_available") is not False
        or receipt.get("sbom_tool_version") != SBOM_TOOL_VERSION
        or receipt.get("audit_tool_version") != AUDIT_TOOL_VERSION
        or not isinstance(receipt.get("uv_version"), str)
        or not receipt["uv_version"].startswith("uv ")
        or any(receipt.get(key) != value for key, value in analysis.items())
        or analysis["known_vulnerability_count"] != 0
        or analysis["unscanned_third_party"]
        or analysis["unlicensed_third_party"]
    ):
        raise SupplyChainInvalid("supply-chain scan or license gate is not clear")
    return analysis


def _run(command: list[str], *, cwd: Path, environment: dict[str, str],
         timeout: int = 600, allow_audit_findings: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True,
        check=False, timeout=timeout,
    )
    if result.returncode and not (allow_audit_findings and result.returncode == 1):
        raise RuntimeError(f"supply-chain command failed with exit {result.returncode}: {command[0]}")
    return result


def _remove_private_locations(value: object) -> int:
    """Drop local and credential-bearing links from otherwise unchanged SBOM data."""
    removed = 0
    if isinstance(value, dict):
        references = value.get("externalReferences")
        if isinstance(references, list):
            safe = []
            for reference in references:
                url = reference.get("url") if isinstance(reference, dict) else None
                if not isinstance(url, str) or _private_url(url):
                    removed += 1
                else:
                    safe.append(reference)
            value["externalReferences"] = safe
        for nested in value.values():
            removed += _remove_private_locations(nested)
    elif isinstance(value, list):
        for nested in value:
            removed += _remove_private_locations(nested)
    return removed


def _generate_stage(root: Path, wheel: Path, output: Path) -> dict:
    """Install and scan a candidate without exposing intermediate files."""
    from trainguard.environment import source_sha256
    from trainguard.execution_inputs import execution_inputs_sha256

    root = root.resolve()
    wheel = wheel.resolve(strict=True)
    project_license_digest = verify_candidate_license(wheel, root / "LICENSE")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in RAW_FILES | {RECEIPT}):
        raise FileExistsError("supply-chain output already exists")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("VIRTUAL_ENV", None)
    environment["PYTHONNOUSERSITE"] = "1"
    export = _run(
        ["uv", "export", "--locked", "--no-dev", "--no-emit-project", "--format",
         "requirements.txt"], cwd=root, environment=environment,
    )
    (output / "supply-chain-requirements.txt").write_text(export.stdout, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="supply-chain-env-") as temporary:
        temporary_root = Path(temporary)
        runtime = temporary_root / "runtime"
        tools = temporary_root / "tools"
        for destination in (runtime, tools):
            _run(["uv", "venv", "--python", sys.executable, str(destination)],
                 cwd=root, environment=environment)
        runtime_python = runtime / "bin" / "python"
        tool_python = tools / "bin" / "python"
        _run(
            ["uv", "pip", "install", "--python", str(runtime_python), "--require-hashes",
             "-r", str(output / "supply-chain-requirements.txt")],
            cwd=root, environment=environment,
        )
        _run(["uv", "pip", "install", "--python", str(runtime_python), "--no-deps",
              str(wheel)], cwd=root, environment=environment)
        installed_source = _run(
            [str(runtime_python), "-c",
             "from trainguard.environment import source_sha256; print(source_sha256())"],
            cwd=root, environment=environment,
        ).stdout.strip()
        if installed_source != source_sha256():
            raise SupplyChainInvalid("installed candidate wheel differs from package source")
        _run(
            ["uv", "pip", "install", "--python", str(tool_python), "--require-hashes",
             "-r", str(root / "scripts/supply-chain-tools.txt")],
            cwd=root, environment=environment,
        )
        installed = _run(
            ["uv", "pip", "list", "--python", str(runtime_python), "--format", "json"],
            cwd=root, environment=environment,
        )
        installed_data = json.loads(installed.stdout)
        (output / "supply-chain-installed.json").write_text(
            json.dumps(installed_data, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        raw_sbom = temporary_root / "sbom-raw.json"
        _run(
            [str(tools / "bin" / "cyclonedx-py"), "environment", str(runtime_python),
             "--of", "JSON", "--sv", "1.6", "--output-reproducible",
             "--output-file", str(raw_sbom)],
            cwd=root, environment=environment,
        )
        bom = json.loads(raw_sbom.read_text(encoding="utf-8"))
        redactions = _remove_private_locations(bom)
        (output / "supply-chain-sbom.json").write_text(
            json.dumps(bom, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        license_rows = [
            {"name": item["name"], "version": item["version"],
             "licenses": item.get("licenses", [])}
            for item in bom["components"]
        ]
        (output / "supply-chain-licenses.json").write_text(
            json.dumps({"schema_version": 1, "source": "CycloneDX declared package metadata",
                        "components": license_rows}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        site_packages = _run(
            [str(runtime_python), "-c",
             "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            cwd=root, environment=environment,
        ).stdout.strip()
        audit_path = output / "supply-chain-audit.json"
        audit = _run(
            [str(tools / "bin" / "pip-audit"), "--path", site_packages,
             "--vulnerability-service", "pypi", "--format", "json", "--output",
             str(audit_path), "--cache-dir", str(temporary_root / "audit-cache"),
             "--progress-spinner", "off"],
            cwd=root, environment=environment, allow_audit_findings=True,
        )
        if not audit_path.is_file():
            raise SupplyChainInvalid("known-vulnerability scanner produced no JSON output")
        analysis = _analysis(output)
        if audit.returncode != (1 if analysis["known_vulnerability_count"] else 0):
            raise SupplyChainInvalid("known-vulnerability scanner exit differs from its JSON results")
        sbom_version = _run([str(tools / "bin" / "cyclonedx-py"), "--version"],
                            cwd=root, environment=environment).stdout.strip()
        audit_version = _run([str(tools / "bin" / "pip-audit"), "--version"],
                             cwd=root, environment=environment).stdout.strip()
        if sbom_version != SBOM_TOOL_VERSION or audit_version != f"pip-audit {AUDIT_TOOL_VERSION}":
            raise SupplyChainInvalid("supply-chain tool versions differ from their pinned inputs")
    status = "PASS" if (
        analysis["known_vulnerability_count"] == 0
        and not analysis["unscanned_third_party"]
        and not analysis["unlicensed_third_party"]
    ) else "BLOCKED"
    uv_version = _run(["uv", "--version"], cwd=root, environment=environment).stdout.strip()
    receipt = {
        "schema_version": 1,
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_sha256(),
        "execution_inputs_sha256": execution_inputs_sha256(root),
        "wheel_sha256": _sha256(wheel),
        "lock_sha256": _sha256(root / "uv.lock"),
        "tool_lock_sha256": _sha256(root / "scripts/supply-chain-tools.txt"),
        "security_policy_sha256": _sha256(root / "SECURITY.md"),
        "first_party_license_sha256": project_license_digest,
        "security_channel_record_sha256": _sha256(
            root / "docs/commercial/security-channel-2026-09-27.json"
        ),
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "audit_service": "pypi",
        "audit_exit_code": audit.returncode,
        "audit_tool_version": AUDIT_TOOL_VERSION,
        "uv_version": uv_version,
        "sbom_tool_version": SBOM_TOOL_VERSION,
        "sbom_private_references_removed": redactions,
        "database_snapshot_available": False,
        **analysis,
        "files": {name: _sha256(output / name) for name in sorted(RAW_FILES)},
    }
    (output / RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n",
                                    encoding="utf-8")
    if status == "PASS":
        verify_supply_chain(output, {
            key: receipt[key] for key in (
                "source_sha256", "execution_inputs_sha256", "wheel_sha256", "lock_sha256",
                "tool_lock_sha256", "security_policy_sha256",
                "security_channel_record_sha256", "first_party_license_sha256",
                "candidate_version",
            )
        })
    return receipt


def _publish_scan_output(stage: Path, output: Path) -> None:
    """Publish a complete privacy-checked scan; mark it only after every file moves."""
    names = RAW_FILES | {RECEIPT}
    if any((stage / name).is_symlink() or not (stage / name).is_file() for name in names):
        raise SupplyChainInvalid("staged supply-chain evidence is incomplete or linked")
    receipt = _load(stage / RECEIPT)
    hashes = receipt.get("files") if isinstance(receipt, dict) else None
    if not isinstance(hashes, dict) or set(hashes) != RAW_FILES or any(
        _sha256(stage / name) != hashes[name] for name in RAW_FILES
    ):
        raise SupplyChainInvalid("staged supply-chain evidence differs from its receipt")
    _verify_output_privacy(stage)
    if output.is_symlink():
        raise SupplyChainInvalid("supply-chain output directory is linked")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() or (output / name).is_symlink()
           for name in names | {PRIVACY_MARKER}):
        raise FileExistsError("supply-chain output already exists")
    published = []
    try:
        for name in sorted(names):
            os.replace(stage / name, output / name)
            published.append(name)
        (output / PRIVACY_MARKER).write_text("privacy-checked\n", encoding="utf-8")
    except BaseException:
        for name in published:
            (output / name).unlink(missing_ok=True)
        raise


def generate(root: Path, wheel: Path, output: Path) -> dict:
    """Stage the scan outside the published result and release only safe files."""
    output = output.absolute()
    if output.is_symlink():
        raise SupplyChainInvalid("supply-chain output directory is linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".supply-chain-stage-", dir=output.parent) as temporary:
        stage = Path(temporary)
        receipt = _generate_stage(root, wheel, stage)
        _publish_scan_output(stage, output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(Path(__file__).resolve().parents[1], args.wheel, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "component_count": result["component_count"],
        "third_party_count": result["third_party_count"],
        "known_vulnerability_count": result["known_vulnerability_count"],
        "unscanned_third_party": result["unscanned_third_party"],
        "unlicensed_third_party": result["unlicensed_third_party"],
        "wheel_sha256": result["wheel_sha256"],
    }, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
