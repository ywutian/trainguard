"""Fail closed unless every commercial release gate has source-bound evidence."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from trainguard.config import ProjectConfig, load_config
from trainguard.events import write_json_atomic
from trainguard.evidence_lineage import require_evidence_only_descendant
from trainguard.execution_inputs import execution_inputs_sha256

REQUIRED_GATES = {
    "local_package", "local_cpu", "customer_workload", "persistent_checkpoint",
    "cross_host_fencing", "real_gpu_matrix", "security_operations",
    "commercial_contract", "paid_pilot", "supported_matrix", "sustained_operations",
}
HOSTED_REPOSITORY = "ywutian/trainguard"
LOCAL_RAW_FILES = {
    "result.json", "pytest.xml", "acceptance.json", "static.txt", "tests.txt",
    "cpu-acceptance.txt", "package.txt", "wheel.txt", "fresh-install.txt",
    "upgrade-boundary.txt", "supply-chain.txt", "supply-chain-sbom.json",
    "supply-chain-licenses.json", "supply-chain-audit.json",
    "supply-chain-installed.json", "supply-chain-requirements.txt",
    "supply-chain-receipt.json",
}
LOCAL_CPU_CASES = (
    ("sync-worker_exit", "sync", "worker_exit", "none", True),
    ("async-worker_exit", "async", "worker_exit", "none", True),
    ("sync-save_interrupt", "sync", "save_interrupt", "none", True),
    ("async-save_interrupt", "async", "save_interrupt", "none", True),
    ("sync-corrupt", "sync", "corrupt", "none", True),
    ("async-corrupt", "async", "corrupt", "none", True),
    ("sync-hang", "sync", "hang", "none", True),
    ("omit-rng", "sync", "worker_exit", "rng", False),
    ("omit-optimizer", "sync", "worker_exit", "optimizer", False),
    ("omit-cursor", "sync", "worker_exit", "cursor", False),
)


def _supply_module():
    script = Path(__file__).resolve().parent / "supply_chain.py"
    spec = importlib.util.spec_from_file_location("release_supply_chain", script)
    if spec is None or spec.loader is None:
        raise ValueError("supply-chain verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_supply_chain(raw_dir: Path, expected: dict[str, str]) -> None:
    _supply_module().verify_supply_chain(raw_dir, expected)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _private_reporting_enabled(root: Path) -> bool:
    checked = subprocess.run(
        ["gh", "api", f"repos/{HOSTED_REPOSITORY}/private-vulnerability-reporting"],
        cwd=root, capture_output=True, text=True, check=False, timeout=30,
    )
    if checked.returncode:
        raise ValueError("private vulnerability reporting setting could not be verified")
    try:
        enabled = json.loads(checked.stdout)["enabled"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("private vulnerability reporting setting is invalid") from exc
    return enabled is True


def package_source_sha256(root: Path) -> str:
    source = root / "src" / "trainguard"
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        name = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _previous_release(root: Path, pin: object, current_version: str) -> dict:
    """Bind the upgrade rehearsal to one reviewed, ancestral release commit."""
    if not isinstance(pin, dict) or set(pin) != {
        "git_commit", "version", "wheel_sha256", "lock_sha256"
    }:
        raise ValueError("approved previous release identity is incomplete")
    commit = pin["git_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(pin["version"], str)
        or not _is_sha256(pin["wheel_sha256"])
        or not _is_sha256(pin["lock_sha256"])
    ):
        raise ValueError("approved previous release identity is malformed")
    try:
        old_parts = tuple(int(part) for part in pin["version"].split("."))
        new_parts = tuple(int(part) for part in current_version.split("."))
    except ValueError as exc:
        raise ValueError("approved previous release version is invalid") from exc
    if len(old_parts) != 3 or len(new_parts) != 3 or (
        old_parts[:2] != new_parts[:2] or old_parts[2] + 1 != new_parts[2]
    ):
        raise ValueError("approved previous release is not the preceding patch version")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root, capture_output=True, check=False,
    )
    if ancestor.returncode:
        raise ValueError("approved previous release commit is not in candidate history")
    metadata = subprocess.run(
        ["git", "show", f"{commit}:pyproject.toml"],
        cwd=root, capture_output=True, check=False,
    )
    locked = subprocess.run(
        ["git", "show", f"{commit}:uv.lock"],
        cwd=root, capture_output=True, check=False,
    )
    if metadata.returncode or locked.returncode:
        raise ValueError("approved previous release files are unavailable")
    try:
        recorded_version = tomllib.loads(metadata.stdout.decode("utf-8"))["project"]["version"]
    except (UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("approved previous release metadata is invalid") from exc
    if recorded_version != pin["version"] or hashlib.sha256(locked.stdout).hexdigest() != pin[
        "lock_sha256"
    ]:
        raise ValueError("approved previous release does not match its commit")
    return pin


def _archive_source_sha256(items: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, content in sorted(items.items()):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


GPU_TESTS = {
    "test_actual_cuda_recovery_campaign[ddp-fp32]",
    "test_actual_cuda_recovery_campaign[ddp-bf16]",
    "test_actual_cuda_recovery_campaign[ddp-fp16]",
    "test_actual_cuda_recovery_campaign[fsdp2-fp32]",
}
DEVICE_PREFLIGHT_SKIP = (
    "tests.test_device_upgrade",
    "test_cuda_configuration_is_accepted_but_missing_devices_fail_before_launch",
)

# These cases guard the behaviors most likely to produce a misleading green gate
# if the test suite is narrowed or a critical test is replaced by a new one.
REQUIRED_TEST_IDENTITIES = {
    ("tests.test_campaign", "test_cpu_campaign_closes_recovery_matrix"),
    ("tests.test_capacity", "test_guarded_completion_rejects_only_one_verified_candidate"),
    ("tests.test_capacity", "test_guarded_audit_counts_uncommitted_candidate_bytes"),
    ("tests.test_capacity", "test_guarded_checkpoint_limit_includes_manifest_and_commit_marker"),
    ("tests.test_completion_upgrade", "test_malformed_rank_digest_invalidates_previous_success"),
    ("tests.test_completion_upgrade", "test_malformed_attempt_identity_invalidates_previous_success"),
    ("tests.test_completion_upgrade", "test_succeeded_run_rejects_mismatched_saved_status_identity[attempt_id]"),
    ("tests.test_completion_upgrade", "test_succeeded_run_rejects_mismatched_saved_status_identity[config]"),
    ("tests.test_completion_upgrade", "test_completed_run_cross_media_tampering_fails_closed"),
    ("tests.test_completion_upgrade", "test_resume_requires_current_platform_and_storage_identity[platform]"),
    ("tests.test_completion_upgrade", "test_resume_requires_current_platform_and_storage_identity[storage_device]"),
    ("tests.test_completion_upgrade", "test_resume_requires_current_platform_and_storage_identity[cuda_available]"),
    ("tests.test_checkpoint_crash_matrix",
     "test_process_exit_at_checkpoint_publication_boundary[after_commit_before_index]"),
    ("tests.test_external_workload", "test_external_two_rank_recovery_and_omitted_state_controls"),
    ("tests.test_external_workload", "test_preflight_freezes_the_verified_bytes_before_source_changes"),
    ("tests.test_local_reference_store", "test_local_reference_two_processes_have_one_head_cas_winner"),
    ("tests.test_local_topology_simulation",
     "test_two_local_launch_agents_recover_exactly_after_worker_exit"),
    ("tests.test_privacy", "test_guarded_missing_wrong_key_and_sample_tampering_fail_closed"),
    ("tests.test_privacy", "test_guarded_recovery_matches_uninterrupted_reference"),
    ("tests.test_privacy", "test_guarded_signed_duplicate_synthetic_samples_fail_completion_audit"),
    ("tests.test_privacy", "test_guarded_unicode_mac_invalidates_previous_success"),
    ("tests.test_privacy", "test_guarded_accumulated_synthetic_samples_validate"),
    ("tests.test_recovery_integration",
     "test_selected_checkpoint_mutation_before_worker_load_fails_closed"),
    ("tests.test_recovery_integration", "test_validator_detects_omitted_recovery_state"),
    ("tests.test_remote_dcp_roundtrip",
     "test_real_dcp_bytes_publish_fallback_and_restore_through_remote_model"),
    ("tests.test_remote_protocol", "test_in_flight_old_head_write_loses_to_epoch_barrier"),
    ("tests.test_remote_protocol", "test_in_flight_old_payload_is_rejected_and_remains_unpublished"),
    ("tests.test_remote_training_restore",
     "test_two_rank_training_recovers_from_remote_model_after_local_loss"),
    ("tests.test_sdist_members",
     "test_source_distribution_excludes_generated_output_and_rejects_injection"),
    ("tests.test_delivery_bundle_verify",
     "test_transferred_bundle_rejects_self_consistent_file_list_with_stale_artifact"),
    ("tests.test_release_gate", "test_local_cpu_receipt_rejects_self_consistent_case_tampering"),
    ("tests.test_validation", "test_failed_restore_before_training_can_retry"),
    ("tests.test_scaler_boundary", "test_nonfinite_gradient_does_not_advance_optimizer_or_scheduler"),
}


def _collected_test_identities(root: Path) -> set[tuple[str, str]]:
    """Collect the complete checkout suite independently of the saved JUnit file."""
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--disable-warnings",
         "-o", "addopts=", "tests"],
        cwd=root, env=environment, capture_output=True, text=True, check=False, timeout=90,
    )
    if completed.returncode:
        raise ValueError("current test suite cannot be collected completely")
    identities: list[tuple[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.startswith("tests/") or "::" not in line:
            continue
        parts = line.split("::")
        path = Path(parts[0])
        if path.suffix != ".py" or not parts[-1]:
            raise ValueError("current test collection contains an invalid identity")
        classname = ".".join((*path.with_suffix("").parts, *parts[1:-1]))
        identities.append((classname, parts[-1]))
    collected = set(identities)
    if len(collected) != len(identities) or not REQUIRED_TEST_IDENTITIES <= collected:
        raise ValueError("current test collection lacks required recovery controls")
    return collected


def _validate_pytest_result(raw_dir: Path, details: dict) -> None:
    root = ElementTree.parse(raw_dir / "pytest.xml").getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise ValueError("local raw test suite is missing")
    cases = [case for suite in suites for case in suite.findall("testcase")]
    identities = [(case.get("classname"), case.get("name")) for case in cases]
    collected = _collected_test_identities(Path(__file__).resolve().parents[1])
    declared = sum(int(suite.get("tests", -1)) for suite in suites)
    declared_failures = sum(int(suite.get("failures", -1)) for suite in suites)
    declared_errors = sum(int(suite.get("errors", -1)) for suite in suites)
    declared_skips = sum(int(suite.get("skipped", -1)) for suite in suites)
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = [case for case in cases if case.find("skipped") is not None]
    gpu_skipped = [case for case in skipped if case.get("classname") == "tests.test_gpu_acceptance"]
    device_skipped = [case for case in skipped if (
        case.get("classname"), case.get("name")
    ) == DEVICE_PREFLIGHT_SKIP]
    recorded_tests = details.get("tests")
    expected_tests = {
        "passed": len(cases) - len(skipped), "skipped_no_gpu": len(gpu_skipped)
    }
    if isinstance(recorded_tests, dict) and "skipped_device_preflight" in recorded_tests:
        expected_tests["skipped_device_preflight"] = len(device_skipped)
    if (
        declared != len(cases)
        or any(not all(isinstance(part, str) and part for part in identity)
               for identity in identities)
        or len(set(identities)) != len(identities)
        or set(identities) != collected
        or declared_failures != failures
        or declared_errors != errors
        or declared_skips != len(skipped)
        or failures
        or errors
        or not isinstance(recorded_tests, dict)
        or len(skipped) != len(gpu_skipped) + len(device_skipped)
        or len(gpu_skipped) > len(GPU_TESTS)
        or bool(gpu_skipped and device_skipped)
        or {case.get("name") for case in gpu_skipped} - GPU_TESTS
        or any(case.find("skipped").get("message") != "requires two actual CUDA devices"
               for case in gpu_skipped)
        or any(case.find("skipped").get("message") !=
               "this test verifies the unavailable-device preflight"
               for case in device_skipped)
        or (device_skipped and "skipped_device_preflight" not in recorded_tests)
        or len(cases) - len(skipped) < 156
        or recorded_tests != expected_tests
    ):
        raise ValueError("local raw test suite is incomplete, failing, or over-skipped")


def _expected_local_cpu_config(root: Path) -> dict:
    base = load_config(root / "configs/cpu_demo.yaml")
    raw = base.model_dump()
    raw["model"]["dropout"] = max(raw["model"]["dropout"], 0.2)
    return ProjectConfig.model_validate(raw).model_dump()


def _local_cpu_config_sha256(root: Path) -> str:
    content = json.dumps(_expected_local_cpu_config(root), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _local_case_difference_hashes_complete(case: dict) -> bool:
    validation = case.get("validation")
    hashes = validation.get("difference_sha256") if isinstance(validation, dict) else None
    if not isinstance(hashes, list) or any(not _is_sha256(value) for value in hashes):
        return False
    observed = set(hashes)
    if len(observed) != len(hashes):
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
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    return {digest(value) for value in required} <= observed <= {
        digest(value) for value in allowed
    }


def _local_cpu_acceptance_complete(acceptance: object) -> bool:
    """Independently verify each campaign case before local evidence passes."""
    if not isinstance(acceptance, dict) or acceptance.get("status") != "SUCCEEDED" or (
        acceptance.get("reference_status") != "VALIDATED"
    ):
        return False
    cases = acceptance.get("cases")
    if not isinstance(cases, list) or len(cases) != len(LOCAL_CPU_CASES):
        return False
    expected = {
        name: (mode, fault, omitted, exact)
        for name, mode, fault, omitted, exact in LOCAL_CPU_CASES
    }
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str):
            return False
        name = case["name"]
        if name not in expected or name in seen:
            return False
        seen.add(name)
        mode, fault, omitted, exact = expected[name]
        if (
            case.get("name"), case.get("mode"), case.get("fault"), case.get("omit_state")
        ) != (name, mode, fault, omitted) or (
            case.get("expected_exact") is not exact
        ) or (
            case.get("status") != "PASSED"
            or type(case.get("recovery_count")) is not int
            or case["recovery_count"] != 1
            or case.get("fault_attributed") is not True
            or not isinstance(case.get("validation"), dict)
            or case["validation"].get("passed") is not exact
            or not _local_case_difference_hashes_complete(case)
        ):
            return False
    return seen == set(expected)


def _verify_artifacts(root: Path, wheel: Path, sdist: Path, source_digest: str) -> None:
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    if (
        not wheel.name.startswith(f"trainguard-{version}-")
        or not wheel.name.endswith(".whl")
        or sdist.name != f"trainguard-{version}.tar.gz"
        or not zipfile.is_zipfile(wheel)
        or not tarfile.is_tarfile(sdist)
    ):
        raise ValueError("release artifacts have invalid names or archive formats")
    from verify_sdist import verify_sdist

    verify_sdist(root, sdist, version)
    _supply_module().verify_candidate_license(wheel, root / "LICENSE")
    with zipfile.ZipFile(wheel) as archive:
        files = archive.namelist()
        if len(files) != len(set(files)):
            raise ValueError("wheel contains duplicate members")
        code = {
            name.removeprefix("trainguard/"): archive.read(name)
            for name in files if name.startswith("trainguard/") and name.endswith(".py")
        }
        metadata = f"trainguard-{version}.dist-info/METADATA"
        template = "trainguard/templates/cpu_demo.yaml"
        if metadata not in files or template not in files:
            raise ValueError("wheel metadata or packaged template is missing")
        details = archive.read(metadata).decode("utf-8")
        if "Name: trainguard\n" not in details or f"Version: {version}\n" not in details:
            raise ValueError("wheel package metadata differs")
        if archive.read(template) != (root / "src" / template).read_bytes():
            raise ValueError("wheel template differs from source")
        entry_points = f"trainguard-{version}.dist-info/entry_points.txt"
        record_name = f"trainguard-{version}.dist-info/RECORD"
        if entry_points not in files or record_name not in files:
            raise ValueError("wheel entry point or record is missing")
        if archive.read(entry_points).decode("utf-8").strip() != (
            "[console_scripts]\ntrainguard = trainguard.cli:app"
        ):
            raise ValueError("wheel command entry point differs")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        if len(rows) != len(files) or {row[0] for row in rows} != set(files):
            raise ValueError("wheel record does not cover all members")
        for row in rows:
            if len(row) != 3:
                raise ValueError("wheel record row is malformed")
            name, encoded, size = row
            if name == record_name:
                if encoded or size:
                    raise ValueError("wheel self record is malformed")
                continue
            content = archive.read(name)
            expected = "sha256=" + base64.urlsafe_b64encode(
                hashlib.sha256(content).digest()
            ).rstrip(b"=").decode("ascii")
            if encoded != expected or size != str(len(content)):
                raise ValueError("wheel record digest differs")
    if _archive_source_sha256(code) != source_digest:
        raise ValueError("wheel package code differs from source")
    prefix = f"trainguard-{version}/"
    with tarfile.open(sdist, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("source distribution contains duplicate members")
        code = {
            member.name.removeprefix(prefix + "src/trainguard/"): archive.extractfile(member).read()
            for member in members
            if member.isfile() and member.name.startswith(prefix + "src/trainguard/")
            and member.name.endswith(".py")
        }
        lock = archive.extractfile(prefix + "uv.lock")
        if lock is None or lock.read() != (root / "uv.lock").read_bytes():
            raise ValueError("source distribution dependency lock differs")
    if _archive_source_sha256(code) != source_digest:
        raise ValueError("source distribution package code differs from source")
    with tempfile.TemporaryDirectory(prefix="release-rebuild-") as temporary:
        result = subprocess.run(
            ["uv", "build", "--wheel", "--sdist", "--build-constraints",
             "build-constraints.txt", "--require-hashes", "--out-dir", temporary],
            cwd=root, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise ValueError("candidate source cannot be rebuilt")
        for submitted in (wheel, sdist):
            rebuilt = Path(temporary) / submitted.name
            if not rebuilt.is_file() or _digest(submitted) != _digest(rebuilt):
                raise ValueError(f"{submitted.name}: complete artifact differs from current source build")


def _local_evidence(root: Path, details: dict, source_digest: str,
                    artifacts: dict[str, str], previous_release: dict) -> None:
    location = details.get("raw_evidence_dir")
    hashes = details.get("raw_files")
    if not isinstance(location, str) or not isinstance(hashes, dict) or set(hashes) != LOCAL_RAW_FILES:
        raise ValueError("local evidence does not contain the required raw result files")
    raw_dir = (root / location).resolve()
    if not raw_dir.is_relative_to(root) or not raw_dir.is_dir():
        raise ValueError("local raw evidence directory is invalid")
    for name in LOCAL_RAW_FILES:
        path = raw_dir / name
        if path.is_symlink() or not path.is_file() or _digest(path) != hashes[name]:
            raise ValueError(f"local raw evidence is missing or changed: {name}")
    result = json.loads((raw_dir / "result.json").read_text(encoding="utf-8"))
    input_digest = execution_inputs_sha256(root)
    if details.get("execution_inputs_sha256") != input_digest:
        raise ValueError("local evidence was produced with different execution inputs")
    execution_commit = result.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or result.get("execution_commit_after") != execution_commit
        or details.get("execution_commit") != execution_commit
    ):
        raise ValueError("local evidence has no stable execution commit")
    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=True,
    ).stdout.strip()
    require_evidence_only_descendant(root, execution_commit, candidate_commit)
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    expected_gates = {
        "static", "tests", "cpu-acceptance", "package", "wheel", "fresh-install",
        "upgrade-boundary", "supply-chain",
    }
    gates = result.get("gates")
    if (
        result.get("status") != "SUCCEEDED"
        or result.get("source_sha256") != source_digest
        or result.get("execution_inputs_sha256") != input_digest
        or result.get("execution_inputs_after_sha256") != input_digest
        or result.get("version") != version
        or not isinstance(result.get("artifact_sha256"), dict)
        or {name: digest for name, digest in result["artifact_sha256"].items()
            if name.endswith((".whl", ".tar.gz"))} != {
            name: digest for name, digest in artifacts.items() if name != "uv.lock"
        }
        or result.get("lock_sha256") != artifacts["uv.lock"]
        or not isinstance(gates, list)
        or len(gates) != len(expected_gates)
        or {gate.get("name") for gate in gates if isinstance(gate, dict)} != expected_gates
        or any(
            gate.get("exit_code") != 0
            or gate.get("timed_out") is not False
            or gate.get("execution_inputs_before_sha256") != input_digest
            or gate.get("execution_inputs_after_sha256") != input_digest
            or Path(gate.get("output", "")).name != f"{gate['name']}.txt"
            for gate in gates
        )
    ):
        raise ValueError("local raw result does not prove all required gates")
    _validate_pytest_result(raw_dir, details)
    acceptance = json.loads((raw_dir / "acceptance.json").read_text(encoding="utf-8"))
    environment = acceptance.get("environment") if isinstance(acceptance, dict) else None
    expected_config = _expected_local_cpu_config(root)
    platform_digest = result.get("platform_sha256")
    if (
        not _local_cpu_acceptance_complete(acceptance)
        or result.get("acceptance") != acceptance
        or acceptance.get("config_sha256") != _local_cpu_config_sha256(root)
        or acceptance.get("run") != {
            "device": "cpu", "backend": "gloo", "world_size": 2,
            "total_steps": expected_config["training"]["total_steps"],
        }
        or not isinstance(environment, dict)
        or not _is_sha256(platform_digest)
        or environment != {
            "git_commit": execution_commit, "source_sha256": source_digest,
            "python": result.get("python"), "torch": result.get("torch"),
            "platform_sha256": platform_digest,
            "world_size": 2, "device": "cpu", "storage": "local filesystem",
        }
    ):
        raise ValueError("local raw CPU acceptance matrix is incomplete")
    wheel_name = next(name for name in artifacts if name.endswith(".whl"))
    _verify_supply_chain(raw_dir, {
        "candidate_version": version,
        "source_sha256": source_digest,
        "execution_inputs_sha256": input_digest,
        "wheel_sha256": artifacts[wheel_name],
        "lock_sha256": artifacts["uv.lock"],
        "tool_lock_sha256": _digest(root / "scripts/supply-chain-tools.txt"),
        "security_policy_sha256": _digest(root / "SECURITY.md"),
        "first_party_license_sha256": _digest(root / "LICENSE"),
        "security_channel_record_sha256": _digest(
            root / "docs/commercial/security-channel-2026-09-27.json"
        ),
    })
    supply_result = json.loads((raw_dir / "supply-chain.txt").read_text(encoding="utf-8").splitlines()[-1])
    wheel_result = json.loads((raw_dir / "wheel.txt").read_text(encoding="utf-8").splitlines()[-1])
    install = json.loads((raw_dir / "fresh-install.txt").read_text(encoding="utf-8").splitlines()[-1])
    upgrade = json.loads((raw_dir / "upgrade-boundary.txt").read_text(encoding="utf-8").splitlines()[-1])
    if (
        supply_result.get("status") != "PASS"
        or supply_result.get("wheel_sha256") != artifacts[wheel_name]
        or wheel_result.get("passed") is not True
        or wheel_result.get("version") != version
        or wheel_result.get("source_sha256") != source_digest
        or install.get("version") != version
        or install.get("wheel_sha256") != artifacts[wheel_name]
        or install.get("installed_outside_checkout") is not True
        or install.get("completed_run") is not True
        or install.get("recovered_run_matches_reference") is not True
        or install.get("support_export_checked") is not True
        or install.get("run_data_preserved_after_uninstall") is not True
        or install.get("run_data_files_checked", 0) < 1
        or install.get("attributed_faults") != 1
        or install.get("recoveries") != 1
        or upgrade.get("current_version") != version
        or upgrade.get("current_wheel_sha256") != artifacts[wheel_name]
        or upgrade.get("current_lock_sha256") != artifacts["uv.lock"]
        or upgrade.get("previous_commit_sha") != previous_release["git_commit"]
        or upgrade.get("previous_version") != previous_release["version"]
        or upgrade.get("previous_wheel_sha256") != previous_release["wheel_sha256"]
        or upgrade.get("previous_lock_sha256") != previous_release["lock_sha256"]
        or upgrade.get("new_version_rejected_interrupted_old_run") is not True
        or upgrade.get("old_locked_environment_resumed_exactly") is not True
        or upgrade.get("old_run_files_unchanged_after_rejection", 0) < 1
    ):
        raise ValueError("local install or upgrade raw evidence is incomplete")


def _receipt(path: Path, gate_id: str, source_digest: str, artifacts: dict[str, str],
             previous_release: dict) -> dict:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{gate_id}: evidence receipt is unreadable") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != 1
        or receipt.get("gate_id") != gate_id
        or receipt.get("decision") != "PASS"
        or receipt.get("candidate_source_sha256") != source_digest
        or receipt.get("artifacts") != artifacts
        or not isinstance(receipt.get("scope"), str)
        or not receipt["scope"]
        or not isinstance(receipt.get("reviewer_role"), str)
        or not receipt["reviewer_role"]
        or not isinstance(receipt.get("approval_reference"), str)
        or not receipt["approval_reference"]
        or not isinstance(receipt.get("record_reference"), str)
        or not receipt["record_reference"]
        or not isinstance(receipt.get("record_sha256"), str)
        or len(receipt["record_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in receipt["record_sha256"])
        or not isinstance(receipt.get("reviewed_at"), str)
    ):
        raise ValueError(f"{gate_id}: evidence receipt lacks version, scope or approval binding")
    try:
        datetime.fromisoformat(receipt["reviewed_at"])
    except ValueError as exc:
        raise ValueError(f"{gate_id}: evidence review time is invalid") from exc
    if gate_id in {"local_package", "local_cpu"}:
        if (
            not isinstance(receipt.get("execution_commit"), str)
            or receipt.get("execution_inputs_sha256") != execution_inputs_sha256(
                Path(__file__).resolve().parents[1]
            )
        ):
            raise ValueError(f"{gate_id}: local execution identity differs")
        checks = receipt.get("checks")
        required = (
            {"wheel_identity", "fresh_install", "upgrade_recovery", "supply_chain"}
            if gate_id == "local_package" else {"full_suite", "cpu_fault_matrix"}
        )
        if not isinstance(checks, dict) or not all(checks.get(name) is True for name in required):
            raise ValueError(f"{gate_id}: local gate checks are incomplete")
        root = Path(__file__).resolve().parents[1]
        record = (root / receipt["record_reference"]).resolve()
        if not record.is_relative_to(root) or not record.is_file():
            raise ValueError(f"{gate_id}: local evidence record is missing")
        if _digest(record) != receipt["record_sha256"]:
            raise ValueError(f"{gate_id}: local evidence record digest differs")
        details = json.loads(record.read_text(encoding="utf-8"))
        if not isinstance(details, dict) or not isinstance(details.get("checks"), dict):
            raise ValueError(f"{gate_id}: local evidence record has an invalid shape")
        if (
            details.get("schema_version") != 1
            or details.get("status") != "SUCCEEDED"
            or details.get("candidate_source_sha256") != source_digest
            or details.get("execution_inputs_sha256") != receipt["execution_inputs_sha256"]
            or details.get("execution_commit") != receipt["execution_commit"]
            or details.get("artifacts") != artifacts
            or details["checks"].get("full_suite") is not True
            or details["checks"].get("cpu_fault_matrix") is not True
            or details["checks"].get("wheel_identity") is not True
            or details["checks"].get("fresh_install") is not True
            or details["checks"].get("upgrade_recovery") is not True
            or details["checks"].get("supply_chain") is not True
        ):
            raise ValueError(f"{gate_id}: local evidence record does not prove its checks")
        _local_evidence(root, details, source_digest, artifacts, previous_release)
    return receipt


def _hosted_linux_from_run(
    root: Path, run_id: int, wheel: Path, sdist: Path, *, commit: str,
    version: str, source_digest: str, input_digest: str, artifacts: dict[str, str],
) -> dict:
    """Fetch one workflow directly and recheck its downloaded raw evidence."""
    if type(run_id) is not int or run_id < 1:
        raise ValueError("hosted Linux workflow run ID is invalid")
    authenticated = subprocess.run(
        ["gh", "auth", "status", "--active", "--hostname", "github.com"],
        cwd=root, capture_output=True, text=True, check=False, timeout=30,
    )
    if authenticated.returncode:
        raise ValueError("hosted workflow account is not authenticated")
    with tempfile.TemporaryDirectory(prefix="hosted-linux-evidence-") as temporary:
        work = Path(temporary)
        metadata = work / "workflow.json"
        viewed = subprocess.run(
            ["gh", "run", "view", str(run_id), "--json",
             "databaseId,headSha,conclusion,event,jobs,workflowName",
             "--repo", HOSTED_REPOSITORY],
            cwd=root, capture_output=True, text=True, check=False, timeout=120,
        )
        if viewed.returncode:
            raise ValueError("hosted Linux workflow metadata could not be fetched")
        metadata.write_text(viewed.stdout, encoding="utf-8")
        for lane, label in (("3.11", "python311"), ("3.12", "python312")):
            for kind, artifact_name in (
                ("summary", f"recovery-summary-{lane}"),
                ("packages", f"verified-packages-{lane}"),
            ):
                destination = work / f"{label}-{kind}"
                fetched = subprocess.run(
                    ["gh", "run", "download", str(run_id), "--name", artifact_name,
                     "--dir", str(destination), "--repo", HOSTED_REPOSITORY],
                    cwd=root, capture_output=True, text=True, check=False, timeout=300,
                )
                if fetched.returncode:
                    raise ValueError(f"hosted Linux {artifact_name} could not be fetched")
        report_path = work / "verified.json"
        checked = subprocess.run(
            [sys.executable, str(root / "scripts/check_hosted_linux_evidence.py"),
             "--workflow-metadata", str(metadata),
             "--python311-summary", str(work / "python311-summary"),
             "--python311-packages", str(work / "python311-packages"),
             "--python312-summary", str(work / "python312-summary"),
             "--python312-packages", str(work / "python312-packages"),
             "--wheel", str(wheel.resolve()), "--sdist", str(sdist.resolve()),
             "--report", str(report_path)],
            cwd=root, capture_output=True, text=True, check=False, timeout=300,
        )
        if checked.returncode or not report_path.is_file():
            raise ValueError("hosted Linux raw evidence did not pass current-candidate checks")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "HOSTED_LINUX_EVIDENCE_CONSISTENT"
            or report.get("workflow_run_id") != run_id
            or report.get("candidate_git_commit") != commit
            or report.get("candidate_version") != version
            or report.get("candidate_source_sha256") != source_digest
            or report.get("candidate_execution_inputs_sha256") != input_digest
            or report.get("lock_sha256") != artifacts["uv.lock"]
            or report.get("artifacts") != {
                name: digest for name, digest in artifacts.items() if name != "uv.lock"
            }
            or set(report.get("matrix", {})) != {"3.11", "3.12"}
            or report.get("customer_environment_validated") is not False
            or report.get("production_release_authorized") is not False
        ):
            raise ValueError("hosted Linux evidence differs from the candidate identity")
        report["workflow_and_artifacts_fetched_live"] = True
        report["workflow_repository"] = HOSTED_REPOSITORY
        report["cryptographic_signature_verified"] = False
        return report


def evaluate(manifest: Path, wheel: Path, sdist: Path, hosted_run_id: int | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("release gate manifest schema is invalid")
    gates = data.get("gates")
    if (
        not isinstance(gates, list)
        or any(not isinstance(gate, dict) for gate in gates)
        or {gate.get("id") for gate in gates} != REQUIRED_GATES
    ):
        raise ValueError("release gate list is incomplete or duplicated")
    if len(gates) != len(REQUIRED_GATES):
        raise ValueError("release gate list contains duplicates")
    source_digest = package_source_sha256(root)
    input_digest = execution_inputs_sha256(root)
    if data.get("candidate_source_sha256") != source_digest:
        raise ValueError("release gate manifest does not match current package source")
    if data.get("candidate_execution_inputs_sha256") != input_digest:
        raise ValueError("release gate manifest does not match current execution inputs")
    current_version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    previous_release = _previous_release(root, data.get("previous_release"), current_version)
    for path in (wheel, sdist, root / "uv.lock"):
        if not path.is_file():
            raise ValueError("required release artifact is missing")
    _verify_artifacts(root, wheel, sdist, source_digest)
    artifacts = {wheel.name: _digest(wheel), sdist.name: _digest(sdist),
                 "uv.lock": _digest(root / "uv.lock")}

    checked = []
    local_execution_commits = set()
    for gate in gates:
        state = gate.get("status")
        if state not in {"PASS", "FAIL", "BLOCKED"}:
            raise ValueError("release gate has an invalid status")
        record = {"id": gate["id"], "status": state}
        if state == "PASS":
            location = gate.get("evidence")
            if not isinstance(location, str) or not location:
                raise ValueError(f"{gate['id']}: passing gate lacks evidence")
            raw_path = root / location
            path = raw_path.resolve()
            if not path.is_relative_to(root) or not path.is_file() or raw_path.is_symlink():
                raise ValueError(f"{gate['id']}: evidence path is invalid")
            digest = _digest(path)
            if digest != gate.get("sha256"):
                raise ValueError(f"{gate['id']}: evidence digest differs")
            receipt = _receipt(path, gate["id"], source_digest, artifacts, previous_release)
            if gate["id"] in {"local_package", "local_cpu"}:
                local_execution_commits.add(receipt["execution_commit"])
            record.update(evidence=location, sha256=digest)
        elif not isinstance(gate.get("reason"), str) or not gate["reason"]:
            raise ValueError(f"{gate['id']}: nonpassing gate lacks a reason")
        else:
            record["reason"] = gate["reason"]
        checked.append(record)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip())
    if len(local_execution_commits) > 1:
        raise ValueError("local gate receipts use different execution commits")
    receipts_complete = all(gate["status"] == "PASS" for gate in checked) and not dirty
    local_experiment_allowed = (
        not dirty
        and all(gate["status"] == "PASS" for gate in checked
                if gate["id"] in {"local_package", "local_cpu"})
        and not any(gate["status"] == "FAIL" for gate in checked)
    )
    hosted_evidence = None
    hosted_reason = "hosted Linux workflow run was not supplied"
    if not local_experiment_allowed:
        hosted_reason = "local candidate gates or clean checkout are incomplete"
    elif hosted_run_id is not None:
        try:
            hosted_evidence = _hosted_linux_from_run(
                root, hosted_run_id, wheel, sdist, commit=commit,
                version=current_version, source_digest=source_digest,
                input_digest=input_digest, artifacts=artifacts,
            )
        except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
            hosted_reason = str(exc)
        else:
            hosted_reason = None
    private_reporting_enabled = False
    if local_experiment_allowed and hosted_evidence is not None:
        try:
            private_reporting_enabled = _private_reporting_enabled(root)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            hosted_reason = str(exc)
        else:
            if not private_reporting_enabled:
                hosted_reason = "private vulnerability reporting is disabled"
    linux_customer_evaluation_allowed = (
        local_experiment_allowed and hosted_evidence is not None and private_reporting_enabled
    )
    final_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    final_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip())
    if (
        final_commit != commit
        or final_dirty != dirty
        or package_source_sha256(root) != source_digest
        or execution_inputs_sha256(root) != input_digest
    ):
        raise ValueError("candidate identity changed during release evidence review")
    return {
        "schema_version": 1,
        "checked_at": datetime.now(UTC).isoformat(),
        "status": "REVIEW_REQUIRED" if receipts_complete else "BLOCKED",
        "evaluation_allowed": linux_customer_evaluation_allowed,
        "local_experiment_allowed": local_experiment_allowed,
        "linux_customer_evaluation_allowed": linux_customer_evaluation_allowed,
        "private_vulnerability_reporting_enabled": private_reporting_enabled,
        "linux_customer_evaluation_reason": hosted_reason,
        "hosted_linux_workflow_run_id": hosted_run_id,
        "hosted_linux_evidence": hosted_evidence,
        "customer_environment_validated": False,
        "decision_scope": (
            "hosted Linux candidate evaluation"
            if linux_customer_evaluation_allowed else
            "local experiments only" if local_experiment_allowed else
            "no evaluation authorized"
        ),
        "production_release_authorized": False,
        "candidate_source_sha256": source_digest,
        "candidate_execution_inputs_sha256": input_digest,
        "local_execution_commit": next(iter(local_execution_commits), None),
        "previous_release": previous_release,
        "git_commit": commit,
        "git_dirty": dirty,
        "artifacts": artifacts,
        "gates": checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("docs/commercial/release-gates.json"))
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hosted-run-id", type=int)
    args = parser.parse_args()
    try:
        report = evaluate(args.manifest, args.wheel, args.sdist, args.hosted_run_id)
    except (
        OSError, ValueError, KeyError, TypeError, IndexError, StopIteration,
        ElementTree.ParseError, tarfile.TarError, zipfile.BadZipFile,
    ) as exc:
        report = {"schema_version": 1, "checked_at": datetime.now(UTC).isoformat(), "status": "INVALID",
                  "reason": str(exc)}
    write_json_atomic(args.report, report)
    print(report["status"])
    return {"REVIEW_REQUIRED": 2, "INVALID": 1, "BLOCKED": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
