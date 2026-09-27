import json
from pathlib import Path

import pytest

from trainguard.run_store import RunStore
from trainguard.support import SupportBundleError, build_support_bundle, export_support_bundle


def _run_with_secret_metadata(root: Path) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    secret = "CUSTOMER_SECRET_827"
    (run_dir / "run.json").write_text(
        json.dumps({
            "run_id": "safe-id", "status": "FAILED", "reason": secret,
            "config_fingerprint": "a" * 64,
            "environment": {"source_sha256": "b" * 64, "secret": secret},
            "config": {"data": {"path": "/private/customer/data"}},
        })
    )
    (run_dir / "launcher.log").write_text(secret + " sample_ids=[123] /private/customer/data")
    store = RunStore(run_dir / "run.sqlite3")
    store.create_run("safe-id", "a" * 64, "now")
    store.start_attempt("safe-id", "attempt-001", 1, None, 0)
    store.finish_attempt("attempt-001", "FAILED", 1, secret)
    store.record_checkpoint("/private/customer/data", "safe-id", None, None, "INVALID", secret)
    store.close()
    return run_dir


def test_support_bundle_excludes_raw_evidence_and_secrets(tmp_path: Path) -> None:
    run_dir = _run_with_secret_metadata(tmp_path)
    output = tmp_path / "support.json"
    export_support_bundle(run_dir, output)
    value = output.read_text()
    assert "CUSTOMER_SECRET_827" not in value
    assert "/private/customer/data" not in value
    assert "sample_ids" not in value
    assert json.loads(value)["checkpoints"] == {"invalid": 1}
    assert json.loads(value)["attempts"][0]["status"] == "FAILED"


def test_support_bundle_fails_closed_on_missing_index(tmp_path: Path) -> None:
    run_dir = _run_with_secret_metadata(tmp_path)
    (run_dir / "run.sqlite3").unlink()
    with pytest.raises(SupportBundleError, match="index"):
        build_support_bundle(run_dir)
    assert not (tmp_path / "support.json").exists()
