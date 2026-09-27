import json
import os
import sqlite3
from pathlib import Path

import pytest

from trainguard import benchmark, campaign, environment
from trainguard.config import load_config
from trainguard.controller import resume, run
from trainguard.run_store import RunStore


def test_existing_run_index_migration_preserves_legacy_identity(tmp_path: Path) -> None:
    path = tmp_path / "run.sqlite3"
    with sqlite3.connect(path) as database:
        database.execute(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, "
            "config_fingerprint TEXT NOT NULL, started_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        database.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            ("legacy", "SUCCEEDED", "a" * 64, "start", "finish"),
        )
    store = RunStore(path)
    store.close()
    with sqlite3.connect(path) as database:
        assert database.execute(
            "SELECT run_id, status, config_fingerprint, evidence_schema_version, "
            "measurement_sha256 FROM runs"
        ).fetchone() == ("legacy", "SUCCEEDED", "a" * 64, 1, None)


class ChangedDistribution:
    def __init__(self, original, change: str):
        self.original = original
        self.metadata = original.metadata
        self.version = "99.0" if change == "version" else original.version
        self.change = change

    def read_text(self, filename: str):
        value = self.original.read_text(filename)
        if filename == "RECORD" and self.change == "record":
            return value + "different-installed-wheel,,\n"
        return value


def test_installed_inventory_is_complete_and_does_not_expose_origin() -> None:
    packages = environment.installed_distributions()
    names = [package["name"] for package in packages]
    assert names == sorted(set(names))
    assert "sympy" in names  # Indirect runtime dependency of the locked torch install.
    assert "trainguard" in names
    assert all(len(package["record_sha256"]) == 64 for package in packages)
    assert "file://" not in json.dumps(packages)
    assert str(Path(__file__).parents[1]) not in json.dumps(packages)


def test_direct_url_credentials_are_only_recorded_as_digest(monkeypatch) -> None:
    class PrivateDistribution:
        def __init__(self):
            self.metadata = {"Name": "private-library"}
            self.version = "1.0"

        def read_text(self, filename: str):
            if filename == "RECORD":
                return "private_library.py,sha256=abc,1\n"
            if filename == "direct_url.json":
                return '{"url":"https://secret-user:secret-token@private.example/library.whl"}'
            return None

    monkeypatch.setattr(
        environment.importlib.metadata, "distributions", lambda: [PrivateDistribution()]
    )
    inventory = environment.installed_distributions()
    serialized = json.dumps(inventory)
    assert inventory[0]["direct_url_sha256"] is not None
    assert "secret-user" not in serialized
    assert "secret-token" not in serialized
    assert "private.example" not in serialized


def test_third_party_editable_dependency_is_rejected_without_path_leak(monkeypatch) -> None:
    class EditableDistribution:
        def __init__(self):
            self.metadata = {"Name": "local-helper"}
            self.version = "1.0"

        def read_text(self, filename: str):
            if filename == "RECORD":
                return "local_helper.pth,sha256=abc,1\n"
            if filename == "direct_url.json":
                return '{"url":"file:///private/workspace", "dir_info":{"editable":true}}'
            return None

    monkeypatch.setattr(
        environment.importlib.metadata, "distributions", lambda: [EditableDistribution()]
    )
    with pytest.raises(
        ValueError, match="editable dependency has no frozen source identity: local-helper"
    ) as error:
        environment.installed_distributions()
    assert "/private/workspace" not in str(error.value)


def test_missing_installed_record_is_rejected(monkeypatch) -> None:
    class IncompleteDistribution:
        def __init__(self):
            self.metadata = {"Name": "untracked-dependency"}
            self.version = "1.0"

        def read_text(self, filename: str):
            return None

    monkeypatch.setattr(
        environment.importlib.metadata, "distributions", lambda: [IncompleteDistribution()]
    )
    with pytest.raises(ValueError, match="installed package identity is incomplete"):
        environment.installed_distributions()


def test_resume_rejects_indirect_version_or_wheel_metadata_change(
    tmp_path: Path, monkeypatch
) -> None:
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    config = tmp_path / "config.json"
    config.write_text(json.dumps(load_config(source).model_dump()))
    with monkeypatch.context() as patch:
        patch.setattr(RunStore, "create_run", lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit(73)))
        with pytest.raises(SystemExit, match="73"):
            run(config, tmp_path / "runs")
    run_dir = next((tmp_path / "runs").iterdir())
    saved = json.loads((run_dir / "run.json").read_text())
    assert any(
        package["name"] == "sympy" for package in saved["environment"]["installed_distributions"]
    )
    missing_inventory = json.loads(json.dumps(saved))
    del missing_inventory["environment"]["installed_distributions"]
    (run_dir / "run.json").write_text(json.dumps(missing_inventory))
    with pytest.raises(ValueError, match="installed_distributions identity is missing"):
        resume(run_dir)
    (run_dir / "run.json").write_text(json.dumps(saved))
    original = environment.importlib.metadata.distributions
    for change in ("version", "record"):

        def changed(change=change):
            for distribution in original():
                if distribution.metadata.get("Name", "").lower() == "sympy":
                    yield ChangedDistribution(distribution, change)
                else:
                    yield distribution

        with monkeypatch.context() as patch:
            patch.setattr(environment.importlib.metadata, "distributions", changed)
            with pytest.raises(ValueError, match="installed_distributions differs"):
                resume(run_dir)
        assert not (run_dir / "attempts").exists()
    current_threads = os.environ.get("OMP_NUM_THREADS")
    with monkeypatch.context() as patch:
        patch.setenv("OMP_NUM_THREADS", "999" if current_threads != "999" else "998")
        with pytest.raises(ValueError, match="environment_options differs"):
            resume(run_dir)
    assert not (run_dir / "attempts").exists()
    assert resume(run_dir), (run_dir / "launcher.log").read_text()


@pytest.mark.parametrize(
    ("filename", "resume_function", "schema"),
    [
        ("acceptance.json", campaign.resume_campaign, 1),
        ("results.json", benchmark.resume_benchmark, 2),
    ],
)
def test_historical_campaign_or_benchmark_without_inventory_is_rejected(
    tmp_path: Path, filename, resume_function, schema
) -> None:
    config = load_config(Path(__file__).parents[1] / "configs" / "cpu_demo.yaml")
    identity = environment.environment_snapshot(config.run.world_size, config.run.device, tmp_path)
    del identity["installed_distributions"]
    (tmp_path / filename).write_text(
        json.dumps(
            {
                "schema_version": schema,
                "config": config.model_dump(),
                "environment": identity,
            }
        )
    )
    with pytest.raises(ValueError, match="installed_distributions identity is missing"):
        resume_function(tmp_path)
