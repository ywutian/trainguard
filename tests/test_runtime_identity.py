import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import zipfile
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
            return value.replace("\n", "\r\n")
        return value

    def locate_file(self, filename: str):
        return self.original.locate_file(filename)


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
    monkeypatch.setattr(environment, "_verify_record_files", lambda *_: None)
    inventory = environment.installed_distributions()
    serialized = json.dumps(inventory)
    assert inventory[0]["direct_url_sha256"] is not None
    assert "secret-user" not in serialized
    assert "secret-token" not in serialized
    assert "private.example" not in serialized


def test_installed_file_bytes_must_match_record_at_runtime_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalDistribution:
        def locate_file(self, filename: str):
            return tmp_path / filename

    monkeypatch.setattr(environment.sys, "prefix", str(tmp_path))
    path = tmp_path / "dependency.py"
    path.write_bytes(b"pass\n")
    digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=")
    record = f"dependency.py,sha256={digest.decode()},{path.stat().st_size}\n"
    environment._verify_record_files(LocalDistribution(), record, "dependency")
    path.write_bytes(b"fail\n")
    with pytest.raises(ValueError, match="file differs from record"):
        environment._verify_record_files(LocalDistribution(), record, "dependency")
    with pytest.raises(ValueError, match="hash is missing"):
        environment._verify_record_files(LocalDistribution(), "dependency.py,,5\n", "dependency")


def test_startup_identity_detects_new_import_hook_and_hash_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(site_dir))
        original = environment.startup_identity_sha256()
        (site_dir / "untracked.pth").write_text("import builtins; builtins.marker = True\n")
        assert environment.startup_identity_sha256() != original
        (site_dir / "untracked.pth").unlink()
        patch.setenv("PYTHONHASHSEED", "1729")
        assert environment.startup_identity_sha256() != original


def test_startup_identity_detects_explicit_import_tree_and_zip_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "imports"
    root.mkdir()
    hook = root / "sitecustomize.py"
    hook.write_text("value = 'one'\n")
    monkeypatch.setenv("PYTHONPATH", str(root))
    first = environment.startup_identity_sha256()
    hook.write_text("value = 'two'\n")
    assert environment.startup_identity_sha256() != first
    package = root / "otherpackage"
    package.mkdir()
    module = package / "trainer.py"
    module.write_text("value = 'one'\n")
    first = environment.startup_identity_sha256()
    module.write_text("value = 'two'\n")
    assert environment.startup_identity_sha256() != first

    archive = tmp_path / "imports.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("sitecustomize.py", "value = 'one'\n")
    monkeypatch.setenv("PYTHONPATH", str(archive))
    first = environment.startup_identity_sha256()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("sitecustomize.py", "value = 'two'\n")
    assert environment.startup_identity_sha256() != first
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/sitecustomize.py", "value = 'one'\n")
    monkeypatch.setenv("PYTHONPATH", str(archive / "nested"))
    first = environment.startup_identity_sha256()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/sitecustomize.py", "value = 'two'\n")
    assert environment.startup_identity_sha256() != first


def test_startup_identity_detects_added_import_root_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "added-imports"
    root.mkdir()
    module = root / "external_module.py"
    module.write_text("value = 'one'\n")
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(root))
        first = environment.startup_identity_sha256()
        module.write_text("value = 'two'\n")
        assert environment.startup_identity_sha256() != first


def test_startup_identity_detects_custom_root_under_interpreter_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment.sysconfig.get_paths()
    prefix = tmp_path / "python-prefix"
    root = prefix / "custom-imports"
    root.mkdir(parents=True)
    module = root / "customer_module.py"
    module.write_text("value = 'one'\n")
    with monkeypatch.context() as patch:
        patch.setattr(environment.sys, "base_prefix", str(prefix))
        patch.syspath_prepend(str(root))
        first = environment.startup_identity_sha256()
        module.write_text("value = 'two'\n")
        assert environment.startup_identity_sha256() != first


def test_startup_identity_binds_interpreter_controls(tmp_path: Path) -> None:
    script = (
        "import json, sys; "
        "from trainguard.environment import startup_identity_sha256; "
        "print(json.dumps({'optimize': sys.flags.optimize, "
        "'debug': __debug__, 'identity': startup_identity_sha256()}))"
    )

    def snapshot(level: str, cache: Path) -> dict:
        settings = os.environ.copy()
        settings.update({
            "PYTHONOPTIMIZE": level,
            "PYTHONPYCACHEPREFIX": str(cache),
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=settings, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    snapshots = [snapshot(str(level), tmp_path / "cache-a") for level in range(3)]
    assert [item["optimize"] for item in snapshots] == [0, 1, 2]
    assert [item["debug"] for item in snapshots] == [True, False, False]
    assert len({item["identity"] for item in snapshots}) == 3
    assert snapshot("0", tmp_path / "cache-b")["identity"] != snapshots[0]["identity"]


def test_added_import_root_cannot_shadow_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "trainguard.pyc").write_bytes(b"shadow")
    with monkeypatch.context() as patch:
        patch.syspath_prepend(str(tmp_path))
        with pytest.raises(ValueError, match="may shadow"):
            environment.startup_identity_sha256()


def test_worker_ignores_implicit_working_directory_package_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow = tmp_path / "trainguard"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("__version__ = '0.3.6'\n")
    (shadow / "trainer.py").write_text(
        "from pathlib import Path\nPath('shadow-ran').write_text('unsafe')\n"
    )
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    run_dir, succeeded = run(source, tmp_path / "runs")
    assert succeeded, (run_dir / "launcher.log").read_text()
    assert not (tmp_path / "shadow-ran").exists()


def test_explicit_import_root_cannot_shadow_application(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "trainguard"
    package.mkdir()
    (package / "trainer.py").write_text("raise SystemExit(1)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(ValueError, match="may shadow"):
        environment.startup_identity_sha256()
    archive = tmp_path / "shadow.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/trainguard/trainer.py", "raise SystemExit(1)\n")
    monkeypatch.setenv("PYTHONPATH", str(archive / "nested"))
    with pytest.raises(ValueError, match="may shadow"):
        environment.startup_identity_sha256()


@pytest.mark.parametrize("filename", ["trainguard.pyc", "trainguard.abi3.so"])
def test_explicit_import_root_rejects_top_level_module_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str,
) -> None:
    (tmp_path / filename).write_bytes(b"shadow")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(ValueError, match="may shadow"):
        environment.startup_identity_sha256()


@pytest.mark.parametrize("existing", [False, True])
def test_run_output_inside_explicit_import_root_is_rejected_before_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool,
) -> None:
    imports = tmp_path / "imports"
    if existing:
        imports.mkdir()
    monkeypatch.setenv("PYTHONPATH", str(imports))
    output = imports / "runs"
    source = Path(__file__).parents[1] / "configs" / "cpu_demo.yaml"
    with pytest.raises(ValueError, match="outside active Python import roots"):
        run(source, output)
    assert not output.exists()


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
    with monkeypatch.context() as patch:
        patch.setenv("PYTHONPATH", "/temporary/import-shadow")
        with pytest.raises(ValueError, match="startup_identity_sha256 differs"):
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
