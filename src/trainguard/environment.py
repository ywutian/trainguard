"""Reproducible source identity and non-secret runtime measurements."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import zipfile
from io import StringIO
from pathlib import Path

import torch


def _verify_record_files(distribution: importlib.metadata.Distribution, record: str,
                         name: str) -> None:
    """Check installed bytes against the wheel's saved file hashes at a run boundary."""
    prefix = Path(sys.prefix).resolve()
    seen = set()
    rows = list(csv.reader(StringIO(record)))
    if not rows:
        raise ValueError(f"installed package file record is empty: {name}")
    for row in rows:
        if len(row) != 3 or not row[0] or row[0] in seen:
            raise ValueError(f"installed package file record is invalid: {name}")
        seen.add(row[0])
        filename, digest_field, size_field = row
        if not digest_field and not size_field and filename.endswith(".dist-info/RECORD"):
            continue
        algorithm, separator, expected = digest_field.partition("=")
        if separator != "=" or algorithm not in {"sha256", "sha384", "sha512"} or not expected:
            raise ValueError(f"installed package file hash is missing or invalid: {name}")
        try:
            size = int(size_field)
            path = Path(distribution.locate_file(filename))
            resolved = path.resolve(strict=True)
            metadata = path.lstat()
            if (
                size < 0 or not resolved.is_relative_to(prefix)
                or not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size
            ):
                raise ValueError(f"installed package file differs from record: {name}")
            digest = hashlib.new(algorithm)
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
            if actual != expected:
                raise ValueError(f"installed package file differs from record: {name}")
        except (OSError, TypeError) as exc:
            raise ValueError(f"installed package file cannot be verified: {name}") from exc


def installed_distributions() -> list[dict[str, str | None]]:
    """Record the installed package metadata used by this interpreter."""
    packages = []
    names = set()
    for distribution in importlib.metadata.distributions():
        original_name = distribution.metadata.get("Name")
        version = distribution.version
        record = distribution.read_text("RECORD")
        if not original_name or not version or not record or not record.strip():
            raise ValueError("installed package identity is incomplete")
        name = re.sub(r"[-_.]+", "-", original_name).lower()
        if name in names:
            raise ValueError(f"installed package identity is ambiguous: {name}")
        names.add(name)
        direct_url = distribution.read_text("direct_url.json")
        if direct_url is not None:
            try:
                origin = json.loads(direct_url)
            except json.JSONDecodeError as exc:
                raise ValueError(f"installed package origin is invalid: {name}") from exc
            if not isinstance(origin, dict):
                raise ValueError(f"installed package origin is invalid: {name}")
            directory_info = origin.get("dir_info", {})
            if not isinstance(directory_info, dict):
                raise ValueError(f"installed package origin is invalid: {name}")
            if directory_info.get("editable") and name != "trainguard":
                raise ValueError(f"editable dependency has no frozen source identity: {name}")
        _verify_record_files(distribution, record, name)
        packages.append(
            {
                "name": name,
                "version": version,
                "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
                "direct_url_sha256": (
                    hashlib.sha256(direct_url.encode("utf-8")).hexdigest()
                    if direct_url is not None
                    else None
                ),
            }
        )
    return sorted(packages, key=lambda package: package["name"])


def source_sha256() -> str:
    source = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    # Package bytes have the same identity in a checkout and an installed wheel.
    # Installed package metadata and Python are checked separately at resume.
    for path in sorted(source.rglob("*.py")):
        name = path.relative_to(source).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)

    return digest.hexdigest()


def _import_entry_sha256(path: Path) -> str:
    """Hash an explicit import root, including code and data reachable from it."""
    digest = hashlib.sha256()
    files = 0
    total_bytes = 0

    def add(name: bytes, kind: bytes) -> None:
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(kind)

    def visit(current: Path, relative: Path) -> None:
        nonlocal files, total_bytes
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            add(os.fsencode(str(relative)), b"M")
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("Python import path contains a linked file")
        name = os.fsencode(str(relative))
        if stat.S_ISDIR(metadata.st_mode):
            add(name, b"D")
            for child in sorted(current.iterdir()):
                visit(child, relative / child.name)
        elif stat.S_ISREG(metadata.st_mode):
            files += 1
            total_bytes += metadata.st_size
            if files > 100000 or total_bytes > 2 * 1024**3:
                raise ValueError("Python import path exceeds identity limits")
            add(name, b"F")
            digest.update(metadata.st_size.to_bytes(8, "big"))
            with current.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        else:
            raise ValueError("Python import path contains a non-regular file")

    visit(path, Path("."))
    return digest.hexdigest()


def _zip_import_container(path: Path) -> tuple[Path, str] | None:
    """Resolve both an archive entry and Python's archive/subdirectory syntax."""
    for candidate in (path, *path.parents):
        if candidate.is_file():
            if zipfile.is_zipfile(candidate):
                prefix = path.relative_to(candidate).as_posix()
                return candidate, "" if prefix == "." else prefix.rstrip("/") + "/"
            return None
    return None


def _shadows_application(path: Path, source_root: Path) -> bool:
    if path == source_root:
        return False
    if path.is_dir() and any(
        child.name == "trainguard" or child.name.startswith("trainguard.")
        for child in path.iterdir()
    ):
        return True
    zip_root = _zip_import_container(path)
    if zip_root is None:
        return False
    archive_path, prefix = zip_root
    with zipfile.ZipFile(archive_path) as archive:
        return any(
            name.startswith(prefix) and (
                name[len(prefix):] == "trainguard"
                or name[len(prefix):].startswith("trainguard.")
                or name[len(prefix):].startswith("trainguard/")
            )
            for name in archive.namelist()
        )


def _inherited_import_paths() -> list[Path]:
    """Exclude only the interpreter's implicit script/cwd entry, removed in workers."""
    paths = []
    script = sys.argv[0] if sys.argv else ""
    script_directory = (
        Path(script).resolve().parent if script and script not in {"-c", "-m", "-"} else None
    )
    cwd = Path.cwd().resolve()
    implicit_removed = False
    for entry in sys.path:
        resolved = Path(entry or cwd).resolve()
        if (
            not implicit_removed and not sys.flags.safe_path
            and (not entry or resolved == cwd or resolved == script_directory)
        ):
            implicit_removed = True
            continue
        paths.append(resolved)
    return paths


def require_output_outside_import_roots(output: Path) -> None:
    """Avoid self-changing import identity as a run creates its own artifacts."""
    destination = output.resolve()
    roots = _inherited_import_paths()
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath is not None:
        for entry in pythonpath.split(os.pathsep):
            if not entry:
                raise ValueError("PYTHONPATH contains an empty import root")
            roots.append(Path(entry).resolve())
    for root in roots:
        if not root.is_file() and destination.is_relative_to(root):
            raise ValueError("run output must be outside active Python import roots")


def startup_identity_sha256() -> str:
    """Bind import lookup paths, startup hooks, and Python path controls without exposing paths."""
    paths = [Path(entry or os.getcwd()).resolve() for entry in sys.path]
    inherited_paths = _inherited_import_paths()
    pythonpath = os.environ.get("PYTHONPATH")
    import_entries = []
    if pythonpath is not None:
        entries = pythonpath.split(os.pathsep)
        if any(not entry for entry in entries):
            raise ValueError("PYTHONPATH contains an empty import root")
        source_root = Path(__file__).resolve().parents[1]
        for entry in entries:
            original = Path(entry)
            if original.is_symlink():
                raise ValueError("PYTHONPATH contains a linked import root")
            root = original.resolve()
            if _shadows_application(root, source_root):
                raise ValueError("PYTHONPATH may shadow the application package")
            zip_root = _zip_import_container(root)
            import_entries.append((
                str(root), _import_entry_sha256(zip_root[0] if zip_root is not None else root)
            ))
    startup_files = []
    scan_paths = set(paths)
    if pythonpath:
        scan_paths.update(Path(entry).resolve() for entry in pythonpath.split(os.pathsep))
    for directory in sorted(scan_paths):
        zip_root = _zip_import_container(directory)
        if zip_root is not None or directory.is_file():
            startup_files.append((
                str(directory), _import_entry_sha256(zip_root[0] if zip_root is not None else directory)
            ))
            continue
        if not directory.is_dir():
            continue
        candidates = set(directory.glob("*.pth"))
        candidates.update(directory / name for name in (
            "sitecustomize.py", "sitecustomize.pyc", "usercustomize.py", "usercustomize.pyc"
        ))
        for candidate in sorted(candidates):
            if not candidate.exists() and not candidate.is_symlink():
                continue
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("Python startup file is linked or not a regular file")
            startup_files.append((str(candidate), hashlib.sha256(candidate.read_bytes()).hexdigest()))
    source_root = Path(__file__).resolve().parents[1]
    package_paths = sysconfig.get_paths()
    installed_roots = {
        Path(package_paths[name]).resolve()
        for name in ("purelib", "platlib") if package_paths.get(name)
    }
    stdlib_root = Path(package_paths["stdlib"]).resolve()
    interpreter_roots = {
        stdlib_root,
        stdlib_root / "lib-dynload",
    }
    explicit_roots = {
        Path(entry).resolve() for entry in pythonpath.split(os.pathsep)
    } if pythonpath is not None else set()
    extra_import_paths = []
    for root in sorted(set(inherited_paths) - explicit_roots):
        if root == source_root or root in installed_roots or root in interpreter_roots:
            continue
        if _shadows_application(root, source_root):
            raise ValueError("Python import path may shadow the application package")
        archive = _zip_import_container(root)
        extra_import_paths.append((
            str(root), _import_entry_sha256(archive[0] if archive is not None else root)
        ))
    hooks = {}
    for name in ("sitecustomize", "usercustomize"):
        module = sys.modules.get(name)
        if module is None:
            hooks[name] = None
            continue
        source = getattr(module, "__file__", None)
        if not isinstance(source, str):
            raise TypeError(f"Python startup hook identity is incomplete: {name}")
        hooks[name] = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    payload = {
        "sys_path": [str(path) for path in paths],
        "pythonpath_entries": import_entries,
        "extra_import_paths": extra_import_paths,
        "startup_files": startup_files,
        "hooks": hooks,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                "NVIDIA_VISIBLE_DEVICES", "PYTHONSAFEPATH",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_snapshot(world_size: int, device: str, storage_path: Path) -> dict:
    require_output_outside_import_roots(storage_path)
    repository = Path(__file__).resolve().parents[2]
    checkout_source = repository / "src" / "trainguard" / "environment.py"
    is_checkout = checkout_source.resolve() == Path(__file__).resolve()

    def git(*arguments):
        if not is_checkout:
            return None
        result = subprocess.run(
            ["git", *arguments], cwd=repository, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    memory = None
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            memory = int(result.stdout)
    elif hasattr(os, "sysconf"):
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(storage_path)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pydantic", "pyyaml", "typer")
    }
    git_status = git("status", "--porcelain")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory,
        "world_size": world_size,
        "worker_threads": 1,
        "device": device,
        "storage": "local filesystem",
        "storage_device": storage_path.stat().st_dev,
        "disk_free_bytes": disk.free,
        "versions": versions,
        "installed_distributions": installed_distributions(),
        "startup_identity_sha256": startup_identity_sha256(),
        "source_sha256": source_sha256(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if git_status is None else bool(git_status),
        "environment_options": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
