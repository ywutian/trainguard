"""Record a bounded source test identity for release-gate timeout diagnosis."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from trainguard.events import write_json_atomic

TEST_MODULE = re.compile(r"test_[A-Za-z0-9_]{1,96}\.py")
TEST_CLASS = re.compile(r"Test[A-Za-z0-9_]{1,96}")
TEST_FUNCTION = re.compile(r"test_[A-Za-z0-9_]{1,128}")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--safe-progress-file", type=Path, default=None)


def _source_identity(item: pytest.Item) -> str | None:
    source = Path(item.path)
    if (
        source.parent.resolve() != Path(__file__).resolve().parent
        or source.is_symlink()
        or TEST_MODULE.fullmatch(source.name) is None
    ):
        return None
    function = getattr(item, "originalname", None)
    if not isinstance(function, str) or TEST_FUNCTION.fullmatch(function) is None:
        return None
    test_class = getattr(item, "cls", None)
    if test_class is None:
        return f"{source.name}::{function}"
    class_name = getattr(test_class, "__name__", None)
    if not isinstance(class_name, str) or TEST_CLASS.fullmatch(class_name) is None:
        return None
    return f"{source.name}::{class_name}::{function}"


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    progress_file = item.config.getoption("--safe-progress-file")
    if progress_file is not None:
        write_json_atomic(
            progress_file,
            {"schema_version": 1, "identity": _source_identity(item)},
        )
    yield
