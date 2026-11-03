"""Test repository collection policy rather than array operations.

Keep these checks in tests/repo so the benchmark opt-in gate cannot hide them.
"""

import sys
from types import SimpleNamespace

import pytest

from ..benchmark import conftest as benchmark_collection


def _config(target):
    return SimpleNamespace(
        args=[str(target)], invocation_params=SimpleNamespace(args=())
    )


@pytest.mark.parametrize(
    ("path_kind", "plugin_available", "explicit", "expected"),
    [
        pytest.param("symlink", False, False, True, id="symlink-no-plugin"),
        pytest.param("symlink", False, True, True, id="symlink-explicit-no-plugin"),
        pytest.param("symlink", True, False, True, id="symlink-with-plugin"),
        pytest.param("canonical", True, False, True, id="ordinary-with-plugin"),
        pytest.param("other", False, False, None, id="unrelated-directory"),
    ],
)
def test_benchmark_collection_paths(
    monkeypatch,
    tmp_path,
    path_kind,
    plugin_available,
    explicit,
    expected,
):
    module = benchmark_collection
    here = module._HERE
    path = here / "test_numpy.py"
    if path_kind == "symlink":
        link = tmp_path / "checkout"
        try:
            link.symlink_to(here.parents[1], target_is_directory=True)
        except OSError as error:
            if sys.platform == "win32" and getattr(error, "winerror", None) == 1314:
                pytest.skip("Creating symlinks requires a Windows privilege")
            raise
        path = link / "tests" / "benchmark" / "test_numpy.py"
    elif path_kind == "other":
        path = here.parent / "unit" / "test_core.py"
    monkeypatch.setattr(module, "_benchmark_plugin_available", lambda: plugin_available)
    config = _config(path.parent if explicit else path.parents[1])
    assert module.pytest_ignore_collect(path, config) is expected
