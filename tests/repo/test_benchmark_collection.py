import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def benchmark_collection():
    source = Path(__file__).parents[1] / "benchmark" / "conftest.py"
    spec = importlib.util.spec_from_file_location("benchmark_collection", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("plugin_available", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("noncanonical", [False, True])
def test_benchmark_collection_paths(
    benchmark_collection, monkeypatch, plugin_available, explicit, noncanonical
):
    module = benchmark_collection
    here = module._HERE
    path = here / "test_numpy.py"
    if noncanonical:
        path = here.parent / "unit" / ".." / "benchmark" / "test_numpy.py"
    monkeypatch.setattr(module, "_benchmark_plugin_available", lambda: plugin_available)
    config = SimpleNamespace(
        args=[str(here)] if explicit else [str(here.parent)],
        invocation_params=SimpleNamespace(args=()),
    )
    expected = None if plugin_available and explicit else True
    assert module.pytest_ignore_collect(path, config) is expected


def test_benchmark_collection_leaves_other_directories(
    benchmark_collection, monkeypatch
):
    module = benchmark_collection
    monkeypatch.setattr(module, "_benchmark_plugin_available", lambda: False)
    config = SimpleNamespace(args=[], invocation_params=SimpleNamespace(args=()))
    path = module._HERE.parent / "unit" / "test_core.py"
    assert module.pytest_ignore_collect(path, config) is None
