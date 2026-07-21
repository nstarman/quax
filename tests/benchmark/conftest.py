"""Keep the benchmark suite out of ordinary test runs.

Benchmarks are collected only when *both*:

1. a benchmark plugin (``pytest-benchmark`` or ``pytest-codspeed``, from the `bench`
   dependency group) is installed, and
2. the invocation explicitly targets them — a ``--benchmark*`` flag, ``--codspeed``,
   or a positional path pointing at (or inside) this directory.

So a plain ``pytest`` / ``pytest tests`` never runs them (even if a plugin happens
to be installed), while the intended commands do:

    uv run --group bench pytest tests/benchmark --benchmark-only   # local
    uv run --group bench pytest tests/benchmark --codspeed         # CodSpeed CI

Rather than hand-parse ``sys.argv`` (which means reasoning about which short
options consume a value — ``-k VALUE`` does, ``-v`` does not), we use pytest's
already-parsed positional paths (``config.args``) inside ``pytest_ignore_collect``.
"""

from pathlib import Path


_HERE = Path(__file__).parent.resolve()


def _benchmark_plugin_available() -> bool:
    """Whether a plugin providing the ``benchmark`` fixture is installed."""
    for plugin in ("pytest_benchmark", "pytest_codspeed"):
        try:
            __import__(plugin)
            return True
        except ImportError:
            continue
    return False


def _explicitly_targeted(config) -> bool:
    """Whether the invocation explicitly asked for the benchmark suite."""
    # A `--benchmark*` flag (pytest-benchmark) or `--codspeed` (CodSpeed) opts in.
    args = config.invocation_params.args
    if any(a.startswith("--benchmark") or a == "--codspeed" for a in args):
        return True
    # A positional path resolving to (or inside) this directory. `config.args`
    # holds paths with options already stripped by pytest, so boolean flags like
    # `-v` and value options like `-k VALUE` are handled correctly for free.
    for arg in config.args:
        path = Path(str(arg).split("::", 1)[0])
        try:
            resolved = path.resolve()
        except (OSError, ValueError):  # pragma: no cover - defensive
            continue
        if resolved == _HERE or _HERE in resolved.parents:
            return True
    return False


def pytest_ignore_collect(collection_path, config):
    """Ignore this directory's contents unless benchmarks were explicitly requested."""
    path = Path(collection_path)
    if path != _HERE and _HERE not in path.parents:
        return None  # not under the benchmark directory; not our concern

    if not _benchmark_plugin_available():
        return True  # no benchmark plugin -> never collect benchmarks

    return None if _explicitly_targeted(config) else True
