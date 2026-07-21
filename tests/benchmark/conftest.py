"""Keep the benchmark suite out of ordinary test runs.

Benchmarks are collected only when *both*:

1. ``pytest-benchmark`` is installed (the `bench` dependency group), and
2. the invocation explicitly targets them — either a ``--benchmark*`` flag, or a
   positional path pointing at (or inside) this directory.

So a plain ``pytest`` / ``pytest tests`` never runs them (even if the plugin
happens to be installed), while the intended command does:

    uv run --group bench pytest tests/benchmark --benchmark-only

Rather than hand-parse ``sys.argv`` (which means reasoning about which short
options consume a value — ``-k VALUE`` does, ``-v`` does not), we use pytest's
already-parsed positional paths (``config.args``) inside ``pytest_ignore_collect``.
"""

from pathlib import Path


_HERE = Path(__file__).parent.resolve()


def _explicitly_targeted(config) -> bool:
    """Whether the invocation explicitly asked for the benchmark suite."""
    # A `--benchmark*` flag (e.g. `--benchmark-only`) always opts in.
    if any(a.startswith("--benchmark") for a in config.invocation_params.args):
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

    try:
        import pytest_benchmark  # noqa: F401
    except ImportError:
        return True  # plugin absent -> never collect benchmarks

    return None if _explicitly_targeted(config) else True
