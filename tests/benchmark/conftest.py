"""Keep the benchmark suite out of ordinary test runs.

Benchmarks are collected only when *both*:

1. ``pytest-benchmark`` is installed (the `bench` dependency group), and
2. the invocation explicitly targets them — either the ``tests/benchmark`` path or
   any ``--benchmark*`` flag appears on the command line.

So a plain ``pytest`` / ``pytest tests`` never runs them (even if the plugin happens
to be installed in the environment), while the intended command does:

    uv run --group bench pytest tests/benchmark --benchmark-only
"""

import sys
from pathlib import Path


_HERE = Path(__file__).parent.resolve()


def _benchmarks_requested() -> bool:
    """True only if the invocation explicitly targets the benchmark suite.

    "Explicitly" means a ``--benchmark*`` flag, or a positional path pointing at
    (or inside) this directory. A bare substring match would over-trigger — e.g.
    ``pytest -k benchmark`` selects tests by keyword but must not pull the whole
    benchmark suite into a normal run.
    """
    try:
        import pytest_benchmark  # noqa: F401
    except ImportError:
        return False

    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith("--benchmark"):
            return True
        if arg.startswith("-"):
            continue  # an option, not a path
        prev = argv[i - 1] if i else ""
        if len(prev) == 2 and prev[0] == "-" and prev[1] != "-":
            continue  # value consumed by a short option like `-k` / `-m`
        try:
            target = Path(arg.split("::", 1)[0]).resolve()
        except (OSError, ValueError):
            continue
        if target == _HERE or _HERE in target.parents:
            return True
    return False


if not _benchmarks_requested():
    collect_ignore_glob = ["*"]
