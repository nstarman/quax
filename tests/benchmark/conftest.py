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


def _benchmarks_requested() -> bool:
    try:
        import pytest_benchmark  # noqa: F401
    except ImportError:
        return False
    return any("benchmark" in arg for arg in sys.argv[1:])


if not _benchmarks_requested():
    collect_ignore_glob = ["*"]
