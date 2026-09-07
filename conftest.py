"""Execute the code examples in `README.md` and the quax skill.

Both are copied verbatim by people and by agents, so an example that stops
working as JAX moves is a real defect. Sybil turns each ```python block into
its own pytest item, reported against the line it lives on.

`patterns` are matched with `Path.match`, which is right-anchored, so
`README.md` also matches each example package's own README. Those are excluded:
like `docs/`, they carry illustrative fragments never meant to run standalone.

A block needing an optional dependency is skipped by the preceding comment,
which renders as nothing:

    <!--- skip: next if(not have_diffrax, 'diffrax not installed') -->

A region needing an optional feature is bracketed instead:

    <!--- skip: start if(not have_hijax, 'hijax unavailable') -->
    ...
    <!--- skip: end -->
"""

from importlib.util import find_spec

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser


def _have_hijax() -> bool:
    """Whether `quax.experimental.hijax` imports on the installed JAX.

    A guarded import rather than `find_spec`, because the question is not
    whether the module exists but whether this JAX is new enough for it: the
    module itself raises `ImportError` below its JAX floor. Quax supports a much
    older JAX than hijax does, so on the oldest supported dependencies the
    hijax examples in the docs cannot run and must be skipped.
    """
    try:
        import quax.experimental.hijax  # noqa: F401
    except ImportError:
        return False
    return True


def setup(namespace):
    """Expose optional-feature flags to `skip: ... if(...)` conditions.

    `find_spec` rather than a guarded import for diffrax: it answers the same
    question without pulling the package in, and keeps this file type-checkable
    when the optional dependency is not installed.
    """
    namespace["have_diffrax"] = find_spec("diffrax") is not None
    namespace["have_hijax"] = _have_hijax()


pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    patterns=[
        "README.md",
        "SKILL.md",
        "autodiff.md",
        "control-flow.md",
        "hijax.md",
        "how-to/*.md",
        "tutorials/*.md",
        "how-quaxify-works.md",
        "faq.md",
    ],
    excludes=["examples/*/README.md"],
    setup=setup,
).pytest()
