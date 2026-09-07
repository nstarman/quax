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
"""

from importlib.util import find_spec

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser


def setup(namespace):
    """Expose optional-dependency flags to `skip: next if(...)` conditions.

    `find_spec` rather than a guarded import: it answers the same question
    without pulling the package in, and keeps this file type-checkable when the
    optional dependency is not installed.
    """
    namespace["have_diffrax"] = find_spec("diffrax") is not None


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
