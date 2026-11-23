"""Shape tokens shared by the examples, for `bearshape` annotations.

`bearshape` writes variadic (`~B`), arithmetic (`N + 1`), fixed (`2`) and "any shape"
(`...`) dimensions as runtime expressions, which a static type checker rejects inside a
type annotation. Each token is therefore a plain static type under `TYPE_CHECKING` and
the real `bearshape` dimension at runtime. See
https://github.com/beartype/bearshape/blob/main/docs/features/static-typing.md.
"""

from typing import Literal, TYPE_CHECKING, TypeAlias

from bearshape import Dimension


if TYPE_CHECKING:
    AnyShape: TypeAlias = Literal["..."]
    VBatch: TypeAlias = Literal["*batch"]
    Two: TypeAlias = Literal[2]
else:
    AnyShape = ...
    VBatch = ~Dimension("batch")
    Two = Dimension(2)
