# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
"""Draw the quax logo: a duck.

Quax's logo is Material Design Icons' "duck", the icon its docs header already
shows. This writes it as an SVG, sharp at any size, for the favicon and anywhere
else the logo is needed; for a bitmap, name a .png and give its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
from pathlib import Path


# Material Design Icons' "duck", by Pictogrammers, under the Apache License 2.0
# (https://pictogrammers.com/library/mdi/), in its 24-unit grid.
DUCK = (
    "M8.5,5A1.5,1.5 0 0,0 7,6.5A1.5,1.5 0 0,0 8.5,8"
    "A1.5,1.5 0 0,0 10,6.5A1.5,1.5 0 0,0 8.5,5M10,2A5,5 0 0,1 15,7"
    "C15,8.7 14.15,10.2 12.86,11.1C14.44,11.25 16.22,11.61 18,12.5"
    "C21,14 22,12 22,12C22,12 21,21 15,21H9C9,21 4,21 4,16"
    "C4,13 7,12 6,10C2,10 2,6.5 2,6.5C3,7 4.24,7 5,6.65"
    "C5.19,4.05 7.36,2 10,2Z"
)
INK = "#000"

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="512" height="512">
  <path d="{duck}" fill="{ink}"/>
</svg>
"""


def svg() -> str:
    """Return the logo as SVG text."""
    return SVG.format(duck=DUCK, ink=INK)


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description="Draw the quax logo: a duck.")
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        # Only a PNG needs a renderer; uv installs it from the header above.
        import resvg_py  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
