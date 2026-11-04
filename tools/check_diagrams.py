"""Fails if any D2 source has no rendered SVG, if an SVG is older than its source or the
theme, or if an SVG has no source. Fails on zero diagrams."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    theme = max((ROOT / "diagrams" / n).stat().st_mtime for n in ("_theme.d2", "_post.d2"))
    sources = sorted(p for p in (ROOT / "diagrams").glob("*/*.d2"))
    problems = []
    for src in sources:
        svg = ROOT / "images" / src.relative_to(ROOT / "diagrams").with_suffix(".svg")
        if not svg.exists():
            problems.append(f"missing {svg.relative_to(ROOT)}")
        elif svg.stat().st_mtime < max(src.stat().st_mtime, theme):
            problems.append(f"stale {svg.relative_to(ROOT)}")
    for svg in sorted((ROOT / "images").glob("*/*.svg")):
        if not (
            ROOT / "diagrams" / svg.relative_to(ROOT / "images").with_suffix(".d2")
        ).exists():
            problems.append(f"orphan {svg.relative_to(ROOT)} has no source")
    if not sources:
        print("diagrams: 0 sources found. Nothing checked is a failure.")
        return 1
    if problems:
        print("\n".join(problems))
        print(
            f"diagrams: {len(problems)} problem(s) in {len(sources)} diagrams. Run `make diagrams`."
        )
        return 1
    print(f"diagrams: {len(sources)} diagrams checked, all rendered and current.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
