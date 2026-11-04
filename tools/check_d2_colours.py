"""Fails on colour literals in diagram sources. Colours belong to the classes in
diagrams/_theme.d2, which tools/tokens.py generates from the design tokens.
Fails on zero files: a check that examined nothing did not pass."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LITERAL = re.compile(r"#[0-9A-Fa-f]{3}(?:[0-9A-Fa-f]{3})?\b|\b(?:rgb|rgba|hsl|hsla|oklch)\(")


def main() -> int:
    files = sorted(p for p in (ROOT / "diagrams").glob("*/*.d2"))
    hits = []
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
            for m in LITERAL.finditer(code):
                hits.append(f"{f.relative_to(ROOT)}:{i}: colour literal {m.group(0)}; use a class from _theme.d2")
    if not files:
        print("d2-colours: 0 diagram sources found. Nothing checked is a failure.")
        return 1
    if hits:
        print("\n".join(hits))
        print(f"d2-colours: {len(hits)} colour literal(s) in {len(files)} diagram sources checked.")
        return 1
    print(f"d2-colours: {len(files)} diagram sources checked, no colour literals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
