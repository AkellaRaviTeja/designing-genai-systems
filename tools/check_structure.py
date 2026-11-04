"""Checks that every system chapter has the fixed headings, in order, and that every
image a chapter references exists. Fails when it finds no chapters to check."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = [
    r"The brief",
    r"Questions to ask",
    r"The numbers",
    r"Version 0\b",
    r"Where .+ breaks?$",
    r"Version 1\b",
    r"The final design",
    r"Deployment",
    r"Cost",
    r"Optimisation",
    r"Operating it",
    r"Variants",
    r"Trade-offs",
]


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    heads = [
        m.group(1).strip() for m in re.finditer(r"^## (.+?)\s*(\{.*\})?$", text, re.M)
    ]
    problems = []
    pos = 0
    for pattern in REQUIRED:
        rx = re.compile(pattern)
        for j in range(pos, len(heads)):
            if rx.match(heads[j]):
                pos = j + 1
                break
        else:
            problems.append(f"{path.name}: missing or out of order: '## {pattern}'")
    breaks = sum(1 for h in heads if re.match(r"Where .+ breaks?$", h))
    versions = sum(1 for h in heads if re.match(r"Version \d+", h))
    if versions != breaks + 1:
        problems.append(
            f"{path.name}: {versions} versions for {breaks} breaks; expected one more version than breaks"
        )
    for img in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text):
        target = (path.parent / img).resolve()
        if not target.exists():
            problems.append(f"{path.name}: image not found: {img}")
    return problems


def main() -> int:
    chapters = sorted((ROOT / "systems").glob("*.qmd"))
    problems = [p for c in chapters for p in check(c)]
    if not chapters:
        print("structure: 0 system chapters found. Nothing checked is a failure.")
        return 1
    if problems:
        print("\n".join(problems))
        print(
            f"structure: {len(problems)} problem(s) in {len(chapters)} chapters checked."
        )
        return 1
    print(
        f"structure: {len(chapters)} chapters checked, {len(REQUIRED)} required sections each, all present."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
