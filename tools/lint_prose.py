"""Prose lint for the book. Stdlib only.

Fails on:
  * em or en dashes in any source file (prose, code, diagrams, data);
  * filler words and phrases in chapter prose;
  * American spellings in chapter prose (code spans and fenced blocks are skipped);
  * "Conclusion" or "Summary" headings.

Fails on empty input too: a check that examined nothing did not pass.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_EXTS = {
    ".qmd",
    ".md",
    ".py",
    ".d2",
    ".toml",
    ".yml",
    ".yaml",
    ".scss",
    ".css",
    ".html",
}
PROSE_EXTS = {".qmd", ".md"}
SKIP_DIRS = {"_book", "_freeze", ".quarto", ".git", "node_modules", "_site"}

DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")

FILLER = [
    r"\bdelve\w*",
    r"\brobust\w*",
    r"\bseamless\w*",
    r"\bleverag\w+",
    r"\bgame[- ]changer",
    r"\bcutting[- ]edge",
    r"\bunlock\w*",
    r"\bharness(ing)? the power",
    r"\bit'?s (important|worth) (to note|noting)",
    r"\bin today'?s\b",
    r"\blandscape\b",
    r"\bever[- ]evolving",
    r"\btapestry",
    r"\bembark\w*",
    r"\bnavigat\w+ the complexit\w+",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bin conclusion\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bparadigm\b",
    r"\bsupercharg\w+",
    r"\bempower\w*",
    r"\belevate\w*",
    r"\bdive (deep|into)\b",
    r"\bdeep dive\b",
    r"\bat the end of the day\b",
    r"\bplays a (crucial|vital|key) role\b",
    r"\bcrucial\b",
    r"\bpivotal\b",
    r"\bmyriad\b",
    r"\brealm\b",
    r"\bstreamlin\w+",
    r"\bin the world of\b",
    r"\blet'?s explore\b",
    r"\bbustling\b",
    r"\btestament\b",
    r"\bboasts?\b",
]
FILLER_RE = re.compile("|".join(FILLER), re.IGNORECASE)

# American spelling -> British. Matched case-insensitively on whole words in prose only.
US_SPELLINGS = {
    r"optimiz\w*": "optimis-",
    r"organiz\w*": "organis-",
    r"behavior\w*": "behaviour",
    r"analyz\w*": "analys-",
    r"colors?\b": "colour",
    r"favorite\w*": "favourite",
    r"minimiz\w*": "minimis-",
    r"maximiz\w*": "maximis-",
    r"utiliz\w*": "utilis-",
    r"prioritiz\w*": "prioritis-",
    r"summariz\w*": "summaris-",
    r"tokeniz\w*": "tokenis-",
    r"normaliz\w*": "normalis-",
    r"recogniz\w*": "recognis-",
    r"characteriz\w*": "characteris-",
    r"labeled": "labelled",
    r"labeling": "labelling",
    r"modeling": "modelling",
    r"canceled": "cancelled",
    r"licens(e|ing) (fee|cost)s?": "licence",
    r"catalog\b": "catalogue",
    r"amortiz\w*": "amortis-",
    r"parallelliz\w*|paralleliz\w*": "parallelis-",
    r"quantiz\w*": "quantis-",
    r"standardiz\w*": "standardis-",
    r"customiz\w*": "customis-",
    r"materializ\w*": "materialis-",
    r"serializ\w*": "serialis-",
    r"finaliz\w*": "finalis-",
}
US_RE = [
    (re.compile(r"\b(" + k + r")", re.IGNORECASE), v) for k, v in US_SPELLINGS.items()
]
# Product and API names that legitimately use American spelling.
ALLOW = re.compile(
    r"PagedAttention|Optimizer|ColorBrewer|OpenAI's Batch|catalog API|Unity Catalog|AWS Glue Data Catalog",
    re.IGNORECASE,
)

BAD_HEADING = re.compile(
    r"^#{1,6}\s+(conclusion|summary|in summary|wrapping up|final thoughts)\b",
    re.IGNORECASE,
)


def files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if (
            p.is_file()
            and p.suffix in SOURCE_EXTS
            and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS)
        ):
            out.append(p)
    return sorted(out)


def prose_lines(text: str):
    """Yield (line_no, line) outside fenced blocks, YAML front matter and HTML comments,
    with inline code, URLs and link targets removed."""
    fence = None
    in_front = False
    for i, line in enumerate(text.split("\n"), 1):
        if i == 1 and line.strip() == "---":
            in_front = True
            continue
        if in_front:
            if line.strip() == "---":
                in_front = False
            continue
        stripped = line.lstrip()
        m = re.match(r"(`{3,}|~{3,}|:{3,})", stripped)
        if m and m.group(1)[0] != ":":
            if fence is None:
                fence = m.group(1)
            elif stripped.startswith(fence):
                fence = None
            continue
        if fence is not None:
            continue
        line = re.sub(r"`[^`]*`", "", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"\{\{<[^>]*>\}\}", "", line)
        line = re.sub(r"\{[#.][^}]*\}", "", line)
        yield i, line


def main() -> int:
    paths = files()
    hits: list[str] = []
    prose_files = 0
    for path in paths:
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.split("\n"), 1):
            if DASHES.search(line):
                hits.append(f"{rel}:{i}: em or en dash: {line.strip()[:90]}")
        if path.suffix not in PROSE_EXTS or rel.name in {"LICENSE.md"}:
            continue
        prose_files += 1
        for i, line in prose_lines(text):
            if BAD_HEADING.match(line):
                hits.append(f"{rel}:{i}: heading not allowed: {line.strip()}")
            for m in FILLER_RE.finditer(line):
                hits.append(f"{rel}:{i}: filler '{m.group(0)}'")
            if ALLOW.search(line):
                line = ALLOW.sub("", line)
            for rx, fix in US_RE:
                for m in rx.finditer(line):
                    hits.append(
                        f"{rel}:{i}: American spelling '{m.group(1)}' (use {fix})"
                    )

    if not paths or prose_files == 0:
        print(
            f"lint: checked {len(paths)} files, {prose_files} prose files. Nothing to check is a failure."
        )
        return 1
    if hits:
        print("\n".join(hits))
        print(
            f"lint: {len(hits)} problem(s) in {len(paths)} files checked ({prose_files} prose)."
        )
        return 1
    print(f"lint: {len(paths)} files checked ({prose_files} prose), no problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
