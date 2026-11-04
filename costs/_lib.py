"""Shared helpers for the cost scenarios. Stdlib only."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOURS_PER_MONTH = 730
DAYS_PER_MONTH = 30.4
WEEKS_PER_MONTH = 4.35


class Data:
    """Read-only view over a TOML file that records which entries a scenario used."""

    def __init__(self, path: Path, require_verified: bool):
        self._d = tomllib.loads(path.read_text(encoding="utf-8"))
        self._require_verified = require_verified
        self.used: set[str] = set()

    def __call__(self, dotted: str) -> dict:
        node = self._d
        parts = dotted.split(".")
        # Keys themselves can contain dots only through TOML quoting, which we never use,
        # but model ids contain hyphens, so split on dots is safe.
        for part in parts:
            if part not in node:
                raise KeyError(f"{dotted}: no entry '{part}'")
            node = node[part]
        if self._require_verified and not node.get("verified", False):
            raise ValueError(f"{dotted} is not verified; a cost table may not use it")
        self.used.add(dotted)
        return node

    @property
    def raw(self) -> dict:
        return self._d


def prices() -> Data:
    return Data(ROOT / "data" / "prices.toml", require_verified=True)


def hardware() -> Data:
    return Data(ROOT / "data" / "hardware.toml", require_verified=False)


def llm_cost(
    p: dict,
    inp: float,
    out: float,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
    batch: bool = False,
) -> float:
    """USD for a token mix. `inp` excludes cached tokens. Token counts are raw counts."""
    cost = (
        inp * p["input"]
        + out * p["output"]
        + cache_read * p.get("cache_read", p["input"])
        + cache_write * p.get("cache_write", p["input"])
    ) / 1e6
    if batch:
        cost *= 1 - p.get("batch_discount", 0.0)
    return cost


def usd(x: float) -> str:
    """Money for estimates: cents below $10, whole dollars below $1,000, two significant
    figures above that. Sub-cent amounts keep two significant figures."""
    if x == 0:
        return "$0"
    ax = abs(x)
    if ax < 1:
        return f"${sig(x, 2):.6f}".rstrip("0").rstrip(".")
    if ax < 10:
        return f"${x:.2f}"
    if ax < 1000:
        return f"${x:,.0f}"
    return f"${sig(x, 2):,.0f}"


def usd_exact(x: float) -> str:
    """A list price as the vendor writes it."""
    s = f"{x:,.4f}".rstrip("0")
    if s.endswith("."):
        s += "00"
    elif len(s.split(".")[1]) == 1:
        s += "0"
    return "$" + s


def sig(x: float, n: int = 2) -> float:
    if x == 0:
        return 0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def num(x: float, n: int = 2) -> str:
    """Counts to n significant figures with K/M/B suffixes above 10,000."""
    ax = abs(x)
    if ax >= 1e9:
        return f"{sig(x / 1e9, n):g}B"
    if ax >= 1e6:
        return f"{sig(x / 1e6, n):g}M"
    if ax >= 1e4:
        return f"{sig(x / 1e3, n):g}K"
    if ax >= 100:
        return f"{sig(x, n):,.0f}"
    return f"{sig(x, n):g}"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class Table:
    def __init__(self, *cols: str, align: str | None = None):
        self.cols = cols
        self.align = align or ("l" + "r" * (len(cols) - 1))
        self.rows: list[tuple[str, ...]] = []

    def add(self, *cells) -> "Table":
        if len(cells) != len(self.cols):
            raise ValueError(
                f"row has {len(cells)} cells, table has {len(self.cols)} columns"
            )
        self.rows.append(tuple(str(c) for c in cells))
        return self

    def md(self) -> str:
        if not self.rows:
            raise ValueError("empty table")
        marks = {"l": ":--", "r": "--:", "c": ":-:"}
        out = [
            "| " + " | ".join(self.cols) + " |",
            "| " + " | ".join(marks[a] for a in self.align) + " |",
        ]
        out += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(out) + "\n"
