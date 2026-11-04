"""Draws the cover art on the home page cards: one SVG per part, each a pattern taken from
what the part is about. Deterministic, so the output only changes when this file does.

  python3 tools/covers.py           write assets/covers/*.svg
  python3 tools/covers.py --check   fail if any cover is stale
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokens import PARTS, oklch_to_hex  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "covers"
W, H = 320, 200


def ink(hue: float, L: float = 0.42, C: float = 0.11) -> str:
    return oklch_to_hex(L, C, hue)[0]


def frame(body: str, hue: float) -> str:
    bg = oklch_to_hex(0.965, 0.02, hue)[0]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img">'
        f'<rect width="{W}" height="{H}" fill="{bg}"/>{body}</svg>\n'
    )


def foundations(h):
    c, faint = ink(h), ink(h, 0.86, 0.02)
    g = "".join(f'<path d="M{x} 0V{H}" stroke="{faint}"/>' for x in range(0, W, 16))
    g += "".join(f'<path d="M0 {y}H{W}" stroke="{faint}"/>' for y in range(0, H, 16))
    pts = " ".join(
        f"{x},{170 - 130 * (1 - math.exp(-x / 90)):.1f}" for x in range(24, 300, 6)
    )
    steps = " ".join(
        f"{x},{170 - 40 * (1 + (x - 24) // 70):.1f}" for x in range(24, 300, 2)
    )
    return g + (
        f'<polyline points="{steps}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="4 4" opacity="0.6"/>'
        f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3"/>'
        f'<path d="M24 20V172H300" fill="none" stroke="{c}" stroke-width="2"/>'
    )


def text(h):
    c, lite = ink(h), ink(h, 0.8, 0.06)
    rnd = random.Random(1)
    out = ""
    y = 26
    while y < H - 16:
        x = 28
        while x < W - 40:
            w = rnd.choice([18, 26, 34, 42, 58])
            if x + w > W - 28:
                break
            col = c if rnd.random() < 0.18 else lite
            out += (
                f'<rect x="{x}" y="{y}" width="{w}" height="9" rx="4.5" fill="{col}"/>'
            )
            x += w + 7
        y += 20
    return out


def retrieval(h):
    c, lite = ink(h), ink(h, 0.75, 0.05)
    rnd = random.Random(7)
    pts = [(rnd.uniform(20, W - 20), rnd.uniform(18, H - 18)) for _ in range(90)]
    q = (176, 98)
    near = sorted(pts, key=lambda p: (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)[:5]
    out = f'<circle cx="{q[0]}" cy="{q[1]}" r="52" fill="none" stroke="{c}" stroke-dasharray="3 4"/>'
    out += "".join(
        f'<path d="M{q[0]} {q[1]}L{x:.1f} {y:.1f}" stroke="{c}" stroke-width="1.5"/>'
        for x, y in near
    )
    out += "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{c if (x, y) in near else lite}"/>'
        for x, y in pts
    )
    return (
        out
        + f'<rect x="{q[0] - 6}" y="{q[1] - 6}" width="12" height="12" fill="{c}" transform="rotate(45 {q[0]} {q[1]})"/>'
    )


def agents(h):
    c, lite = ink(h), ink(h, 0.8, 0.06)
    cx, cy, r = 160, 100, 58
    nodes = [
        (cx + r * math.cos(a), cy + r * math.sin(a))
        for a in [i * 2 * math.pi / 5 - math.pi / 2 for i in range(5)]
    ]
    out = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{lite}" stroke-width="10"/>'
    out += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="6 6"/>'
    for i, (x, y) in enumerate(nodes):
        out += f'<rect x="{x - 13:.1f}" y="{y - 13:.1f}" width="26" height="26" rx="6" fill="{c if i == 0 else "#ffffff"}" stroke="{c}" stroke-width="2"/>'
    for x, y in [(36, 40), (284, 46), (40, 164), (280, 160)]:
        out += f'<circle cx="{x}" cy="{y}" r="7" fill="{lite}"/><path d="M{x} {y}L{cx} {cy}" stroke="{lite}" stroke-width="1.5"/>'
    return out


def realtime(h):
    c, lite = ink(h), ink(h, 0.78, 0.06)
    out = ""
    for i, x in enumerate(range(20, W - 16, 8)):
        amp = 70 * abs(math.sin(i / 4.3)) * (0.35 + 0.65 * abs(math.sin(i / 11)))
        col = c if 14 < i < 24 else lite
        out += f'<rect x="{x}" y="{100 - amp / 2:.1f}" width="5" height="{max(4, amp):.1f}" rx="2.5" fill="{col}"/>'
    return (
        out + f'<path d="M132 22V178M196 22V178" stroke="{c}" stroke-dasharray="3 4"/>'
    )


def generation(h):
    rnd = random.Random(3)
    out = ""
    s = 16
    for y in range(0, H, s):
        for x in range(0, W, s):
            t = (x / W + y / H) / 2 + rnd.uniform(-0.12, 0.12)
            L = 0.92 - 0.5 * max(0, min(1, t))
            out += f'<rect x="{x}" y="{y}" width="{s - 2}" height="{s - 2}" rx="2" fill="{oklch_to_hex(L, 0.12, h + 40 * t)[0]}"/>'
    return out


def platform(h):
    c, lite = ink(h), ink(h, 0.85, 0.04)
    rnd = random.Random(11)
    out = ""
    for row in range(4):
        for col in range(8):
            x, y = 26 + col * 35, 22 + row * 41
            busy = rnd.random() < 0.32
            out += f'<rect x="{x}" y="{y}" width="28" height="33" rx="4" fill="{c if busy else lite}"/>'
            if busy:
                out += f'<rect x="{x + 5}" y="{y + 25}" width="18" height="3" rx="1.5" fill="#ffffff" opacity="0.7"/>'
    return out


def capstones(h):
    c, lite = ink(h), ink(h, 0.83, 0.05)
    out = ""
    for i in range(5):
        x, y = 64 + i * 14, 34 + i * 18
        out += f'<rect x="{x}" y="{y}" width="168" height="96" rx="10" fill="{"#ffffff" if i < 4 else lite}" stroke="{c}" stroke-width="2"/>'
    return (
        out
        + f'<path d="M134 140H216M134 154H190" stroke="{c}" stroke-width="5" stroke-linecap="round"/>'
    )


DRAW = {
    "foundations": foundations,
    "text": text,
    "retrieval": retrieval,
    "agents": agents,
    "realtime": realtime,
    "generation": generation,
    "platform": platform,
    "capstones": capstones,
}


def main() -> int:
    files = {
        OUT / f"{name}.svg": frame(fn(PARTS[name][0]), PARTS[name][0])
        for name, fn in DRAW.items()
    }
    if "--check" in sys.argv:
        stale = [p for p, t in files.items() if not p.exists() or p.read_text() != t]
        for p in stale:
            print(f"covers: stale {p.relative_to(ROOT)}; run `make covers`")
        if stale:
            return 1
        print(f"covers: {len(files)} covers checked, all current.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    for p, t in files.items():
        p.write_text(t)
    print(f"covers: wrote {len(files)} covers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
