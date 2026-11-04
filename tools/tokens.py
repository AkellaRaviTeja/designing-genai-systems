"""Design tokens for the book site, in one place.

Writes docs/design/tokens.json (the source of truth, Bearing token schema) and
theme/_tokens.scss (the binding Quarto's SCSS reads). Colours are defined in oklch;
sRGB hex is computed here, with chroma reduced until the colour fits the gamut.

  python3 tools/tokens.py           write both files
  python3 tools/tokens.py --check   fail if either file is stale
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKENS = ROOT / "docs" / "design" / "tokens.json"
SCSS = ROOT / "theme" / "_tokens.scss"
SCSS_DARK = ROOT / "theme" / "_tokens-dark.scss"


def oklch_to_hex(L: float, C: float, h: float) -> tuple[str, float]:
    """Hex for an oklch colour, lowering chroma until it is inside sRGB. Returns the
    hex and the chroma actually used."""
    while True:
        a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
        l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
        m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
        s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
        rgb = (
            4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
        )
        if all(-1e-4 <= x <= 1 + 1e-4 for x in rgb) or C <= 0:
            break
        C = max(0.0, C - 0.002)

    def gamma(x: float) -> int:
        x = min(1.0, max(0.0, x))
        x = 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
        return round(x * 255)

    return "#" + "".join(f"{gamma(x):02x}" for x in rgb), C


def colour(L: float, C: float, h: float) -> dict:
    hx, used = oklch_to_hex(L, C, h)
    return {"oklch": f"oklch({L:.3f} {used:.3f} {h:g})", "hex": hx}


STEPS = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000]
LIGHTNESS = [
    0.995,
    0.982,
    0.967,
    0.928,
    0.872,
    0.76,
    0.655,
    0.555,
    0.455,
    0.36,
    0.27,
    0.215,
    0.17,
]


def scale(chroma_peak: float, hue: float, flat: bool = False) -> dict:
    out = {}
    for step, L in zip(STEPS, LIGHTNESS):
        # Chroma peaks in the middle of the scale and falls towards white and black.
        c = chroma_peak if flat else chroma_peak * (1 - abs(L - 0.6) / 0.6) ** 0.7
        out[str(step)] = colour(L, c, hue)
    return out


# A cool neutral at very low chroma, and a vermilion accent kept from the diagrams.
PRIMITIVES = {
    "neutral": scale(0.005, 286, flat=True),
    "accent": scale(0.19, 38),
    "success": scale(0.14, 150),
    "warning": scale(0.14, 75),
    "danger": scale(0.18, 25),
    "info": scale(0.14, 255),
}

LIGHT = {
    "bg": "neutral.50",
    "bg-subtle": "neutral.100",
    "surface": "neutral.0",
    "surface-raised": "neutral.0",
    "overlay": "neutral.0",
    "text": "neutral.900",
    "text-muted": "neutral.600",
    "text-disabled": "neutral.400",
    "link": "neutral.900",
    "border": "neutral.200",
    "border-strong": "neutral.500",
    "accent": "accent.600",
    "accent-hover": "accent.700",
    "accent-active": "accent.800",
    "on-accent": "neutral.0",
    "accent-subtle": "accent.50",
    "success": "success.700",
    "on-success": "neutral.0",
    "success-subtle": "success.50",
    "warning": "warning.700",
    "on-warning": "neutral.0",
    "warning-subtle": "warning.50",
    "danger": "danger.700",
    "on-danger": "neutral.0",
    "danger-subtle": "danger.50",
    "info": "info.700",
    "on-info": "neutral.0",
    "info-subtle": "info.50",
    "focus": "neutral.900",
    "selection": "info.100",
    # Book-specific roles.
    "ink-panel": "neutral.950",
    "on-ink-panel": "neutral.100",
    "chip": "neutral.100",
}

DARK = {
    "bg": colour(0.175, 0.005, 286),
    "bg-subtle": colour(0.205, 0.005, 286),
    "surface": colour(0.215, 0.005, 286),
    "surface-raised": colour(0.25, 0.006, 286),
    "overlay": colour(0.27, 0.006, 286),
    "text": colour(0.945, 0.004, 286),
    "text-muted": colour(0.74, 0.006, 286),
    "text-disabled": colour(0.52, 0.006, 286),
    "link": colour(0.945, 0.004, 286),
    "border": colour(0.31, 0.006, 286),
    "border-strong": colour(0.56, 0.006, 286),
    "accent": colour(0.72, 0.15, 40),
    "accent-hover": colour(0.78, 0.13, 40),
    "accent-active": colour(0.84, 0.11, 40),
    "on-accent": colour(0.17, 0.005, 286),
    "accent-subtle": colour(0.26, 0.04, 40),
    "success": colour(0.75, 0.12, 150),
    "on-success": colour(0.17, 0.005, 286),
    "success-subtle": colour(0.26, 0.04, 150),
    "warning": colour(0.8, 0.12, 75),
    "on-warning": colour(0.17, 0.005, 286),
    "warning-subtle": colour(0.27, 0.04, 75),
    "danger": colour(0.72, 0.14, 25),
    "on-danger": colour(0.17, 0.005, 286),
    "danger-subtle": colour(0.26, 0.04, 25),
    "info": colour(0.75, 0.11, 255),
    "on-info": colour(0.17, 0.005, 286),
    "info-subtle": colour(0.26, 0.04, 255),
    "focus": colour(0.945, 0.004, 286),
    "selection": colour(0.35, 0.06, 255),
    "ink-panel": colour(0.13, 0.005, 286),
    "on-ink-panel": colour(0.92, 0.004, 286),
    "chip": colour(0.27, 0.006, 286),
}

# Part colours for the cards on the home page and the chapter openers. Pastels in light,
# deep and quiet in dark; text on them is always the text role.
PARTS = {
    "foundations": (286, 0.03),
    "text": (200, 0.06),
    "retrieval": (165, 0.06),
    "agents": (300, 0.06),
    "realtime": (255, 0.06),
    "generation": (75, 0.07),
    "platform": (30, 0.06),
    "capstones": (120, 0.05),
}
for name, (h, c) in PARTS.items():
    LIGHT[f"part-{name}"] = colour(0.88, c, h)
    DARK[f"part-{name}"] = colour(0.3, c * 0.6, h)

FONT = {
    "families": {
        "display": "'Overused Grotesk', 'Helvetica Neue', system-ui, sans-serif",
        "body": "'Overused Grotesk', 'Helvetica Neue', system-ui, sans-serif",
        "mono": "'IBM Plex Mono', ui-monospace, 'SF Mono', Menlo, monospace",
        # Display accent only: chapter numbers, hero words, sidebar numerals. Never running text.
        "pixel": "'Departure Mono', 'IBM Plex Mono', ui-monospace, monospace",
    },
    "sizes": {"xs": 12, "sm": 13, "md": 15, "lg": 17, "xl": 24, "2xl": 40, "hero": 64},
    "weights": {"regular": 400, "medium": 500, "semibold": 600},
    "lineHeights": {"tight": 1.04, "snug": 1.25, "body": 1.62, "mono": 1.55},
    "letterSpacing": {"hero": "-0.045em", "heading": "-0.025em", "body": "-0.003em", "label": "0.12em"},
    "roles": {
        "display": {"family": "display", "size": "2xl", "weight": "medium", "lineHeight": "tight", "tracking": "hero"},
        "title": {"family": "display", "size": "xl", "weight": "semibold", "lineHeight": "snug", "tracking": "heading"},
        "body": {"family": "body", "size": "lg", "weight": "regular", "lineHeight": "body", "tracking": "body"},
        "label": {"family": "mono", "size": "xs", "weight": "medium", "lineHeight": "mono", "tracking": "label"},
        "code": {"family": "mono", "size": "sm", "weight": "regular", "lineHeight": "mono", "tracking": "body"},
        "hero": {"family": "display", "size": "hero", "weight": "medium", "lineHeight": "tight", "tracking": "hero"},
    },
}

TOKENS_DOC = {
    "meta": {
        "name": "Designing GenAI Systems",
        "version": 1,
        "source": "brief",
        "memorable": "A calm grey workbench where each system is a card you open, and every label reads like an instrument.",
    },
    "color": {"primitives": PRIMITIVES, "roles": {"light": LIGHT, "dark": DARK}},
    "font": FONT,
    "space": {
        "base": 8,
        "half": 4,
        "scale": {
            "0": 0,
            "1": 4,
            "2": 8,
            "3": 16,
            "4": 24,
            "5": 32,
            "6": 48,
            "7": 64,
            "8": 96,
        },
    },
    "radius": {"control": 8, "container": 14, "overlay": 18, "full": 999},
    "elevation": {
        "rule": "Elevation above 0 means the element is pressable or it overlays the page. A static block never carries a shadow.",
        "levels": {
            "0": {"light": "none", "dark": "none"},
            "1": {
                "light": "0 1px 2px #3b3b4317, 0 0 0 1px #3b3b4308",
                "dark": "0 0 0 1px #ffffff10",
            },
            "2": {
                "light": "0 2px 4px #3b3b430d, 0 6px 14px #3b3b431a, 0 0 0 1px #3b3b4308",
                "dark": "0 4px 14px #00000066, 0 0 0 1px #ffffff14",
            },
            "3": {
                "light": "0 4px 8px #3b3b4314, 0 16px 32px #3b3b4324, 0 0 0 1px #3b3b4308",
                "dark": "0 12px 32px #000000a0, 0 0 0 1px #ffffff18",
            },
        },
    },
    "motion": {
        "duration": {
            "instant": 80,
            "fast": 150,
            "base": 240,
            "slow": 400,
            "deliberate": 700,
        },
        "easing": {
            "standard": "cubic-bezier(0.2, 0, 0, 1)",
            "enter": "cubic-bezier(0, 0, 0, 1)",
            "exit": "cubic-bezier(0.3, 0, 1, 1)",
            "spring": {"stiffness": 300, "damping": 30, "mass": 1},
        },
        "stagger": 40,
    },
    "breakpoint": {"sm": 576, "md": 768, "lg": 992, "xl": 1400},
    "z": {
        "base": 0,
        "raised": 10,
        "sticky": 100,
        "overlay": 1000,
        "modal": 1100,
        "toast": 1200,
    },
    "focus": {"width": 2, "offset": 2, "role": "focus"},
}


def resolve(ref) -> dict:
    if isinstance(ref, dict):
        return ref
    scale_name, step = ref.split(".")
    return PRIMITIVES[scale_name][step]


def scss(mode: str) -> str:
    roles = LIGHT if mode == "light" else DARK
    head = [
        "// Generated by tools/tokens.py from docs/design/tokens.json. Do not edit.",
        f"// {mode.capitalize()} mode. The light file sets defaults; the dark theme loads the dark",
        "// file after it, and its plain assignments win.",
        "",
        "/*-- scss:defaults --*/",
    ]
    bang = " !default" if mode == "light" else ""
    lines = [f"$t-{r}: {resolve(ref)['hex']}{bang};" for r, ref in roles.items()]
    for lvl, v in TOKENS_DOC["elevation"]["levels"].items():
        lines.append(f"$t-shadow-{lvl}: {v[mode]}{bang};")
    if mode == "light":
        for k, v in FONT["families"].items():
            lines.append(f"$t-font-{k}: {v} !default;")
        for k, v in TOKENS_DOC["radius"].items():
            lines.append(f"$t-radius-{k}: {v}px !default;")
        for k, v in FONT["sizes"].items():
            lines.append(f"$t-size-{k}: {v}px !default;")
        for k, v in FONT["letterSpacing"].items():
            lines.append(f"$t-tracking-{k}: {v} !default;")
    return "\n".join(head + lines) + "\n"


D2_THEME = ROOT / "diagrams" / "_theme.d2"


def d2_theme() -> str:
    """The diagram palette, from the same primitives as the site."""
    hx = lambda ref: resolve(ref)["hex"]  # noqa: E731
    c = lambda L, C, h: colour(L, C, h)["hex"]  # noqa: E731
    pal = {
        "ink": hx("neutral.900"), "muted": hx("neutral.600"), "line": hx("neutral.200"),
        "paper": hx("neutral.0"), "zone": hx("neutral.50"), "ext": hx("neutral.100"), "extline": hx("neutral.500"),
        "model": c(0.935, 0.035, 280), "modelline": c(0.47, 0.13, 280), "modelink": c(0.3, 0.09, 280),
        "store": c(0.94, 0.03, 175), "storeline": c(0.46, 0.08, 175), "storeink": c(0.3, 0.06, 175),
        "queue": c(0.945, 0.04, 80), "queueline": c(0.5, 0.09, 80), "queueink": c(0.33, 0.07, 80),
        "accent": hx("accent.600"), "offline": c(0.46, 0.08, 175),
    }
    sw = 'stroke-width: 2'
    cls = {
        "actor": f'shape: person\n    width: 56\n    style: {{font-size: 22; fill: "{pal["paper"]}"; stroke: "{pal["ink"]}"; {sw}; font-color: "{pal["ink"]}"}}',
        "svc": f'shape: rectangle\n    style: {{font-size: 22; fill: "{pal["paper"]}"; stroke: "{pal["ink"]}"; {sw}; border-radius: 8; font-color: "{pal["ink"]}"}}',
        "model": f'shape: hexagon\n    style: {{font-size: 22; fill: "{pal["model"]}"; stroke: "{pal["modelline"]}"; {sw}; font-color: "{pal["modelink"]}"}}',
        "store": f'shape: cylinder\n    style: {{font-size: 22; fill: "{pal["store"]}"; stroke: "{pal["storeline"]}"; {sw}; font-color: "{pal["storeink"]}"}}',
        "queue": f'shape: queue\n    style: {{font-size: 22; fill: "{pal["queue"]}"; stroke: "{pal["queueline"]}"; {sw}; font-color: "{pal["queueink"]}"}}',
        "ext": f'shape: rectangle\n    style: {{font-size: 22; fill: "{pal["ext"]}"; stroke: "{pal["extline"]}"; {sw}; stroke-dash: 4; border-radius: 8; font-color: "{pal["ink"]}"}}',
        "zone": f'style: {{fill: "{pal["zone"]}"; stroke: "{pal["line"]}"; stroke-width: 1; border-radius: 14; font-color: "{pal["muted"]}"; bold: false; font-size: 24}}',
        "note": f'shape: page\n    style: {{fill: "{pal["paper"]}"; stroke: "{pal["line"]}"; stroke-width: 1; font-color: "{pal["muted"]}"; font-size: 13}}',
        "dot": f'shape: circle\n    width: 12\n    style: {{fill: "{pal["ink"]}"; stroke: "{pal["ink"]}"}}',
        "new": f'style: {{stroke: "{pal["accent"]}"; stroke-width: 3; bold: true}}',
        "gone": 'style: {opacity: 0.3; stroke-dash: 3}',
        "async": f'style: {{font-size: 19; stroke: "{pal["muted"]}"; stroke-dash: 5; font-color: "{pal["muted"]}"}}',
        "offline": f'style: {{font-size: 19; stroke: "{pal["offline"]}"; stroke-dash: 2; font-color: "{pal["offline"]}"}}',
        "hot": f'style: {{font-size: 19; stroke: "{pal["accent"]}"; stroke-width: 3; font-color: "{pal["accent"]}"}}',
    }
    body = "\n".join(f"  {k}: {{\n    {v}\n  }}" for k, v in cls.items())
    return (
        "# Generated by tools/tokens.py: the diagram palette shares the site's primitives. Do not edit.\n"
        "# Import with:  ...@../_theme   (see STYLE.md for what each class means)\n\n"
        "vars: {\n  d2-config: {\n    layout-engine: dagre\n    theme-id: 1\n    pad: 28\n  }\n}\n\n"
        f"classes: {{\n{body}\n}}\n"
    )


def main() -> int:
    files = {TOKENS: json.dumps(TOKENS_DOC, indent=2) + "\n", SCSS: scss("light"), SCSS_DARK: scss("dark"), D2_THEME: d2_theme()}
    if "--check" in sys.argv:
        stale = [p for p, t in files.items() if not p.exists() or p.read_text() != t]
        for p in stale:
            print(f"tokens: stale {p.relative_to(ROOT)}; run `make tokens`")
        n = len(LIGHT) + len(DARK)
        if stale:
            return 1
        print(f"tokens: {n} roles across two modes checked, both outputs current.")
        return 0
    TOKENS.parent.mkdir(parents=True, exist_ok=True)
    for p, t in files.items():
        p.write_text(t)
    print(
        f"tokens: wrote {TOKENS.relative_to(ROOT)} and {SCSS.relative_to(ROOT)} ({len(LIGHT)} roles per mode)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
