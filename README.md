# Designing GenAI Systems

**50 architectures, from the first question to the monthly bill.** By Ravi Teja.

A book about designing systems built on generative models. Each chapter takes a vague request,
asks the questions that turn it into requirements, sizes the problem, builds the simplest version
that could work, breaks it with evidence, and fixes it one step at a time. Then it deploys the
result, prices it with real vendors at three scales, and makes it cheaper.

Read it online at `https://akellaraviteja.github.io/designing-genai-systems/` once Pages is enabled.

## What is in this edition

| Part | Chapters |
|---|---|
| Foundations | F1 The design method · F2 The numbers · F3 Choosing a model · F4 Evaluation as architecture · F5 The optimisation playbook (O1 to O20) · F6 Safety, privacy and observability |
| Retrieval | 07 Simple RAG over a company wiki ★★ |
| Real-time and conversational | 28 Call-centre voice agent ★★★★ |
| The model platform | 39 Self-hosted inference platform ★★★★ |
| Appendices | The roadmap of all fifty systems · Prices |

## How the book stays honest

- **Every price comes from one file.** `data/prices.toml` holds every vendor price with its source
  URL and the date it was checked. `tools/costgen.py` regenerates every cost table and every
  inline figure from it. Unverified entries cannot be used in a cost table.
- **Every number is arithmetic, a dated price, or a labelled rule of thumb.** Worked scenarios are
  marked as such.
- **Every chapter follows the same structure,** checked by `tools/check_structure.py`.
- **Prose is linted** for em and en dashes, filler words and American spellings by
  `tools/lint_prose.py`.
- **No colour literals.** Colours live in `docs/design/tokens.json`; the design lint fails the build
  on a hard-coded colour in the theme or a diagram.
- **Every gate fails on empty input** and prints how many things it checked.

## Build it

You need Python 3.11 or later (standard library only), [Quarto](https://quarto.org) 1.8 or later,
and [D2](https://d2lang.com) 0.7 or later.

```bash
make check      # lint, structure, stale costs, stale diagrams
make build      # render diagrams and costs, then the HTML book into _book/
make pdf        # the PDF edition, through Typst
make preview    # live preview while writing
```

## Layout

| Path | What it is |
|---|---|
| `index.qmd`, `foundations/`, `systems/`, `appendix/` | The chapters |
| `STYLE.md` | Voice, numbers, chapter structure and diagram rules. Read before writing |
| `data/prices.toml` | Every price, with source and date |
| `data/hardware.toml` | GPU datasheet figures and model shapes used in sizing |
| `costs/` | One scenario module per chapter; each builds that chapter's tables and figures |
| `_generated/`, `_variables.yml` | Output of `make costs`. Committed, and checked for staleness in CI |
| `diagrams/` | D2 sources. `_theme.d2` is the shared visual language, generated from the tokens |
| `docs/design/tokens.json` | Design tokens: colour roles for both modes, type, radius, elevation. `make tokens` regenerates the SCSS bindings and the diagram theme from `tools/tokens.py` |
| `assets/covers/` | Home page cover art, drawn by `tools/covers.py` |
| `images/` | Rendered SVGs. Committed, and checked for staleness in CI |
| `theme/` | `book.scss` (all styling, no colour literals), generated token bindings, the title-number script, a Lua filter |
| `tools/` | Cost, token and cover generators, and the gates |
| `scripts/design-lint.sh` | Bearing's design lint: `make design-lint` fails on colour literals in `theme/` outside the generated token files. `tools/check_d2_colours.py` does the same for diagram sources |

## Refreshing prices

1. Edit the entry in `data/prices.toml`: the value, `checked`, and `source` if it moved.
2. `make costs`, then `make check`.
3. Commit the data file and the regenerated output together.

## Contributing

Corrections are welcome as issues or pull requests, especially price updates with a source. Please
follow `STYLE.md`; `make check` must pass.

## Licence

Text and diagrams: [CC BY-NC-SA 4.0](LICENSE-TEXT.md). Code in `tools/`, `costs/` and `theme/`:
[MIT](LICENSE).
