PYTHON ?= python3
D2 ?= d2
QUARTO ?= quarto

D2_SOURCES := $(wildcard diagrams/*/*.d2)

# Design lint (Bearing design-system). Gates colour literals only; the other rules
# (spacing font-size shadow raw family) are available but were not agreed.
export DESIGN_LINT_ROOTS := theme
export DESIGN_LINT_EXCLUDE := (^|/)_tokens(-dark)?\.scss$$
export DESIGN_LINT_RULES := colour
SVGS := $(patsubst diagrams/%.d2,images/%.svg,$(D2_SOURCES))

.PHONY: help all tokens covers diagrams costs lint design-lint check build pdf preview clean

help:
	@echo "make tokens     regenerate docs/design/tokens.json, the SCSS bindings and the diagram theme"
	@echo "make covers     redraw the home page cover art"
	@echo "make diagrams   render D2 sources in diagrams/ to SVG in images/"
	@echo "make costs      regenerate cost tables and inline figures from data/prices.toml"
	@echo "make lint       prose lint and chapter structure check"
	@echo "make design-lint  fail on colour literals outside the token files"
	@echo "make check      lint, design lint, plus fail if anything generated is stale"
	@echo "make build      everything above, then render the HTML book to _book/"
	@echo "make pdf        render the PDF edition (Typst)"
	@echo "make preview    live preview while writing"

all: build

images/%.svg: diagrams/%.d2 diagrams/_theme.d2 diagrams/_post.d2
	@mkdir -p $(dir $@)
	cd $(dir $<) && cat $(notdir $<) ../_post.d2 | $(D2) --pad 24 - $(abspath $@)

diagrams: $(SVGS)
	@test -n "$(SVGS)" || (echo "diagrams: 0 sources found" && exit 1)
	@echo "diagrams: $(words $(SVGS)) diagrams current"

tokens:
	$(PYTHON) tools/tokens.py

covers:
	$(PYTHON) tools/covers.py

costs:
	$(PYTHON) tools/costgen.py

design-lint:
	sh scripts/design-lint.sh
	$(PYTHON) tools/check_d2_colours.py

lint:
	$(PYTHON) tools/lint_prose.py
	$(PYTHON) tools/check_structure.py

check: lint design-lint
	$(PYTHON) tools/tokens.py --check
	$(PYTHON) tools/covers.py --check
	$(PYTHON) tools/costgen.py --check
	$(PYTHON) tools/check_diagrams.py

build: tokens covers diagrams costs lint
	$(QUARTO) render --to html

pdf: diagrams costs
	$(QUARTO) render --to typst

preview: diagrams costs
	$(QUARTO) preview

clean:
	rm -rf _book .quarto
