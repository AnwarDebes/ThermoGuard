PYTHON ?= python3
PYTHONPATH := .

.PHONY: all install test reproduce figures paper clean help

help:
	@echo "Targets:"
	@echo "  install     -- pip install -e ."
	@echo "  test        -- run unit tests (~4s)"
	@echo "  reproduce   -- run all 3 experiments (~3 min on CPU)"
	@echo "  figures     -- regenerate paper/figures/*.{png,pdf}"
	@echo "  paper       -- build paper/thermoguard.pdf"
	@echo "  all         -- reproduce + figures + paper"
	@echo "  clean       -- delete build artefacts"

install:
	pip install -e .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q tests/

reproduce: results/forecast_accuracy.json results/risk_stratification.json results/triage.json

results/forecast_accuracy.json: experiments/forecast_accuracy.py thermoguard/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/forecast_accuracy.py

results/risk_stratification.json: experiments/risk_stratification.py thermoguard/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/risk_stratification.py

results/triage.json: experiments/triage.py thermoguard/*.py
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) experiments/triage.py

figures: reproduce
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/make_figures.py

paper: figures
	cd paper && pdflatex -interaction=nonstopmode thermoguard.tex \
	  && bibtex thermoguard \
	  && pdflatex -interaction=nonstopmode thermoguard.tex \
	  && pdflatex -interaction=nonstopmode thermoguard.tex

all: paper

clean:
	rm -f paper/*.aux paper/*.log paper/*.bbl paper/*.blg paper/*.out paper/*.toc
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
