# VaaniTriage task runner. Run each target inside the matching virtual env:
#   .venv-torch  -> data, baseline, teacher, LLM eval
#   .venv-tf     -> student, distillation, TFLite
PY ?= python

.PHONY: help test lint format labels smoke report

help:
	@echo "make test    - run all tests (framework-specific tests skip if not installed)"
	@echo "make lint    - ruff check + format check"
	@echo "make format  - apply ruff fixes and formatting"
	@echo "make labels  - export data/labels.json"
	@echo "make smoke   - run the whole pipeline on ~200 examples (CPU, < 2 min per step)"
	@echo "make report  - rebuild results/RESULTS.md from results/*/metrics.json"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

labels:
	$(PY) -m vaani.labels --out data/labels.json

# Steps are appended phase by phase as they are implemented.
smoke: labels
	@echo "smoke: phase 0 only (labels export) - later phases add their --smoke steps here"

report:
	$(PY) -m vaani.eval.report
