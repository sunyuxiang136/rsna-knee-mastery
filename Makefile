# Makefile for rsna-knee-mastery
VENV ?= .venv
BIN = $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)

.PHONY: install test lint format format-check check demo clean coverage-html

install:
	$(BIN)pip install -e ".[dev]"

test:
	$(BIN)pytest --cov=rsna_knee --cov-report=term-missing

coverage-html:
	$(BIN)pytest --cov=rsna_knee --cov-report=html
	@echo "Coverage report written to htmlcov/index.html"

lint:
	$(BIN)ruff check .

format:
	$(BIN)ruff format .

format-check:
	$(BIN)ruff format --check .

demo:
	$(BIN)python examples/beginner_demo.py

check: lint format-check test demo
	@echo "All lint, formatting, test, and demo checks passed!"

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info src/*.egg-info
