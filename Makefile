.PHONY: install test lint demo clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=rsna_knee --cov-report=term-missing

lint:
	ruff check .

demo:
	python examples/beginner_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage build dist *.egg-info src/*.egg-info
