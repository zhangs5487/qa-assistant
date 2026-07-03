.PHONY: help install dev-install test lint format clean-pyc

# ---- Setup ----

install:
	pip install -e .

dev-install:
	pip install -e ".[dev,crawler,cleaner,storage]"

install-openai:
	pip install -e ".[llm-openai]"

# ---- Testing ----

test:
	pytest -v

test-cov:
	pytest -v --cov --cov-report=html

test-unit:
	pytest -v tests/

# ---- Linting ----

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

# ---- Data Pipeline ----

bootstrap-db:
	python scripts/bootstrap_db.py

crawl:
	python scripts/run_crawl.py

pipeline:
	python scripts/run_pipeline.py

# ---- Utility ----

clean-pyc:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

clean-data:
	rm -rf data/raw/* data/clean/* data/exports/*
