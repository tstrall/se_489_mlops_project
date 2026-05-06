.PHONY: install dev data train predict test lint format clean docker_build docker_run docs

# Python and tools from virtual environment
PYTHON = .venv/bin/python
UV = UV_CACHE_DIR=.venv/.uv-cache .venv/bin/uv
PYTEST = .venv/bin/pytest
RUFF = .venv/bin/ruff
PRECOMMIT = .venv/bin/pre-commit
MKDOCS = .venv/bin/mkdocs

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	$(UV) pip install -U pip
	$(UV) pip install -r requirements.txt
	$(UV) pip install -e .

dev: install
	$(UV) pip install -r requirements_dev.txt
	$(PRECOMMIT) install

data:
	$(PYTHON) -m se_489_mlops_project.data.make_dataset

train:
	$(PYTHON) -m se_489_mlops_project.train_model

predict:
	$(PYTHON) -m se_489_mlops_project.predict_model --input data/processed/processed_data.csv --output predictions.csv

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check .
	$(RUFF) format --check .

format:
	$(RUFF) check --fix .
	$(RUFF) format .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

docker_build:
	docker build -t se_489_mlops_project -f dockerfiles/Dockerfile .

docker_run:
	docker run --rm se_489_mlops_project

docs:
	$(MKDOCS) build -f docs/mkdocs.yml
