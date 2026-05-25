.PHONY: install dev data train predict api test lint format clean docker_build docker_run docker_api docs

# Python and tools from virtual environment
PYTHON = .venv/bin/python
UV = UV_CACHE_DIR=.venv/.uv-cache .venv/bin/uv
PIP = .venv/bin/pip
PYTEST = .venv/bin/pytest
RUFF = .venv/bin/ruff
PRECOMMIT = .venv/bin/pre-commit
MKDOCS = .venv/bin/mkdocs

# Note: 'uv' is a faster alternative to pip. Install with: pip install uv
# Then replace 'pip install' with 'uv pip install' in the commands below.

install:
	if [ -x .venv/bin/uv ]; then $(UV) pip install -U pip; else $(PIP) install -U pip; fi
	if [ -x .venv/bin/uv ]; then $(UV) pip install -r requirements.txt; else $(PIP) install -r requirements.txt; fi
	if [ -x .venv/bin/uv ]; then $(UV) pip install -e .; else $(PIP) install -e .; fi

dev: install
	if [ -x .venv/bin/uv ]; then $(UV) pip install -r requirements_dev.txt; else $(PIP) install -r requirements_dev.txt; fi
	$(PRECOMMIT) install

data:
	$(PYTHON) -m se_489_mlops_project.data.make_dataset

train:
	$(PYTHON) -m se_489_mlops_project.train_model

predict:
	$(PYTHON) -m se_489_mlops_project.predict_model --input data/processed/processed_data.csv --output predictions.csv

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

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

docker_api:
	docker compose up api

docs:
	$(MKDOCS) build -f docs/mkdocs.yml
