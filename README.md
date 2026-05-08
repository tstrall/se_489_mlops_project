# SE 489 MLOps Project - HelpEvents

Predict SLA violations from event sequences

## Team Information

- **Project Lead:** Ted Strall (tstrall@depaul.edu)
- **Team Members:**
  - Calvin Au (cau@depaul.edu)
  - Seshagiri Kalyana Venkatesh Adavi (sadavi@depaul.edu)
  - Julisa Delfin (jdelfing@depaul.edu)

## Project Overview

### Problem Statement

IT helpdesk teams operate under Service Level Agreements (SLAs) that define the maximum time allowed to resolve a support ticket. When tickets breach these thresholds, the consequences range from contractual penalties to degraded customer trust. Currently, support teams have no early-warning system — they only discover an SLA violation after it has already occurred. The goal of this project is to predict, as early as possible in a ticket's lifecycle, whether that ticket is likely to violate its SLA, so support managers can intervene proactively.

### What We Are Building

HelpEvents is an end-to-end MLOps pipeline that trains a binary classification model on real-world helpdesk data. Given a set of features about a support ticket — its priority, type, number of contributing agents, comment activity, and the history of workflow state changes — the model predicts whether the ticket will exceed the 7-day SLA threshold.

The raw dataset comes from a public Mendeley repository and contains over 150,000 helpdesk tickets spanning 2016 to 2023, including a change-history table that records every workflow event associated with each ticket. We transform these event sequences into structured, per-ticket features using a deterministic preprocessing pipeline. Feature engineering steps include computing ticket event density (events per active day), contributor-to-comment ratios, log-scaled resolution time, and binary priority flags. All transformations are implemented as reusable Python functions so the same logic runs identically at training time and inference time.

### Model and Framework

The classifier is a `RandomForestClassifier` wrapped in a scikit-learn `Pipeline` with a `StandardScaler` preprocessing stage. Random Forest was chosen because it handles mixed feature types naturally, is robust to outliers, and produces well-calibrated probability estimates that are critical for ranking tickets by violation risk. We evaluate with accuracy, precision, recall, F1, and ROC-AUC.

The key third-party framework integrated into this project is **MLflow**. Every training run logs its hyperparameters, evaluation metrics, and the serialized model artifact to a local MLflow tracking server. This makes experiments fully reproducible and comparable — a core MLOps requirement.

### Expected Impact

A deployed version of this model would allow a support team to flag high-risk tickets in real time and prioritize them for escalation, directly reducing the rate of SLA violations. Success is measured primarily by ROC-AUC (ability to rank tickets by risk) and recall (minimizing missed violations).

**Key Objectives:**
- [x] Transform event sequences into structured features for modeling
- [x] Build a reproducible machine learning training pipeline
- [x] Track experiments and model performance using MLflow

## Dataset

Source: https://data.mendeley.com/datasets/btm76zndnt/2

The dataset contains helpdesk ticket data including:
- ticket metadata
- interaction events
- timestamps for messages and updates

### Target Variable

SLA violation is defined as:

SLA_violation = 1 if resolution_time > SLA_threshold

## Architecture Diagram

```text
Raw Ticket Events
↓
Data Preprocessing (group by ticket_id)
↓
Feature Engineering (sequence → features)
↓
Model Training (scikit-learn)
↓
Evaluation + MLflow Tracking
```

## Phase Deliverables

### Phase 1: Project Design & Model Development
- See [PHASE1.md](PHASE1.md) for detailed checklist

### Phase 2: Containerization & Monitoring
- See [PHASE2.md](PHASE2.md) for detailed checklist

### Phase 3: CI/CD & Deployment
- See [PHASE3.md](PHASE3.md) for detailed checklist

## Setup Instructions

### Prerequisites
- Python 3.11+ installed
- Git installed
- (Optional) Docker and Docker Compose

### Installation

**Option 1: Using uv (recommended - faster)**
```bash
pip install uv
uv pip install -r requirements.txt
```

**Option 2: Using pip**
```bash
pip install -U pip
pip install -r requirements.txt
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

### Running the Pipeline

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict

# See all available commands
make help
```

## Technology Stack

### Core Dependencies
- **numpy** >= 1.26.0 - Numerical computing
- **pandas** >= 2.2.0 - Data manipulation
- **scikit-learn** >= 1.5.0 - Machine learning algorithms
- **matplotlib** >= 3.9.0 - Visualization
- **tqdm** >= 4.66.0 - Progress bars
- **pyyaml** >= 6.0 - Configuration files
### Experiment Tracking
- **mlflow** >= 2.16.0 - MLflow experiment tracking

### Development Tools
- **pytest** >= 8.0 - Testing framework
- **pytest-cov** >= 5.0 - Code coverage
- **ruff** >= 0.6.0 - Linting and formatting
- **mypy** >= 1.11 - Static type checking
- **pre-commit** >= 3.8 - Git hooks framework

## Project Structure

This template uses the modern **`src/` layout** — the importable package lives in `src/se_489_mlops_project/`, decoupled from the repository root. That forces `pip install -e .` before imports work, which catches packaging bugs early.

```
se_489_mlops_project/                  # Repository root
├── src/
│   └── se_489_mlops_project/          # Importable Python package
│       ├── __init__.py                # Version + package metadata
│       ├── config.py                  # Paths & typed config (PROJECT_ROOT, TrainingConfig, ...)
│       ├── logging_config.py          # setup_logging() + get_logger()
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py             # load_raw / load_processed / save_processed
│       │   └── make_dataset.py        # Raw → processed pipeline CLI
│       ├── features/
│       │   ├── __init__.py
│       │   └── build_features.py      # Feature engineering
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py                # BaseModel ABC (fit/predict/save/load)
│       │   └── model.py               # Concrete Model scaffold
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py             # classification_report, regression_report
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── visualize.py           # Plot helpers
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py                  # JSON helpers
│       │   └── seed.py                # set_seed for reproducibility
│       ├── train_model.py             # Training CLI
│       └── predict_model.py           # Inference CLI
├── tests/                             # Unit and integration tests
│   ├── conftest.py
│   └── test_model.py
├── data/
│   ├── raw/                           # Immutable raw data
│   └── processed/                     # Cleaned, transformed data
├── models/                            # Trained model artifacts (.joblib)
├── notebooks/                         # Jupyter notebooks for exploration
├── reports/
│   └── figures/                       # Generated analysis and figures
├── docs/                              # MkDocs documentation
│   ├── mkdocs.yml
│   ├── index.md
│   ├── getting_started.md
│   └── api.md
├── dockerfiles/                       # Docker configuration
│   └── Dockerfile
├── configs/                           # Hydra configuration (if selected)
│   └── config.yaml
├── api/                               # FastAPI service (if selected)
├── .github/workflows/                 # GitHub Actions CI/CD
│   └── ci.yml
├── PHASE1.md                          # Phase 1 deliverables checklist
├── PHASE2.md                          # Phase 2 deliverables checklist
├── PHASE3.md                          # Phase 3 deliverables checklist
├── .pre-commit-config.yaml            # Pre-commit hooks (Ruff, mypy)
├── Makefile                           # Common commands
├── docker-compose.yaml                # Docker Compose setup
├── pyproject.toml                     # Project config & dependencies
├── requirements.txt                   # Runtime dependencies
├── requirements_dev.txt               # Development dependencies
├── LICENSE
└── README.md
```

### Why `src/` layout?

| | `src/` layout (this template) | Flat layout |
|---|---|---|
| Forces `pip install -e .` before import | ✅ | ❌ |
| Catches packaging bugs early | ✅ | ❌ |
| Adopted by | attrs, httpx, pydantic, flask, sqlalchemy | Older data-science templates |

Data and model artifacts are accessed via the constants in `se_489_mlops_project.config` (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, …) rather than relative paths — code is independent of where you invoke it from.

## Common Commands

```bash
# Install package + runtime dependencies (editable install)
make install

# Install dev tools + pre-commit hooks
make dev

# Run linting and formatting checks
make lint

# Auto-format code
make format

# Run tests
make test

# Clean up build artifacts
make clean

# Docker operations
make docker_build
make docker_run

# Serve documentation locally
make docs
```

## Contribution Summary

- [ ] Team members have been assigned
- [x] Development environment has been set up
- [x] Initial data exploration completed
- [ ] Model baseline established
- [ ] Evaluation metrics defined
- [x] Documentation updated
- [ ] All tests passing
- [ ] Code reviewed and merged

## References

- [Project Documentation](docs/index.md)
- [Phase 1 — Project Design & Model Development](PHASE1.md)
- [Phase 2 — Containerization & Monitoring](PHASE2.md)
- [Phase 3 — CI/CD & Deployment](PHASE3.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
