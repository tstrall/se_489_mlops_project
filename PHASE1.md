# PHASE 1: Project Design & Model Development

## Overview
Phase 1 establishes the foundation for the MLOps project focused on predicting SLA violations from helpdesk event sequences. This phase includes project planning, code organization, dataset selection, baseline model development, and documentation. The goal is to create a reproducible machine learning pipeline with clear structure and experiment tracking.

---

## 1. Project Proposal

- [x] **Scope & Objectives**: Predict SLA violations using helpdesk event data and build a reproducible ML pipeline
- [x] **Detailed Description**: Project focuses on transforming event sequences into structured features and training classification models to predict SLA violations. The system emphasizes reproducibility through script-based pipelines and experiment tracking using MLflow.
- [x] **Dataset Selection**: Selected real-world helpdesk dataset from Mendeley
- [x] **Dataset Description**: Dataset includes ticket metadata, event interactions, and timestamps from 2016–2023
- [x] **Model Considerations**: Binary classification using Random Forest (scikit-learn) with a full sklearn Pipeline including StandardScaler preprocessing
- [x] **Open-Source Tools**: MLflow (third-party experiment tracking — not a course tool); scikit-learn, pandas, numpy for modeling and processing

---

## 2. Code Organization & Setup

- [x] **GitHub Repository**: Repository created using cookiecutter MLOps structure
- [x] **Environment Setup**: Python virtual environment configured
- [x] **Dependency Management**: requirements.txt and requirements_dev.txt defined
- [x] **Project Structure**: Organized using src/ layout with separation of concerns
- [x] **Version Pinning**: Core dependencies pinned to specific versions
- [x] **Installation Documentation**: Setup instructions included in README

---

## 3. Version Control & Collaboration

- [x] **Regular Commits**: Ongoing commits with descriptive messages
- [x] **Branching Strategy**: Using feature-based development workflow
- [x] **Pull Request Process**: `phase1-fixes` branch created; PR to main pending review
- [x] **Team Roles**: Ted Strall acting as project lead and developer
- [ ] **Code Review Guidelines**: Not applicable in Phase 1
- [x] **Commit History**: Maintained for traceability

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: Preprocessing implemented in scripts
- [x] **Normalization**: StandardScaler applied to numeric features inside the sklearn Pipeline
- [x] **Feature Engineering**: `build_features.py` derives `events_per_day`, `comments_per_contributor`, `is_high_priority`, `log_num_events`, and `total_time_days` from raw columns
- [ ] **Data Augmentation**: Not applicable for tabular classification
- [x] **Data Documentation**: Dataset source, schema, and preprocessing steps documented in README and `data/raw/README.md`
- [x] **Data Splits**: 80/20 stratified train/test split (random_state=42)
- [ ] **Data Validation**: Basic validation only (to be expanded in Phase 2)
- [ ] **DVC Setup (Optional)**: Not used (data managed locally)

---

## 5. Model Training

- [x] **Training Environment**: Local Python environment configured
- [x] **Baseline Model**: RandomForestClassifier (n_estimators=100, max_depth=10) inside sklearn Pipeline
- [x] **Hyperparameter Configuration**: Default baseline — n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42
- [x] **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC — all logged to MLflow
- [x] **Model Persistence**: Trained model saved to `models/model.joblib`; also logged as MLflow artifact
- [x] **Training Reproducibility**: Fixed random seed, deterministic preprocessing, script-based pipeline
- [x] **Performance Baseline** (first run on 13,339 test samples):

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9984 |
| Accuracy | 0.9850 |
| Precision | 0.9958 |
| Recall | 0.9830 |
| F1 Score | 0.9894 |

Class distribution — training set: 70.9% violations (1), 29.1% non-violations (0).

> **Note on data leakage:** An initial run with `wf_total_time` included produced perfect scores (1.0) because the SLA violation target is defined directly as `wf_total_time > 7 days`. This column and its log/day derivatives were removed from training features. Scores above reflect the corrected, non-leaky feature set.

---

## 6. Documentation & Reporting

- [x] **README**:
  - [x] Project overview and objectives
  - [x] Setup and installation instructions
  - [x] Quick start guide for running training
  - [x] Dependencies and requirements
  - [ ] Contributing guidelines
  - [x] License information

- [x] **Code Docstrings**: Basic docstrings included
- [x] **Code Style**: Ruff configured
- [x] **Type Hints**: Added in core modules
- [x] **Type Checking**: mypy configured
- [x] **Makefile**:
  - [x] `make setup`
  - [x] `make train`
  - [x] `make test`
  - [x] `make lint`
  - [x] `make format`

- [ ] **CONTRIBUTING.md**: Not yet created
- [ ] **API Documentation**: Not applicable in Phase 1

---

> **Checklist:** Phase 1 establishes a reproducible ML pipeline foundation with structured code, dataset selection, and baseline modeling. Remaining items will be completed in later phases.

---

## 7. Findings, Challenges, and Areas for Improvement

### Findings

The baseline Random Forest model achieves strong performance on the test set (ROC-AUC 0.9984, F1 0.9894) using only ticket metadata and event-history features. The most informative signals appear to be `num_events`, `processing_steps`, `duration_seconds`, and `events_per_day` — features that capture how actively a ticket is being worked at the time of prediction. The dataset is imbalanced (71% violations, 29% non-violations), which is handled naturally by stratified splitting.

The EDA notebook (`notebooks/01_eda.ipynb`) reveals:
- `wf_total_time` is heavily right-skewed with a long tail of multi-month tickets.
- Ticket priority has a notable effect on violation rate — certain priority tiers show significantly higher breach rates.
- The change history table averages around 4 events per ticket, with high-event tickets correlating strongly with SLA breaches.

### Challenges

**Data leakage:** The most significant issue encountered during Phase 1. The original feature set included `wf_total_time`, which is the direct input to the SLA violation definition. This produced perfect scores on the first run and had to be removed. This is a realistic MLOps concern — in production, `wf_total_time` would not be available at prediction time (you do not know the total resolution time until a ticket is closed). The corrected feature set uses only information available before closure.

**Duplicate wf_* columns:** The `make_dataset.py` script built `feature_cols` by listing base features and then extending with all `wf_*` columns via wildcard. Because `wf_total_time` appeared in both lists, pandas selected it as a two-column DataFrame, causing a crash during feature engineering. Fixed by filtering the wildcard to exclude columns already explicitly listed.

**Python version constraint:** The sandbox environment runs Python 3.10, while `pyproject.toml` specifies `>=3.11`. This blocked editable installs during development. Resolved by running with `PYTHONPATH=src` directly. The production environment (where users run `pip install -e .`) correctly requires 3.11+.

### Areas for Improvement

- **Prediction timing:** The current model uses all ticket history, including post-creation events. For a truly operational early-warning system, features should be computed at a fixed time window (e.g., 24 hours after ticket creation) to mirror what is actually available at prediction time.
- **Class weighting:** The 71/29 class split means false negatives (missed violations) may be more costly than false positives. Phase 2 should explore `class_weight="balanced"` and threshold tuning.
- **Hyperparameter search:** The baseline uses default parameters. Phase 2 should add cross-validated grid or random search logged to MLflow.
- **DVC:** Raw data is currently committed to Git, which works for this dataset size but does not scale. Data version control with DVC should be added in Phase 2.
- **Test coverage:** The current tests cover model save/load and fit/predict but do not test the data pipeline (`make_dataset`, `build_features`). Pipeline tests should be added.
