# PHASE 1: Project Design & Model Development

## Overview
Phase 1 establishes the foundation for the MLOps project focused on predicting SLA violations from helpdesk event sequences. This phase includes project planning, code organization, dataset selection, baseline model development, and documentation. The goal is to create a reproducible machine learning pipeline with clear structure and experiment tracking.

---

## 1. Project Proposal

- [x] **Scope & Objectives**: Predict SLA violations using helpdesk event data and build a reproducible ML pipeline
- [x] **Detailed Description**: Project focuses on transforming event sequences into structured features and training classification models to predict SLA violations. The system emphasizes reproducibility through script-based pipelines and experiment tracking using MLflow.
- [x] **Dataset Selection**: Selected real-world helpdesk dataset from Mendeley
- [x] **Dataset Description**: Dataset includes ticket metadata, event interactions, and timestamps from 2016–2023
- [x] **Model Considerations**: Binary classification using Logistic Regression and Random Forest
- [x] **Open-Source Tools**: scikit-learn (modeling), MLflow (experiment tracking), pandas/numpy (data processing)

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
- [ ] **Pull Request Process**: Not yet implemented
- [x] **Team Roles**: Ted Strall acting as project lead and developer
- [ ] **Code Review Guidelines**: Not applicable in Phase 1
- [x] **Commit History**: Maintained for traceability

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: Preprocessing implemented in scripts
- [x] **Normalization**: Planned via feature engineering pipeline
- [ ] **Data Augmentation**: Not applicable for this dataset
- [x] **Data Documentation**: Dataset source and structure documented in README
- [x] **Data Splits**: Train/test split strategy defined
- [ ] **Data Validation**: Basic validation only (to be expanded later)
- [ ] **DVC Setup (Optional)**: Not used (data managed locally)

---

## 5. Model Training

- [x] **Training Environment**: Local Python environment configured
- [x] **Baseline Model**: Logistic Regression / Random Forest planned
- [x] **Hyperparameter Configuration**: Default baseline parameters used initially
- [x] **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score defined
- [x] **Model Persistence**: Models saved to /models directory
- [x] **Training Reproducibility**: Script-based training with fixed pipeline
- [ ] **Performance Baseline**: To be generated after first training run

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
