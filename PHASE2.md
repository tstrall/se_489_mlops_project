# PHASE 2: Enhancing ML Operations with Containerization & Monitoring

## Overview
Phase 2 focuses on operationalizing HelpEvents by adding containerization, configuration management, experiment tracking, logging, profiling, and monitoring. This phase ensures the SLA violation prediction pipeline can run consistently across environments and can be inspected, debugged, and reproduced by other team members.

---

## 1. Containerization

- [x] **Dockerfile Creation**: Dockerfile created for containerized model training
- [x] **Base Image Selection**: Uses required `python:3.11-slim-bookworm` base image
- [x] **Environment Variables**: `PYTHONUNBUFFERED=1` configured for reliable container logging
- [x] **Build Instructions**: Docker build command documented in README
- [x] **Run Instructions**: Docker run command documented with mounted `models/` and `data/processed/` volumes
- [x] **Container Testing**: Container tested locally and successfully runs training end-to-end
- [ ] **Docker Compose (Optional)**: Not required for current single-service training workflow
- [x] **Environment Consistency**: Containerized training produces comparable results to local training

### Verified Docker Commands

```bash
docker build -f dockerfiles/Dockerfile -t helpevents .
```

```bash
docker run --rm \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data/processed:/app/data/processed" \
  helpevents
```

### Verified Container Metrics

- ROC-AUC: 0.9984
- Accuracy: 0.9837
- Precision: 0.9952
- Recall: 0.9818
- F1 Score: 0.9884

---

## 2. Monitoring & Debugging

- [ ] **Debugging Tools**: pdb/ipdb documentation still pending
- [ ] **Debugging Documentation**: Container debugging notes still pending
- [x] **Debug Scenario 1**: Resolved feature mismatch where training expected `events_per_day` but stale processed data did not contain it
- [ ] **Debug Scenario 2**: Pending
- [x] **Logging for Debugging**: Training pipeline logs data path, row count, feature count, target distribution, train/test split, and model metrics
- [x] **Model Assertion Checks**: Training code validates expected target and feature columns
- [x] **Training Validation**: Training logs target distribution and dataset dimensions before fitting model

### Debug Scenario 1: Feature Contract Mismatch

During Docker testing, training failed because the model expected the `events_per_day` feature, but the mounted processed dataset was stale and did not include that column. The fix was to regenerate processed data with the current feature engineering pipeline using:

```bash
make data
```

Then rerun the containerized training command.

This exposed a real MLOps issue: training code and preprocessing output must maintain a consistent feature contract.

---

## 3. Profiling & Optimization

- [ ] **CPU Profiling**: cProfile script/output pending
- [ ] **Memory Profiling**: Optional; pending
- [ ] **GPU Profiling (if applicable)**: Not applicable; project uses scikit-learn on CPU
- [ ] **Profiling Results**: Pending
- [ ] **Optimization 1**: Pending
- [ ] **Optimization 2**: Pending
- [ ] **Performance Benchmarks**: Pending
- [ ] **Optimization Documentation**: Pending

### Planned Profiling Command

```bash
python -m cProfile -o reports/profile_train.out -m se_489_mlops_project.train_model
```

---

## 4. Experiment Management & Tracking

- [x] **MLflow Setup**: MLflow integrated into training pipeline
- [x] **Metric Logging**: Accuracy, precision, recall, F1, and ROC-AUC logged for each run
- [x] **Parameter Logging**: Model parameters and configuration values logged
- [x] **Model Artifact Logging**: Trained scikit-learn model logged to MLflow and saved to `models/model.joblib`
- [x] **Experiment Comparison**: Four Hydra-driven MLflow runs completed and compared
- [x] **Visualization**: MLflow screenshots added under `reports/screenshots/`
- [x] **Best Model Selection**: Accuracy, F1 score, ROC-AUC, and recall reviewed for model selection
- [x] **Experiment Documentation**: Experiment summary table added below

### Current Baseline Result

- Model: Random Forest
- ROC-AUC: 0.9984
- Accuracy: 0.9837
- Precision: 0.9952
- Recall: 0.9818
- F1 Score: 0.9884

### MLflow Experiment Comparison

Four MLflow runs were completed using Hydra configuration overrides. Each run logged model parameters, evaluation metrics, and trained model artifacts.

| Run | Configuration Change | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|---|---|---:|---:|---:|---:|---:|
| Baseline | `n_estimators=100`, `test_size=0.2` | 0.9984 | 0.9837 | 0.9952 | 0.9818 | 0.9884 |
| Smaller Random Forest | `n_estimators=50`, `test_size=0.2` | 0.9983 | 0.9844 | 0.9949 | 0.9831 | 0.9889 |
| Larger Random Forest | `n_estimators=200`, `test_size=0.2` | 0.9984 | 0.9841 | 0.9959 | 0.9816 | 0.9887 |
| Larger Test Split | `n_estimators=100`, `test_size=0.3` | 0.9984 | 0.9849 | 0.9949 | 0.9837 | 0.9893 |

The larger test split run produced the strongest overall accuracy and F1 score while maintaining a very high ROC-AUC. The smaller random forest also performed well and may be preferable when faster training is more important. MLflow made it easier to compare metrics, confirm parameters, and verify that each run saved trained model artifacts.

#### MLflow Evidence Screenshots

![Baseline MLflow Run](reports/screenshots/mlflow-baseline-run.png)

![50 Estimators MLflow Run](reports/screenshots/mlflow-50-estimators-run.png)

![200 Estimators MLflow Run](reports/screenshots/mlflow-200-estimators-run.png)

![Larger Test Split MLflow Run](reports/screenshots/mlflow-test-size-030-run.png)

---

## 5. Application & Experiment Logging

- [x] **Logger Setup**: Python logging configured for training pipeline
- [ ] **Rich Library Setup**: Pending / optional
- [x] **Log Levels**: INFO and WARNING messages used during training and MLflow execution
- [x] **Log Messages**: Informative logs added at key points in the pipeline
- [x] **Training Log Example**: Docker run output demonstrates training logs
- [ ] **Inference Log Example**: Pending
- [x] **Error Logging**: Training failures expose useful stack traces and context
- [ ] **Performance Logging**: Pending
- [ ] **Log Rotation**: Pending / optional

### Example Training Log Output

```text
INFO | Training with data=/app/data/processed
INFO | Loaded 66691 records from /app/data/processed/processed_data.csv
INFO | Features: 50, Target: sla_violation
INFO | Training Random Forest classifier...
INFO | ROC-AUC Score: 0.9984
INFO | Model saved to /app/models/model.joblib
```

---

## 6. Configuration Management

- [x] **Hydra Setup**: Hydra integrated into training pipeline
- [x] **Config Files**: `configs/config.yaml` created for training configuration
- [x] **Config Structure**: Config includes data, model, and training parameters
- [x] **Config Example 1**: Default Random Forest training config
- [x] **Config Example 2**: CLI override example for changing hyperparameters
- [ ] **Config Validation**: Formal schema validation pending
- [x] **Override Documentation**: CLI override command documented
- [x] **Config Version Control**: Config file committed with source code

### Example Hydra Override

```bash
python -m se_489_mlops_project.train_model model.n_estimators=200
```

---

## 7. Documentation & Repository Updates

- [ ] **README Update**:
  - [ ] Containerization section with Docker usage
  - [ ] Debugging and profiling guide
  - [x] Experiment tracking setup instructions
  - [ ] Configuration management guide
  - [ ] Logging usage examples
- [x] **Architecture Documentation**: README includes architecture overview
- [x] **Setup Guide**: README includes setup and execution commands
- [x] **Examples**: Docker and Hydra examples added/planned
- [x] **Tool Integration**: MLflow, Docker, and Hydra integration documented in progress
- [ ] **Troubleshooting**: Pending
- [ ] **Performance Guide**: Pending
- [x] **Version Compatibility**: Docker base image and Python version documented

---

## Remaining Phase 2 Work

Before Phase 2 submission, the main remaining tasks are:

1. Update README with Docker, Hydra, logging, and profiling sections.
2. Complete remaining monitoring, profiling, logging, and debugging documentation.
3. Add any remaining Docker or GitHub Actions screenshots if required.
4. Complete final README and PHASE2.md cleanup before submission.

---

> **Checklist:** Use this as a guide for documenting Phase 2 deliverables.
