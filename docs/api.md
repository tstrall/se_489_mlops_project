# API Reference

The package is importable as `se_489_mlops_project` after running `pip install -e .`.

## `se_489_mlops_project.config`

Project-wide path constants and typed config dataclasses.

```python
from se_489_mlops_project.config import (
    PROJECT_ROOT,
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
    Config, TrainingConfig, DataConfig, DEFAULT_CONFIG,
)
```

Use these constants instead of hard-coded relative paths — they resolve against the repo root regardless of the current working directory.

## `se_489_mlops_project.logging_config`

```python
from se_489_mlops_project.logging_config import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)
```

## `se_489_mlops_project.data`

| Function | Purpose |
|---|---|
| `load_raw(filename)` | Read CSV from `data/raw/` |
| `load_processed(filename)` | Read CSV from `data/processed/` |
| `save_processed(df, filename)` | Write CSV to `data/processed/` |
| `process_data(input_dir, output_dir)` | Raw → processed pipeline |

CLI: `python -m se_489_mlops_project.data.make_dataset [--input PATH] [--output PATH]`

## `se_489_mlops_project.features`

```python
from se_489_mlops_project.features import build_features

df_features = build_features(df_processed)
```

## `se_489_mlops_project.models`

### `BaseModel` (abstract)

Abstract interface with `fit`, `predict`, `save`, `load`. Extend this for any new estimator.

### `Model`

Reference implementation scaffold. Serializes via `joblib`.

```python
from pathlib import Path
from se_489_mlops_project.models import Model

model = Model(config={"lr": 0.01})
# model.fit(X_train, y_train)
model.save(Path("models/model.joblib"))
reloaded = Model.load(Path("models/model.joblib"))
```

## `se_489_mlops_project.evaluation`

```python
from se_489_mlops_project.evaluation import classification_report, regression_report

metrics = classification_report(y_true, y_pred)
# -> {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
```

## `se_489_mlops_project.visualization`

```python
from se_489_mlops_project.visualization import plot_training_history, plot_confusion_matrix
```

## `se_489_mlops_project.utils`

```python
from se_489_mlops_project.utils import set_seed, save_json, load_json

set_seed(42)
```

## Training / Prediction CLIs

```bash
python -m se_489_mlops_project.train_model --epochs 100 --batch-size 64
python -m se_489_mlops_project.predict_model --model-path models/model.joblib --input data/processed/test.csv
```

## Configuration

Defaults live in `se_489_mlops_project.config.DEFAULT_CONFIG`. Override via CLI flags on the training/prediction entrypoints.

---

**SE 489 MLOps Project** · Version see `se_489_mlops_project.__version__`
