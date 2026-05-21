"""Model training entrypoint."""

from __future__ import annotations

from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from se_489_mlops_project.config import MODELS_DIR
from se_489_mlops_project.config_validation import validate_config
from se_489_mlops_project.logging_config import get_logger, setup_logging
from se_489_mlops_project.models.model import Model
from se_489_mlops_project.utils.seed import set_seed

logger = get_logger(__name__)


def train(
    data_path: Path,
    model_dir: Path,
    processed_file: str,
    target_col: str,
    experiment_name: str,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> None:
    """Train the model and persist the fitted artifact to ``model_dir``."""

    logger.info("Training with data=%s", data_path)

    data_file = data_path / processed_file
    df = pd.read_csv(data_file)
    logger.info("Loaded %d records from %s", len(df), data_file)

    leaky_cols = [
        "id",
        target_col,
        "wf_total_time",
        "total_time_days",
        "log_total_time",
    ]
    x = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    y = df[target_col]

    categorical_cols = x.select_dtypes(include=["object"]).columns
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

    logger.info("Features: %d, Target: %s (n=%d)", x.shape[1], target_col, len(y))
    logger.info("Target distribution: %s", y.value_counts().to_dict())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Train set: %d, Test set: %d", len(x_train), len(x_test))

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "test_size": test_size,
                "random_state": random_state,
                "n_features": x.shape[1],
                "n_train_samples": len(x_train),
                "n_test_samples": len(x_test),
                "sla_threshold_days": 7,
            }
        )

        model = Model(
            config={
                "n_estimators": n_estimators,
                "random_state": random_state,
            }
        )

        logger.info("Training Random Forest classifier...")
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)

        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metrics(
            {
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        logger.info("ROC-AUC Score: %.4f", roc_auc)
        logger.info("Accuracy:      %.4f", accuracy)
        logger.info("Precision:     %.4f", precision)
        logger.info("Recall:        %.4f", recall)
        logger.info("F1 Score:      %.4f", f1)
        logger.info("Confusion Matrix:\n%s", cm)
        logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

        model_path = model_dir / "model.joblib"
        model.save(model_path)

        mlflow.sklearn.log_model(model.pipeline, artifact_path="random_forest_pipeline")
        mlflow.log_artifact(str(model_path), artifact_path="model_joblib")

        logger.info("Model saved to %s", model_path)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """CLI entrypoint for model training."""
    setup_logging(log_file = "logs/train.log")
    validate_config(cfg)
    set_seed(cfg.training.random_state)

    train(
        data_path=Path(cfg.data.processed_dir),
        model_dir=MODELS_DIR,
        processed_file=cfg.data.processed_file,
        target_col=cfg.data.target,
        experiment_name=cfg.experiment.name,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        n_estimators=cfg.model.n_estimators,
    )

    logger.info("Training complete")


if __name__ == "__main__":
    main()

