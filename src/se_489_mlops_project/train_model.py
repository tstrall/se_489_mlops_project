"""Model training entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
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

from se_489_mlops_project.config import DEFAULT_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR
from se_489_mlops_project.logging_config import get_logger, setup_logging
from se_489_mlops_project.models.model import Model
from se_489_mlops_project.utils.seed import set_seed

logger = get_logger(__name__)


def train(
    data_path: Path,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int = 42,
) -> None:
    """Train the model and persist the fitted artifact to ``model_dir``.

    Loads processed data, trains a Random Forest classifier, evaluates on test
    set, logs params/metrics/artifact to MLflow, and saves the model artifact.
    """
    logger.info("Training with data=%s", data_path)

    # Load processed data
    data_file = data_path / "processed_data.csv"
    df = pd.read_csv(data_file)
    logger.info("Loaded %d records from %s", len(df), data_file)

    # Separate features and target
    # NOTE: wf_total_time and its derived columns (total_time_days, log_total_time)
    # are excluded from training features to prevent data leakage.  The SLA
    # violation target is defined as wf_total_time > threshold, so including it
    # would give the model direct access to the answer at training time.
    # These columns are retained in processed_data.csv for auditability but
    # must not be used as model inputs.
    leaky_cols = [
        "id",
        "sla_violation",
        "wf_total_time",
        "total_time_days",
        "log_total_time",
    ]
    x = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    y = df["sla_violation"]

    # Handle categorical columns
    categorical_cols = x.select_dtypes(include=["object"]).columns
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

    logger.info("Features: %d, Target: sla_violation (n=%d)", x.shape[1], len(y))
    logger.info("Target distribution: %s", y.value_counts().to_dict())

    # Train-test split (80-20)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed, stratify=y
    )
    logger.info("Train set: %d, Test set: %d", len(x_train), len(x_test))

    mlflow.set_experiment("sla-violation-prediction")

    with mlflow.start_run():
        # Log training parameters
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 5,
                "test_size": 0.2,
                "random_state": seed,
                "n_features": x.shape[1],
                "n_train_samples": len(x_train),
                "n_test_samples": len(x_test),
                "sla_threshold_days": 7,
            }
        )

        # Create and train model
        model = Model()
        logger.info("Training Random Forest classifier...")
        model.fit(x_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)

        # Compute metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Log metrics to MLflow
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

        # Save model artifact and log to MLflow
        model_path = model_dir / "model.joblib"
        model.save(model_path)
        mlflow.sklearn.log_model(model.pipeline, artifact_path="random_forest_pipeline")
        mlflow.log_artifact(str(model_path), artifact_path="model_joblib")
        logger.info("Model saved to %s", model_path)


def main() -> None:
    """CLI entrypoint for model training."""
    cfg = DEFAULT_CONFIG.training
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    train(
        args.data_path,
        args.model_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        seed=args.seed,
    )
    logger.info("Training complete")


if __name__ == "__main__":
    main()
