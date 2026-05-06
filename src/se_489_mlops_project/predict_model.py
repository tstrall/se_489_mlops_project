"""Model inference entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from se_489_mlops_project.config import MODELS_DIR, PROCESSED_DATA_DIR
from se_489_mlops_project.logging_config import get_logger, setup_logging
from se_489_mlops_project.models.model import Model

logger = get_logger(__name__)


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    """Load model and write predictions for input_path to output_path."""
    logger.info("Loading model from %s", model_path)
    model = Model.load(model_path)

    logger.info("Scoring %s", input_path)
    df = pd.read_csv(input_path)

    # Handle categorical columns the same way as training
    x = df.drop(
        columns=[col for col in df.columns if col in ["id", "sla_violation"]],
        errors="ignore",
    )
    categorical_cols = x.select_dtypes(include=["object"]).columns
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

    # Generate predictions
    preds = model.predict(x)
    pred_probs = model.predict_proba(x)

    # Create output dataframe
    output_df = pd.DataFrame(
        {
            "prediction": preds,
            "probability_no_violation": pred_probs[:, 0],
            "probability_violation": pred_probs[:, 1],
        }
    )
    output_df.to_csv(output_path, index=False)
    logger.info("Writing predictions to %s", output_path)
    logger.info(
        "Predictions summary: %s violations, %s non-violations",
        (output_df["prediction"] == 1).sum(),
        (output_df["prediction"] == 0).sum(),
    )


def main() -> None:
    """CLI entrypoint for batch prediction."""
    parser = argparse.ArgumentParser(
        description="Generate predictions from a trained model"
    )
    parser.add_argument("--model-path", type=Path, default=MODELS_DIR / "model.joblib")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_DIR / "test.csv")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    setup_logging()
    predict(args.model_path, args.input, args.output)
    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
