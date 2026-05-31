"""Generate a Random Forest feature-importance plot for the trained model.

Usage
-----
From the repo root:

    python scripts/plot_feature_importance.py

Output:

    reports/figures/feature_importance.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from se_489_mlops_project.config import MODELS_DIR, PROCESSED_DATA_DIR  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "reports" / "figures" / "feature_importance.png"


def main() -> None:
    """Load the trained model and save a top-feature importance chart."""
    model = joblib.load(MODELS_DIR / "model.joblib")
    pipeline = model.pipeline

    df = pd.read_csv(PROCESSED_DATA_DIR / "processed_data.csv")
    leaky_cols = [
        "id",
        "sla_violation",
        "wf_total_time",
        "total_time_days",
        "log_total_time",
    ]
    x = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    categorical_cols = x.select_dtypes(include=["object"]).columns
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = [
        name.replace("num__", "").replace("remainder__", "")
        for name in preprocessor.get_feature_names_out(x.columns)
    ]

    importances = pd.Series(classifier.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(12).sort_values()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    top_features.plot(kind="barh", ax=ax, color="#31688e")
    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=160)
    plt.close(fig)
    print(f"Feature importance plot saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
