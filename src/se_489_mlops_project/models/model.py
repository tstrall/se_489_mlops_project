"""Model implementation for SLA violation prediction.

Uses scikit-learn RandomForestClassifier with preprocessing pipeline.
Model persistence uses ``joblib`` (the sklearn convention). Only load
artifacts from trusted sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from se_489_mlops_project.models.base import BaseModel


class Model(BaseModel):
    """SLA violation prediction model using Random Forest."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the model with preprocessing and classifier.

        Args:
            config: Optional configuration dictionary for model parameters.
        """
        super().__init__(config)

        # Numerical features to scale
        numeric_features = [
            "issue_contr_count",
            "issue_comments_count",
            "wf_total_time",
            "processing_steps",
            "num_events",
            "duration_seconds",
        ]

        # Preprocessing: scale numeric features
        numeric_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features)],
            remainder="passthrough",
        )

        # Pipeline: preprocess -> classify
        self.pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def fit(self, x: Any, y: Any) -> Model:
        """Fit the model to training data."""
        self.pipeline.fit(x, y)
        return self

    def predict(self, x: Any) -> Any:
        """Generate predictions."""
        return self.pipeline.predict(x)

    def predict_proba(self, x: Any) -> Any:
        """Generate prediction probabilities."""
        return self.pipeline.named_steps["classifier"].predict_proba(x)

    def save(self, path: Path) -> None:
        """Save the model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Model:
        """Load a model from disk."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj
