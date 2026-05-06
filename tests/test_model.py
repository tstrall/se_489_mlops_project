"""Tests for the Model implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from se_489_mlops_project.models.base import BaseModel
from se_489_mlops_project.models.model import Model


class TestModel:
    def test_is_base_model(self) -> None:
        assert issubclass(Model, BaseModel)

    def test_default_config_is_empty_dict(self) -> None:
        assert Model().config == {}

    def test_custom_config_is_stored(self) -> None:
        cfg = {"lr": 0.01, "epochs": 5}
        assert Model(cfg).config == cfg

    def test_fit_returns_model(self) -> None:
        """Test that fit returns the model instance for chaining."""
        x = pd.DataFrame(
            {
                "issue_contr_count": [1, 2, 3],
                "issue_comments_count": [5, 6, 7],
                "wf_total_time": [1000, 2000, 3000],
                "processing_steps": [2, 3, 4],
                "num_events": [10, 20, 30],
                "duration_seconds": [500, 1500, 2500],
                "issue_type_Ticket": [1, 0, 1],
                "issue_priority_High": [0, 1, 0],
            }
        )
        y = np.array([0, 1, 0])

        model = Model()
        result = model.fit(x, y)
        assert result is model

    def test_predict_works(self) -> None:
        """Test that predict returns predictions."""
        x = pd.DataFrame(
            {
                "issue_contr_count": [1, 2, 3],
                "issue_comments_count": [5, 6, 7],
                "wf_total_time": [1000, 2000, 3000],
                "processing_steps": [2, 3, 4],
                "num_events": [10, 20, 30],
                "duration_seconds": [500, 1500, 2500],
                "issue_type_Ticket": [1, 0, 1],
                "issue_priority_High": [0, 1, 0],
            }
        )
        y = np.array([0, 1, 0])

        model = Model().fit(x, y)
        preds = model.predict(x)
        assert len(preds) == len(x)
        assert set(preds).issubset({0, 1})

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "model.joblib"
        original = Model({"lr": 0.05})
        original.save(path)

        loaded = Model.load(path)
        assert isinstance(loaded, Model)
        assert loaded.config == original.config

    def test_load_rejects_wrong_type(self, tmp_path: Path) -> None:
        import joblib

        path = tmp_path / "not_a_model.joblib"
        joblib.dump({"just": "a dict"}, path)

        with pytest.raises(TypeError):
            Model.load(path)
