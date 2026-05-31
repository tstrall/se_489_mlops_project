"""Tests for evaluation metric helpers."""

from __future__ import annotations

from se_489_mlops_project.evaluation.metrics import (
    classification_report,
    regression_report,
)


def test_classification_report_returns_expected_keys() -> None:
    result = classification_report([0, 1, 1], [0, 1, 0])

    assert set(result) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_regression_report_returns_expected_keys() -> None:
    result = regression_report([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])

    assert set(result) == {"mae", "mse", "rmse", "r2"}
    assert result["mse"] >= 0.0
    assert result["rmse"] >= 0.0
