"""Tests for evaluation metric helpers."""

from __future__ import annotations

import math

import pytest

from se_489_mlops_project.evaluation.metrics import (
    classification_report,
    regression_report,
)

# ---------------------------------------------------------------------------
# classification_report
# ---------------------------------------------------------------------------


def test_classification_report_returns_expected_keys() -> None:
    result = classification_report([0, 1, 1], [0, 1, 0])
    assert set(result) == {"accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= value <= 1.0 for value in result.values())


def test_classification_report_perfect_classifier() -> None:
    result = classification_report([0, 1, 0, 1], [0, 1, 0, 1])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(1.0)


def test_classification_report_all_wrong() -> None:
    """Flipping all labels should give 0 accuracy."""
    result = classification_report([0, 0, 1, 1], [1, 1, 0, 0])
    assert result["accuracy"] == pytest.approx(0.0)


def test_classification_report_all_same_prediction() -> None:
    """Predicting all 1s when ground truth is mixed — precision/recall/f1 still valid."""  # noqa: E501
    result = classification_report([0, 1, 1], [1, 1, 1])
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0


def test_classification_report_values_in_range() -> None:
    result = classification_report([0, 1, 1, 0, 1], [0, 0, 1, 0, 1])
    for key, value in result.items():
        assert 0.0 <= value <= 1.0, f"{key}={value} is out of [0, 1]"


# ---------------------------------------------------------------------------
# regression_report
# ---------------------------------------------------------------------------


def test_regression_report_returns_expected_keys() -> None:
    result = regression_report([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])
    assert set(result) == {"mae", "mse", "rmse", "r2"}


def test_regression_report_perfect_prediction() -> None:
    result = regression_report([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["mae"] == pytest.approx(0.0)
    assert result["mse"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_regression_report_rmse_equals_sqrt_mse() -> None:
    result = regression_report([1.0, 2.0, 3.0], [1.5, 2.5, 2.0])
    assert result["rmse"] == pytest.approx(math.sqrt(result["mse"]))


def test_regression_report_mse_nonnegative() -> None:
    result = regression_report([0.0, 1.0, 2.0], [3.0, 1.0, 0.0])
    assert result["mse"] >= 0.0
    assert result["rmse"] >= 0.0
    assert result["mae"] >= 0.0


def test_regression_report_mae_less_than_or_equal_rmse() -> None:
    """MAE <= RMSE always holds (by Cauchy-Schwarz)."""
    result = regression_report([1.0, 2.0, 3.0, 4.0], [1.5, 3.0, 2.0, 5.0])
    assert result["mae"] <= result["rmse"] + 1e-9
