"""Tests for deterministic feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from se_489_mlops_project.features.build_features import build_features


def _base_row(**overrides: object) -> pd.DataFrame:
    """Return a single-row DataFrame with sensible defaults for all required columns."""
    defaults: dict[str, object] = {
        "duration_seconds": 86_400,
        "num_events": 4,
        "wf_total_time": 172_800,
        "issue_contr_count": 2,
        "issue_comments_count": 6,
        "issue_priority": "High",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_build_features_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "duration_seconds": [86_400, 0],
            "num_events": [4, 2],
            "wf_total_time": [172_800, 0],
            "issue_contr_count": [2, 0],
            "issue_comments_count": [6, 3],
            "issue_priority": ["High", "Low"],
        }
    )
    result = build_features(df)

    for col in [
        "events_per_day",
        "total_time_days",
        "comments_per_contributor",
        "is_high_priority",
        "log_total_time",
        "log_num_events",
    ]:
        assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Correctness — normal inputs
# ---------------------------------------------------------------------------

def test_events_per_day_normal() -> None:
    result = build_features(_base_row(duration_seconds=86_400, num_events=4))
    assert result["events_per_day"].iloc[0] == pytest.approx(4.0)


def test_total_time_days_conversion() -> None:
    result = build_features(_base_row(wf_total_time=172_800))
    assert result["total_time_days"].iloc[0] == pytest.approx(2.0)


def test_comments_per_contributor_normal() -> None:
    result = build_features(_base_row(issue_comments_count=6, issue_contr_count=2))
    assert result["comments_per_contributor"].iloc[0] == pytest.approx(3.0)


def test_is_high_priority_high() -> None:
    result = build_features(_base_row(issue_priority="High"))
    assert result["is_high_priority"].iloc[0] == 1


def test_is_high_priority_low() -> None:
    result = build_features(_base_row(issue_priority="Low"))
    assert result["is_high_priority"].iloc[0] == 0


def test_is_high_priority_medium() -> None:
    result = build_features(_base_row(issue_priority="Medium"))
    assert result["is_high_priority"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Edge cases — zero durations and contributors
# ---------------------------------------------------------------------------

def test_events_per_day_zero_duration_falls_back_to_num_events() -> None:
    """When duration is 0, events_per_day should equal num_events (not divide by zero)."""  # noqa: E501
    result = build_features(_base_row(duration_seconds=0, num_events=5))
    assert result["events_per_day"].iloc[0] == pytest.approx(5.0)


def test_comments_per_contributor_zero_contributors() -> None:
    """When contributor count is 0, use raw comment count instead of dividing by zero."""  # noqa: E501
    result = build_features(_base_row(issue_contr_count=0, issue_comments_count=3))
    assert result["comments_per_contributor"].iloc[0] == pytest.approx(3.0)


def test_events_per_day_zero_events_zero_duration() -> None:
    result = build_features(_base_row(duration_seconds=0, num_events=0))
    assert result["events_per_day"].iloc[0] == pytest.approx(0.0)


def test_total_time_days_zero() -> None:
    result = build_features(_base_row(wf_total_time=0))
    assert result["total_time_days"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Log-scaled columns
# ---------------------------------------------------------------------------

def test_log_columns_are_nonnegative() -> None:
    df = pd.DataFrame(
        {
            "duration_seconds": [0, 3600, 86400],
            "num_events": [0, 5, 100],
            "wf_total_time": [0, 7200, 604800],
            "issue_contr_count": [1, 2, 10],
            "issue_comments_count": [0, 3, 50],
            "issue_priority": ["High", "Low", "Medium"],
        }
    )
    result = build_features(df)
    assert (result["log_total_time"] >= 0).all()
    assert (result["log_num_events"] >= 0).all()


def test_log_total_time_zero_input_is_zero() -> None:
    """log1p(0) == 0, so a ticket with zero wf_total_time should give log_total_time=0."""  # noqa: E501
    result = build_features(_base_row(wf_total_time=0))
    assert result["log_total_time"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_build_features_does_not_mutate_input() -> None:
    df = pd.DataFrame(
        {
            "duration_seconds": [1],
            "num_events": [1],
            "wf_total_time": [1],
            "issue_contr_count": [1],
            "issue_comments_count": [1],
            "issue_priority": ["Medium"],
        }
    )
    original_cols = set(df.columns)
    _ = build_features(df)
    assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Multi-row consistency
# ---------------------------------------------------------------------------

def test_build_features_row_count_preserved() -> None:
    df = pd.DataFrame(
        {
            "duration_seconds": [100, 200, 300],
            "num_events": [1, 2, 3],
            "wf_total_time": [1000, 2000, 3000],
            "issue_contr_count": [1, 2, 3],
            "issue_comments_count": [2, 4, 6],
            "issue_priority": ["High", "Medium", "Low"],
        }
    )
    result = build_features(df)
    assert len(result) == len(df)
