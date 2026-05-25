"""Tests for deterministic feature engineering."""

from __future__ import annotations

import pandas as pd

from se_489_mlops_project.features.build_features import build_features


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

    assert result["events_per_day"].tolist() == [4.0, 2.0]
    assert result["total_time_days"].tolist() == [2.0, 0.0]
    assert result["comments_per_contributor"].tolist() == [3.0, 3.0]
    assert result["is_high_priority"].tolist() == [1, 0]
    assert "log_total_time" in result.columns
    assert "log_num_events" in result.columns


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

    _ = build_features(df)

    assert "events_per_day" not in df.columns
