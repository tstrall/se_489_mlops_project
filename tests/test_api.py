"""Tests for FastAPI request normalization helpers and endpoint contracts."""

from __future__ import annotations

from api.main import health, normalize_features

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_endpoint_payload() -> None:
    assert health()["status"] == "ok"


def test_health_returns_service_name() -> None:
    result = health()
    assert "service" in result
    assert result["service"] == "helpevents-sla-api"


# ---------------------------------------------------------------------------
# normalize_features — known categories
# ---------------------------------------------------------------------------


def test_normalize_features_one_hot_encodes_known_issue_type() -> None:
    normalized = normalize_features({"issue_type": "Ticket", "issue_priority": "High"})
    assert normalized["issue_type_Ticket"] == 1
    assert normalized["issue_type_Bug"] == 0
    assert normalized["issue_type_Task"] == 0


def test_normalize_features_one_hot_encodes_known_priority() -> None:
    normalized = normalize_features({"issue_type": "Bug", "issue_priority": "Low"})
    assert normalized["issue_priority_Low"] == 1
    assert normalized["issue_priority_High"] == 0
    assert normalized["issue_priority_Medium"] == 0


def test_normalize_features_none_values_become_zero() -> None:
    normalized = normalize_features(
        {
            "issue_type": "Ticket",
            "issue_priority": "High",
            "wf_open": None,
            "wf_closed": None,
            "duration_seconds": None,
        }
    )
    assert normalized["wf_open"] == 0
    assert normalized["wf_closed"] == 0
    assert normalized["duration_seconds"] == 0


# ---------------------------------------------------------------------------
# normalize_features — unknown / missing categories
# ---------------------------------------------------------------------------


def test_normalize_features_unknown_issue_type_all_zeros() -> None:
    """An unrecognized issue_type should set all issue_type_* columns to 0."""
    normalized = normalize_features(
        {"issue_type": "Unknown_Category_XYZ", "issue_priority": "High"}
    )
    issue_type_cols = [k for k in normalized if k.startswith("issue_type_")]
    assert all(normalized[col] == 0 for col in issue_type_cols)


def test_normalize_features_unknown_priority_all_zeros() -> None:
    """An unrecognized priority should set all issue_priority_* columns to 0."""
    normalized = normalize_features(
        {"issue_type": "Ticket", "issue_priority": "Ultra-Critical"}
    )
    priority_cols = [k for k in normalized if k.startswith("issue_priority_")]
    assert all(normalized[col] == 0 for col in priority_cols)


def test_normalize_features_missing_issue_type_all_zeros() -> None:
    """Omitting issue_type entirely should produce all-zero issue_type columns."""
    normalized = normalize_features({"issue_priority": "Medium"})
    issue_type_cols = [k for k in normalized if k.startswith("issue_type_")]
    assert all(normalized[col] == 0 for col in issue_type_cols)


def test_normalize_features_missing_priority_all_zeros() -> None:
    """Omitting issue_priority entirely should produce all-zero priority columns."""
    normalized = normalize_features({"issue_type": "Bug"})
    priority_cols = [k for k in normalized if k.startswith("issue_priority_")]
    assert all(normalized[col] == 0 for col in priority_cols)


# ---------------------------------------------------------------------------
# normalize_features — output structure
# ---------------------------------------------------------------------------


def test_normalize_features_removes_raw_categorical_keys() -> None:
    """The raw string keys issue_type and issue_priority should not appear in output."""
    normalized = normalize_features({"issue_type": "Ticket", "issue_priority": "High"})
    assert "issue_type" not in normalized
    assert "issue_priority" not in normalized


def test_normalize_features_preserves_numeric_fields() -> None:
    normalized = normalize_features(
        {
            "issue_type": "Bug",
            "issue_priority": "Low",
            "num_events": 10,
            "duration_seconds": 3600,
            "issue_contr_count": 2,
        }
    )
    assert normalized["num_events"] == 10
    assert normalized["duration_seconds"] == 3600
    assert normalized["issue_contr_count"] == 2


def test_normalize_features_exactly_one_issue_type_hot() -> None:
    """Exactly one issue_type column should be 1 for a known type."""
    normalized = normalize_features({"issue_type": "Story", "issue_priority": "High"})
    issue_type_cols = [k for k in normalized if k.startswith("issue_type_")]
    assert sum(normalized[col] for col in issue_type_cols) == 1
    assert normalized["issue_type_Story"] == 1


def test_normalize_features_exactly_one_priority_hot() -> None:
    """Exactly one issue_priority column should be 1 for a known priority."""
    normalized = normalize_features({"issue_type": "Bug", "issue_priority": "Highest"})
    priority_cols = [k for k in normalized if k.startswith("issue_priority_")]
    assert sum(normalized[col] for col in priority_cols) == 1
    assert normalized["issue_priority_Highest"] == 1
