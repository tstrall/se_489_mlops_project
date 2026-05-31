"""Tests for FastAPI request normalization helpers."""

from __future__ import annotations

from api.main import health, normalize_features


def test_health_endpoint_payload() -> None:
    assert health()["status"] == "ok"


def test_normalize_features_one_hot_encodes_known_categories() -> None:
    normalized = normalize_features(
        {
            "issue_type": "Ticket",
            "issue_priority": "High",
            "wf_open": None,
        }
    )

    assert normalized["issue_type_Ticket"] == 1
    assert normalized["issue_type_Bug"] == 0
    assert normalized["issue_priority_High"] == 1
    assert normalized["issue_priority_Medium"] == 0
    assert normalized["wf_open"] == 0
