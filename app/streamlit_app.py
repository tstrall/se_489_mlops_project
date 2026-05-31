"""Streamlit UI for the HelpEvents SLA prediction API."""

from __future__ import annotations

import os
from math import log1p
from typing import Any

import requests  # type: ignore[import-untyped]
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
API_URL = os.getenv("HELPEVENTS_API_URL", DEFAULT_API_URL).rstrip("/")


def build_payload() -> dict[str, Any]:
    """Build a realistic single-ticket payload for the FastAPI service."""
    issue_contr_count = st.number_input("Contributors", min_value=0, value=1)
    issue_comments_count = st.number_input("Comments", min_value=0, value=3)
    processing_steps = st.number_input("Workflow steps", min_value=0, value=4)
    num_events = st.number_input("Event count", min_value=0, value=8)
    duration_seconds = st.number_input(
        "Observed duration seconds", min_value=0, value=3600
    )
    issue_priority = st.selectbox(
        "Priority",
        ["Low", "Medium", "High", "Highest", "Lowest", "unknown"],
        index=1,
    )
    issue_type = st.selectbox(
        "Issue type",
        [
            "Ticket",
            "Bug",
            "Task",
            "Story",
            "Service",
            "Deployment",
            "Epic",
            "Sub-task",
            "Subtask",
            "Project",
            "Retrospective",
            "Sprint Summary",
            "Vacation",
            "HD Service",
        ],
    )

    events_per_day = float(num_events) / max(float(duration_seconds) / 86_400, 1.0)
    comments_per_contributor = float(issue_comments_count) / max(
        float(issue_contr_count), 1.0
    )

    return {
        "issue_num": 0,
        "issue_contr_count": issue_contr_count,
        "issue_priority": issue_priority,
        "issue_type": issue_type,
        "issue_comments_count": issue_comments_count,
        "processing_steps": processing_steps,
        "num_events": num_events,
        "duration_seconds": duration_seconds,
        "wf_in_review": 0,
        "wf_deployment": 0,
        "wf_resolved": 0,
        "wf_open": 0,
        "wf_monitoring": 0,
        "wf_done": 0,
        "wf_pending_customer_approval": 0,
        "wf_rejected": 0,
        "wf_testing_monitoring": 0,
        "wf_in_progress": duration_seconds,
        "wf_reopened": 0,
        "wf_to_do": 0,
        "wf_validation": 0,
        "wf_resolved_under_monitoring": 0,
        "wf_closed": 0,
        "wf_waiting": 0,
        "wf_cancelled": 0,
        "wf_under_review": 0,
        "wf_approved": 0,
        "wf_pending_deployment": 0,
        "events_per_day": events_per_day,
        "comments_per_contributor": comments_per_contributor,
        "is_high_priority": int(issue_priority in {"High", "Highest"}),
        "log_num_events": 0 if num_events == 0 else log1p(num_events),
    }


st.set_page_config(page_title="HelpEvents SLA Predictor", page_icon=":bar_chart:")
st.title("HelpEvents SLA Predictor")
st.caption(
    "Interactive Phase 3 demo backed by the deployed FastAPI prediction service."
)

with st.sidebar:
    st.header("Backend")
    api_url = st.text_input("FastAPI URL", value=API_URL)
    st.caption(
        "Set HELPEVENTS_API_URL in Hugging Face Spaces for the deployed backend."
    )

with st.form("prediction-form"):
    features = build_payload()
    submitted = st.form_submit_button("Predict SLA risk")

if submitted:
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json={"features": features},
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
    else:
        result = response.json()
        probability = result.get("sla_violation_probability")
        prediction = result.get("prediction")

        if probability is not None:
            st.metric("SLA violation probability", f"{probability:.1%}")
            st.progress(min(max(float(probability), 0.0), 1.0))

        if prediction == 1:
            st.warning("Prediction: likely SLA violation. Escalation is recommended.")
        else:
            st.success("Prediction: SLA violation is not expected for this ticket.")

        st.json(result)
