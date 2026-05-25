import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = Path(os.getenv("HELPEVENTS_MODEL_PATH", "models/model.joblib"))
DATA_PATH = Path(os.getenv("HELPEVENTS_DATA_PATH", "data/processed/processed_data.csv"))

app = FastAPI(title="HelpEvents SLA Prediction API")

_model: Any | None = None


class PredictionRequest(BaseModel):
    features: dict[str, Any]


def normalize_features(features: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(features)

    issue_type = normalized.pop("issue_type", None)
    issue_priority = normalized.pop("issue_priority", None)

    issue_type_cols = [
        "issue_type_Bug",
        "issue_type_Deployment",
        "issue_type_Epic",
        "issue_type_HD Service",
        "issue_type_Project",
        "issue_type_Retrospective",
        "issue_type_Service",
        "issue_type_Sprint Summary",
        "issue_type_Story",
        "issue_type_Sub-task",
        "issue_type_Subtask",
        "issue_type_Task",
        "issue_type_Ticket",
        "issue_type_Vacation",
    ]

    issue_priority_cols = [
        "issue_priority_High",
        "issue_priority_Highest",
        "issue_priority_Low",
        "issue_priority_Lowest",
        "issue_priority_Medium",
        "issue_priority_unknown",
    ]

    for col in issue_type_cols:
        normalized[col] = 1 if col == f"issue_type_{issue_type}" else 0

    for col in issue_priority_cols:
        normalized[col] = 1 if col == f"issue_priority_{issue_priority}" else 0

    for key, value in list(normalized.items()):
        if value is None:
            normalized[key] = 0

    return normalized


def get_model() -> Any:
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        _model = joblib.load(MODEL_PATH)

    return _model


@app.get("/")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "helpevents-sla-api",
    }


@app.get("/sample")
def sample() -> dict[str, Any]:
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Processed data not found: {DATA_PATH}",
        )

    df = pd.read_csv(DATA_PATH)

    row = df.drop(columns=["sla_violation"], errors="ignore").iloc[0].to_dict()

    return {"features": row}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    model = get_model()

    normalized = normalize_features(request.features)

    row = pd.DataFrame([normalized])

    try:
        prediction = int(model.predict(row)[0])

        probability = None

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row)[0][1])

        return {
            "prediction": prediction,
            "sla_violation_probability": probability,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc
