# Deployment Guide

This guide summarizes the Phase 3 deployment path for HelpEvents.

## Local API

```bash
make install
make train
make api
```

Open:

```text
http://localhost:8000/
http://localhost:8000/docs
```

## Docker Compose API

```bash
make docker_build
docker compose up api
```

The API is available at:

```text
http://localhost:8000
```

## Cloud Run Command

The FastAPI service is deployed on Cloud Run at:

```text
https://helpevents-api-263032795187.us-central1.run.app
```

Deploy the Docker image with this service command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Use these environment variables when the artifact paths differ:

```text
HELPEVENTS_MODEL_PATH=/app/models/model.joblib
HELPEVENTS_DATA_PATH=/app/data/processed/processed_data.csv
```

## API Request

```bash
export HELPEVENTS_API_URL="https://helpevents-api-263032795187.us-central1.run.app"

curl -X POST "$HELPEVENTS_API_URL/predict" \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Hugging Face Streamlit UI

The app lives at `app/streamlit_app.py` and is deployed at:

```text
https://huggingface.co/spaces/tstrall/helpevents-sla
```

In Hugging Face Spaces, set:

```text
HELPEVENTS_API_URL=https://helpevents-api-263032795187.us-central1.run.app
```

The GitHub workflow `.github/workflows/huggingface-space.yml` syncs the app to the Space when `HF_TOKEN` and `HF_SPACE` secrets are configured.
