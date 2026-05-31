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

Deploy the Phase 2 Docker image and configure the service command:

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
curl -X POST "$HELPEVENTS_API_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "issue_contr_count": 1,
      "issue_comments_count": 3,
      "processing_steps": 4,
      "num_events": 8,
      "duration_seconds": 3600,
      "issue_priority": "Medium",
      "issue_type": "Ticket",
      "events_per_day": 8,
      "comments_per_contributor": 3,
      "is_high_priority": 0,
      "log_num_events": 2.197224577
    }
  }'
```

## Hugging Face Streamlit UI

The app lives at `app/streamlit_app.py`. In Hugging Face Spaces, set:

```text
HELPEVENTS_API_URL=<Cloud Run URL>
```

The GitHub workflow `.github/workflows/huggingface-space.yml` syncs the app to the Space when `HF_TOKEN` and `HF_SPACE` secrets are configured.
