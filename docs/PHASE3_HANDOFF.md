# Phase 3 Handoff Runbook

This runbook lists the remaining operational steps for the final Phase 3 submission. The repository now includes the workflow and app scaffolding; the team still needs to run the cloud services, capture screenshots, and paste live URLs into `PHASE3.md` and `README.md`.

## 1. Local Smoke Tests

```bash
make install
make test
make lint
```

Start the API locally:

```bash
make api
```

Try the endpoints:

```bash
curl http://localhost:8000/
curl http://localhost:8000/sample
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @request.json
```

## 2. Docker

The Phase 2 Dockerfile remains the canonical image:

```bash
make docker_build
docker run --rm se_489_mlops_project
```

Run the API with Docker Compose:

```bash
docker compose up api
```

The API service maps `localhost:8000` to the FastAPI app and mounts `models/` and `data/` read-only.

## 3. GitHub Secrets

Configure these repository secrets before running the deployment workflows:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
- `GCP_ARTIFACT_REGISTRY_HOST`
- `GCP_ARTIFACT_REGISTRY_IMAGE`
- `HF_TOKEN`
- `HF_SPACE`

Example Artifact Registry image value:

```text
us-central1-docker.pkg.dev/<project-id>/<repo-name>/helpevents-sla
```

## 4. CML Pull Request

Open a pull request against `main`. The `.github/workflows/cml.yml` workflow will post a Markdown report back to the PR.

If data is managed with DVC, add the DVC config and make sure the workflow can authenticate to the remote. If DVC is not configured, the workflow posts a report explaining that processed data is unavailable in CI.

Screenshot to capture:

```text
docs/screenshots/cml-pr-comment.png
```

## 5. GCP Deployment

### Artifact Registry

1. Create an Artifact Registry Docker repository.
2. Configure the GitHub secrets listed above.
3. Run the Docker Build and Publish workflow.
4. Capture the pushed image screenshot.

Screenshot:

```text
docs/screenshots/artifact-registry-image.png
```

### Training Job

Run the same image as a Vertex AI custom job or Compute Engine job:

```bash
python -m se_489_mlops_project.train_model +experiment=fast
```

Use data from a GCP bucket or a mounted volume, and write the trained `model.joblib` artifact to a bucket or mounted artifact directory.

Screenshot:

```text
docs/screenshots/gcp-training-job.png
```

### Cloud Run

Deploy the image with this container command:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Set environment variables if the model or data paths differ from the container defaults:

```text
HELPEVENTS_MODEL_PATH=/app/models/model.joblib
HELPEVENTS_DATA_PATH=/app/data/processed/processed_data.csv
```

Screenshots:

```text
docs/screenshots/cloud-run-service.png
docs/screenshots/cloud-run-request.png
```

### Cloud Functions

Deploy `api/main.py` as the FastAPI serving module if your team chooses to include Cloud Functions as a second serving target. If Cloud Run is the only live backend, document that choice in `PHASE3.md`.

Screenshot:

```text
docs/screenshots/cloud-functions-endpoint.png
```

## 6. Hugging Face Space

Create a Streamlit Space and set the Space environment variable:

```text
HELPEVENTS_API_URL=<Cloud Run service URL>
```

Configure GitHub secrets:

```text
HF_TOKEN=<token with write access>
HF_SPACE=<username-or-org>/<space-name>
```

Run the Hugging Face Space Sync workflow, then capture:

```text
docs/screenshots/huggingface-space.png
```

## 7. Demo Recording

Record a 2-5 minute narrated or captioned demo:

1. Open the Hugging Face Space.
2. Enter a realistic support ticket.
3. Submit the prediction.
4. Show the SLA risk output.
5. Show evidence of the backend request using curl, browser DevTools, or Cloud Run logs.

Paste the recording link near the top of `README.md`.

## 8. Cleanup

After the evidence is captured, stop or delete billable GCP resources:

- Cloud Run services not needed after grading.
- Cloud Functions not needed after grading.
- Vertex AI or Compute Engine jobs/instances.
- Temporary buckets.
- Artifact Registry images if not needed.

Capture:

```text
docs/screenshots/gcp-cleanup.png
```
