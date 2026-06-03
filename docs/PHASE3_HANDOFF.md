# Phase 3 Handoff Runbook

This runbook lists operational steps for the final Phase 3 submission. The repository now includes the workflow files, Cloud Run deployment evidence, Artifact Registry evidence, and Hugging Face Space evidence; the remaining final-submission work is the demo recording, any additional CI/CML screenshots, and cleanup evidence.

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

The Dockerfile is now configured to serve the FastAPI API by default:

```bash
docker build -f dockerfiles/Dockerfile -t helpevents-api .
docker run --rm \
  -p 8000:8000 \
  -v "$PWD/models:/app/models" \
  -v "$PWD/data/processed:/app/data/processed" \
  helpevents-api
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

Artifact Registry image used for Cloud Run:

```text
us-central1-docker.pkg.dev/infra-inkwell-457919-t8/helpevents/helpevents-api:latest
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

The Docker image was pushed to Artifact Registry in project `infra-inkwell-457919-t8`, region `us-central1`, repository `helpevents`.

Screenshot:

```text
docs/screenshots/gc-artifactrepo.png
```

### Training Job

The same Artifact Registry image was run as a Compute Engine container VM named `helpevents-training-job` in `us-central1-a`:

```bash
python -m se_489_mlops_project.train_model experiment=fast
```

The job completed successfully with ROC-AUC 0.9975, accuracy 0.9804, and F1 0.9860. The VM should be deleted after screenshot capture to avoid compute charges.

Screenshot:

```text
docs/screenshots/gcp-training-job.png
```

### Cloud Run

The deployed Cloud Run service URL is:

```text
https://helpevents-api-263032795187.us-central1.run.app
```

The container serves FastAPI with:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Screenshots:

```text
docs/screenshots/gc-cloudrun.png
docs/screenshots/gc-swagger.png
docs/screenshots/gc-predict.png
```

### Cloud Functions

Cloud Functions was not used as a second serving target. Cloud Run is the primary live backend because the assignment already required container deployment and the API image had been verified locally before cloud deployment.

Screenshot:

```text
Not applicable; Cloud Run is the final live backend.
```

## 6. Hugging Face Space

The Hugging Face Space is live at:

```text
https://huggingface.co/spaces/tstrall/helpevents-sla
```

The Space environment variable is:

```text
HELPEVENTS_API_URL=https://helpevents-api-263032795187.us-central1.run.app
```

Configure GitHub secrets:

```text
HF_TOKEN=<token with write access>
HF_SPACE=<username-or-org>/<space-name>
```

The Hugging Face Space Sync workflow copies the app to the Space. Screenshots:

```text
docs/screenshots/huggingface-space.png
docs/screenshots/huggingface-predict.png
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
