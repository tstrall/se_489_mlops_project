# PHASE 3: Continuous Machine Learning (CML) & Deployment

> Every item below needs evidence before final submission:
> 1. File or directory reference in this repository.
> 2. Screenshot, live URL, or command output showing the result.
> 3. A short explanation of what was done and why.
>
> Screenshots should be saved under `docs/screenshots/` and linked from this file.

## Phase 3 Status

Phase 3 adds CI/CD automation, CML reporting, cloud deployment evidence, and an interactive Streamlit UI on top of the Phase 1/2 training pipeline. The repository now contains the code, workflow files, deployed Cloud Run API evidence, and Hugging Face Space evidence needed to demonstrate the end-to-end serving path.

## 1. Continuous Integration & Testing

- [x] **1.1 Unit Testing with pytest**
  - File/dir reference: `tests/test_model.py`, `tests/test_features.py`, `tests/test_metrics.py`, `tests/test_api.py`, `tests/test_data_pipeline.py`.
  - Screenshot evidence: `docs/screenshots/ci-green-run.png`, `docs/screenshots/local-pytest-run.png`.
  - Explanation: The test suite covers the model fit/predict/save-load cycle, deterministic feature engineering (including zero-duration and zero-contributor edge cases), evaluation metric invariants (perfect classifier, RMSE = sqrt(MSE)), the data processing pipeline end-to-end, and the FastAPI request normalization layer (unknown categories, missing fields, one-hot encoding correctness). 38 tests run in under 2 seconds and all pass on CI.

- [x] **1.2 GitHub Actions CI Workflow**
  - File/dir reference: `.github/workflows/ci.yml`.
  - Screenshot evidence: `docs/screenshots/ci-green-run.png`.
  - Explanation: The CI workflow installs project and development dependencies, runs Ruff linting and format checks, runs mypy on `src`, and executes pytest with coverage output. This gives reviewers a reproducible signal that every push and pull request still passes core quality checks.
  - DVC note: This repository currently does not include `.dvc/`, `dvc.yaml`, or `.dvc` tracked artifacts. If the team adds DVC before final submission, add `dvc pull` to `.github/workflows/ci.yml` and include the screenshot here.

- [x] **1.3 Pre-commit Hooks**
  - File/dir reference: `.pre-commit-config.yaml`.
  - Screenshot evidence: `docs/screenshots/precommit-run.png`.
  - Explanation: Pre-commit runs Ruff, Ruff format, mypy, trailing whitespace cleanup, EOF checks, and YAML validation before commits. This keeps the final project history cleaner and catches formatting/type issues before CI has to reject a pull request.

## 2. Continuous Docker Building & CML

- [x] **2.1 Automated Docker Builds**
  - File/dir reference: `.github/workflows/docker-build.yml`, `dockerfiles/Dockerfile`, and `docker-compose.yaml`.
  - Screenshot evidence: `docs/screenshots/docker-image.png` for the local image and `docs/screenshots/gc-artifactrepo.png` for the pushed Artifact Registry image.
  - Explanation: The Docker workflow builds the same `dockerfiles/Dockerfile` used in Phase 2, smoke-tests the image, and can push to Docker Hub and GCP Artifact Registry when secrets are configured. The Dockerfile now uses `CMD` instead of a fixed `ENTRYPOINT`, so the same image can train by default or serve FastAPI when Cloud Run/Compose supplies a different command.
  - Required secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `GCP_WORKLOAD_IDENTITY_PROVIDER`, `GCP_SERVICE_ACCOUNT`, `GCP_ARTIFACT_REGISTRY_HOST`, and `GCP_ARTIFACT_REGISTRY_IMAGE`.

- [x] **2.2 Continuous Machine Learning (CML)**
  - File/dir reference: `.github/workflows/cml.yml`, `configs/experiment/fast.yaml`, and `src/se_489_mlops_project/train_model.py`.
  - Screenshot evidence: `docs/screenshots/cml-pr-comment.png`.
  - Explanation: The CML workflow runs on pull requests, installs the project, optionally pulls DVC data if DVC is configured, runs the fast Hydra experiment when processed data is available, and posts a Markdown report back to the PR. This connects model evaluation to code review so reviewers can see model behavior changes alongside code changes.
  - Data note: The CML workflow currently runs in lightweight CI mode because the processed dataset is not committed to Git. Full training in CML requires restoring `data/processed/processed_data.csv` through DVC or an artifact upload step.

## 3. Deployment on Google Cloud Platform (GCP)

- [x] **3.1 GCP Artifact Registry**
  - File/dir reference: `.github/workflows/docker-build.yml`, `dockerfiles/Dockerfile`, and `docs/PHASE3_HANDOFF.md`.
  - Screenshot evidence: `docs/screenshots/gc-artifactrepo.png`.
  - Explanation: The FastAPI Docker image was built and pushed to Google Artifact Registry under the `helpevents` repository. This provides a managed container artifact that Cloud Run can deploy from the same image tested locally.

- [x] **3.2 Custom Training Job on GCP**
  - File/dir reference: `dockerfiles/Dockerfile`, `configs/config.yaml`, `configs/experiment/fast.yaml`, and `src/se_489_mlops_project/train_model.py`.
  - Screenshot evidence: `docs/screenshots/gcp-training-job.png`.
  - Explanation: The same Artifact Registry image used for serving was also launched as a Compute Engine container VM named `helpevents-training-job` in `us-central1-a`. It ran `python -m se_489_mlops_project.train_model experiment=fast` and produced a successful training log with ROC-AUC 0.9975, accuracy 0.9804, and F1 0.9860 before the VM was cleaned up.

- [ ] **3.3 FastAPI + GCP Cloud Functions**
  - File/dir reference: `api/main.py` and `docs/PHASE3_HANDOFF.md`.
  - Live endpoint URL: Not used; Cloud Run is the primary FastAPI serving target for this project.
  - Sample request/response evidence: Not applicable for the final deployed backend; Cloud Run evidence is documented in section 3.4.
  - Explanation: `api/main.py` exposes `/`, `/sample`, and `/predict`, and supports `HELPEVENTS_MODEL_PATH` / `HELPEVENTS_DATA_PATH` environment variables for cloud deployment paths. Cloud Run was selected as the primary serving target because the project already packages the API as a Docker image and Cloud Run directly deploys that verified artifact.

- [x] **3.4 Dockerize & Deploy with GCP Cloud Run**
  - File/dir reference: `dockerfiles/Dockerfile`, `docker-compose.yaml`, `.github/workflows/docker-build.yml`, and `api/main.py`.
  - Live service URL: `https://helpevents-api-263032795187.us-central1.run.app`.
  - Sample request/response evidence: `docs/screenshots/gc-cloudrun.png`, `docs/screenshots/gc-swagger.png`, and `docs/screenshots/gc-predict.png`.
  - Explanation: Cloud Run deploys the Dockerized FastAPI service with `uvicorn api.main:app --host 0.0.0.0 --port 8000`. This makes the trained model reachable through a public HTTPS endpoint while preserving the same reproducible container base used for local Docker testing.

## 4. Interactive UI

- [x] **4.1 Streamlit app on Hugging Face Spaces**
  - File/dir reference: `app/streamlit_app.py` and `.github/workflows/huggingface-space.yml`.
  - Hugging Face Space URL: `https://huggingface.co/spaces/tstrall/helpevents-sla`.
  - Screenshot evidence: `docs/screenshots/huggingface-space.png` and `docs/screenshots/huggingface-predict.png`.
  - Explanation: The Streamlit app lets a non-technical user enter ticket characteristics and calls the deployed Cloud Run FastAPI backend for a prediction. The Hugging Face workflow syncs `app/streamlit_app.py` to the Space using `HF_TOKEN` and `HF_SPACE`, and the Space uses `HELPEVENTS_API_URL` to reach the deployed API.

## 5. End-to-End Demo Recording

- [ ] **5.1 Recording in main README**
  - File/dir reference: `README.md`.
  - Recording link/path for graders: `TODO: paste Loom/YouTube/repo video link`.
  - Explanation: The final README must embed or link a 2-5 minute narrated or captioned walkthrough near the top. The recording should show the Hugging Face UI, a realistic ticket input, the prediction result, and preferably evidence that the request reached the Cloud Run/FastAPI backend.

## 6. Documentation, Repository Updates & Cleanup

- [x] **6.1 Comprehensive README**
  - File/dir reference: `README.md`.
  - Screenshot evidence: Pending final demo screenshot: `docs/screenshots/readme-demo-embed.png`.
  - Explanation: The README now includes a Phase 3 section with CI/CD, Docker, CML, Cloud Run, Hugging Face, and demo-recording guidance. It remains the main front door for reviewers and links to this evidence report.

- [x] **6.2 PHASE3.md**
  - File/dir reference: `PHASE3.md`.
  - Screenshot evidence: this file rendered in GitHub after screenshots and URLs are added.
  - Explanation: This document is structured around the course template and includes repo paths, required screenshots, and concise explanations. Items that require live cloud execution remain unchecked until the team captures real evidence.

- [ ] **6.3 GCP Resource Cleanup**
  - File/dir reference: `docs/PHASE3_HANDOFF.md`.
  - Screenshot evidence: `docs/screenshots/gcp-cleanup.png`.
  - Explanation: After deployment screenshots and the demo recording are captured, delete or stop billable GCP resources such as Cloud Run services, Cloud Functions, Vertex/Compute jobs, buckets, and registry images that are no longer needed. Add a dated cleanup screenshot here so graders can see the team avoided lingering cloud costs.

## Final Submission Checklist

- [x] Add current CI, CML, GCP, Docker, and Hugging Face screenshots under `docs/screenshots/`.
- [x] Paste live Cloud Run and Hugging Face URLs above.
- [ ] Embed or link the 2-5 minute demo recording near the top of `README.md`.
- [x] Run GitHub Actions after secrets are configured and capture green workflow evidence.
- [ ] Capture GCP cleanup evidence after the demo is recorded.
