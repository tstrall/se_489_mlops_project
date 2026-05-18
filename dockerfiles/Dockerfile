
# Multi-stage build for SE 489 MLOps Project

# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy project code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "se_489_mlops_project.train_model"]
