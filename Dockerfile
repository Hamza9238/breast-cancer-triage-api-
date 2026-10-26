# ─── Production Dockerfile ───
# Multi-stage build: Stage 1 trains the model, Stage 2 serves it.
# Only serving deps and model artifacts end up in the final image.

# ===========================================================================
# Stage 1 — Train
# ===========================================================================
FROM python:3.11.12-slim AS trainer

WORKDIR /build

# Install training dependencies first (layer caching)
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy training code and data
COPY params.yaml .
COPY training/ training/
COPY data/ data/

# Build args for versioning
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown

# Train the model
RUN python training/train.py \
    --data-path data/raw/breast_cancer.csv \
    --output-dir artifacts \
    --model-version ${MODEL_VERSION}

# ===========================================================================
# Stage 2 — Serve
# ===========================================================================
FROM python:3.11.12-slim AS server

# Labels for traceability
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.title="breast-cancer-triage-api" \
      org.opencontainers.image.description="Breast cancer triage screening API"

WORKDIR /app

# Install serving dependencies first (layer caching)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application code
COPY app/ app/

# Copy model artifacts from the training stage
COPY --from=trainer /build/artifacts/ artifacts/

# Create a non-root user and switch to it
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Environment variables with sensible defaults
ENV MODEL_PATH=artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    REDIS_HOST="" \
    REDIS_PORT=6379 \
    MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA}

# Expose the API port
EXPOSE 8000

# Health check using curl-free approach (Python is available)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Exec-form ENTRYPOINT + CMD so the server binary is fixed but args can be overridden
ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
