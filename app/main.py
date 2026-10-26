"""
FastAPI service for the breast cancer triage classifier.

Endpoints:
  GET  /health      – liveness check (no Redis dependency)
  POST /predict     – classify one sample
  GET  /model/info  – model metadata and test-set metrics
  GET  /stats       – prediction counter (backed by Redis when available)
"""

import json
import logging
import os
import sys

import joblib
import numpy as np
from fastapi import FastAPI

from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    StatsResponse,
)

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
REDIS_HOST = os.getenv("REDIS_HOST", "")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "0.0.0")
GIT_SHA = os.getenv("GIT_SHA", "unknown")

# ---------------------------------------------------------------------------
# Logging – stdout only
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("breast_cancer_triage")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Breast Cancer Triage API",
    version=MODEL_VERSION,
    description="Screening support tool that classifies cell-nucleus measurements.",
)

# ---------------------------------------------------------------------------
# Startup: load model & model card, optionally connect to Redis
# ---------------------------------------------------------------------------
model = None
model_card = None
redis_client = None
local_counter = 0  # fallback counter when Redis is unavailable


@app.on_event("startup")
async def startup():
    global model, model_card, redis_client

    # Load model artifact
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model artifact not found at {MODEL_PATH}. Exiting.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH}")

    # Load model card
    card_path = os.path.join(os.path.dirname(MODEL_PATH), "model_card.json")
    if os.path.exists(card_path):
        with open(card_path, "r") as f:
            model_card = json.load(f)
        logger.info("Model card loaded.")
    else:
        logger.warning(f"Model card not found at {card_path}. /model/info will have limited data.")
        model_card = {}

    # Connect to Redis if configured
    if REDIS_HOST:
        try:
            import redis
            redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
            )
            redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Stats will use local counter.")
            redis_client = None
    else:
        logger.info("REDIS_HOST not set. Running without Redis.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check. Does not depend on Redis."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify one sample as malignant or benign."""
    global local_counter

    # Build feature array in the order the model expects
    features = np.array([[
        request.mean_radius,
        request.mean_texture,
        request.mean_perimeter,
        request.mean_area,
        request.mean_smoothness,
        request.mean_compactness,
        request.mean_concavity,
        request.mean_concave_points,
        request.mean_symmetry,
        request.mean_fractal_dimension,
        request.radius_error,
        request.texture_error,
        request.perimeter_error,
        request.area_error,
        request.smoothness_error,
        request.compactness_error,
        request.concavity_error,
        request.concave_points_error,
        request.symmetry_error,
        request.fractal_dimension_error,
        request.worst_radius,
        request.worst_texture,
        request.worst_perimeter,
        request.worst_area,
        request.worst_smoothness,
        request.worst_compactness,
        request.worst_concavity,
        request.worst_concave_points,
        request.worst_symmetry,
        request.worst_fractal_dimension,
    ]])

    # Get probabilities — class 0 is malignant
    probabilities = model.predict_proba(features)[0]
    malignant_idx = list(model.classes_).index(0)
    malignant_prob = float(probabilities[malignant_idx])

    label = "malignant" if malignant_prob >= DECISION_THRESHOLD else "benign"

    # Increment prediction counter
    if redis_client:
        try:
            redis_client.incr("prediction_count")
        except Exception as e:
            logger.warning(f"Redis increment failed: {e}")
            local_counter += 1
    else:
        local_counter += 1

    logger.info(f"Prediction: {label} (prob={malignant_prob:.4f})")

    return PredictionResponse(
        label=label,
        malignant_probability=round(malignant_prob, 4),
        threshold=DECISION_THRESHOLD,
        model_version=MODEL_VERSION,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata and test-set metrics."""
    import sklearn

    return ModelInfoResponse(
        model_version=MODEL_VERSION,
        model_family=model_card.get("model_family", "unknown"),
        sklearn_version=sklearn.__version__,
        git_commit=GIT_SHA,
        decision_threshold=DECISION_THRESHOLD,
        feature_count=model_card.get("feature_count", 30),
        test_metrics=model_card.get("test_metrics", {}),
    )


@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Return prediction count from Redis (or local counter)."""
    if redis_client:
        try:
            count = int(redis_client.get("prediction_count") or 0)
        except Exception:
            count = local_counter
    else:
        count = local_counter

    return StatsResponse(predictions_served=count)
