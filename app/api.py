"""
app/api.py
FastAPI application — fraud detection REST API.

Endpoints:
  GET  /health      — service health check
  POST /predict     — fast XGBoost prediction (no agent)
  POST /investigate — full agentic investigation pipeline
  GET  /metrics     — Prometheus metrics scrape endpoint

Run locally:
  uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import time
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi.responses import Response

from app.schemas import (
    TransactionRequest, PredictResponse,
    InvestigateResponse, HealthResponse
)
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────
# These are scraped by Prometheus at GET /metrics
# Counter   — monotonically increasing count
# Histogram — tracks distribution of values (e.g. latency buckets)
# Gauge     — value that goes up and down

PREDICT_COUNTER = Counter(
    'fraud_predictions_total',
    'Total number of predictions made',
    ['endpoint', 'risk_level']
)
FRAUD_COUNTER = Counter(
    'fraud_detected_total',
    'Total number of transactions predicted as fraud'
)
PREDICT_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction endpoint latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)
INVESTIGATE_LATENCY = Histogram(
    'fraud_investigation_latency_seconds',
    'Full agent investigation latency',
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)
FRAUD_RATE_GAUGE = Gauge(
    'fraud_rate_current',
    'Rolling fraud rate from recent predictions'
)

# ── Model artifacts ────────────────────────────────────────────────────────────
# Loaded once at startup — not on every request
model  = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup."""
    global model, scaler
    logger.info("Loading model artifacts...")
    model  = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    logger.info("Model and scaler loaded successfully.")
    yield
    logger.info("Shutting down API.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection API",
    description = "XGBoost + Agentic AI fraud detection system",
    version     = "1.0.0",
    lifespan    = lifespan
)

FEATURE_COLS = [
    'merchant', 'category', 'amt', 'gender', 'city', 'state',
    'zip', 'city_pop', 'job', 'hour', 'day_of_week', 'month',
    'is_weekend', 'is_night', 'age', 'geo_distance'
]

API_VERSION = "1.0.0"

# Rolling fraud rate tracking
_recent_predictions = []
_WINDOW_SIZE = 100


def _update_fraud_rate(is_fraud: int):
    """Update rolling fraud rate gauge for Prometheus."""
    global _recent_predictions
    _recent_predictions.append(is_fraud)
    if len(_recent_predictions) > _WINDOW_SIZE:
        _recent_predictions = _recent_predictions[-_WINDOW_SIZE:]
    rate = sum(_recent_predictions) / len(_recent_predictions)
    FRAUD_RATE_GAUGE.set(rate)


def _build_feature_df(txn: TransactionRequest) -> pd.DataFrame:
    """Build feature DataFrame from request — matches preprocessing order."""
    data = {col: getattr(txn, col) for col in FEATURE_COLS}
    return pd.DataFrame([data])[FEATURE_COLS]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """
    Health check endpoint.
    Used by Docker HEALTHCHECK and load balancers.
    Returns 200 if service is ready, 503 if model not loaded.
    """
    model_ok  = model is not None
    scaler_ok = scaler is not None

    if not model_ok or not scaler_ok:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded"
        )

    return HealthResponse(
        status       = "healthy",
        model_loaded = model_ok,
        scaler_loaded= scaler_ok,
        version      = API_VERSION
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(txn: TransactionRequest):
    """
    Fast fraud prediction — XGBoost only, no agent.
    Use this endpoint for high-volume, low-latency scoring.
    Typical latency: <50ms

    Returns fraud probability, binary prediction, and risk level.
    """
    start = time.time()

    try:
        X        = _build_feature_df(txn)
        X_scaled = scaler.transform(X)
        prob     = float(model.predict_proba(X_scaled)[0, 1])
        pred     = int(prob >= 0.5)
        risk     = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"

        # Prometheus metrics
        latency = time.time() - start
        PREDICT_COUNTER.labels(endpoint="predict", risk_level=risk).inc()
        PREDICT_LATENCY.labels(endpoint="predict").observe(latency)
        if pred == 1:
            FRAUD_COUNTER.inc()
        _update_fraud_rate(pred)

        logger.info(f"[/predict] {txn.transaction_id} | "
                    f"prob={prob:.4f} | risk={risk} | latency={latency:.3f}s")

        return PredictResponse(
            transaction_id    = txn.transaction_id,
            fraud_probability = round(prob, 4),
            prediction        = pred,
            risk_level        = risk,
            threshold         = 0.5
        )

    except Exception as e:
        logger.error(f"[/predict] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/investigate", response_model=InvestigateResponse, tags=["Agent"])
def investigate(txn: TransactionRequest):
    """
    Full agentic fraud investigation.
    Runs the LangGraph pipeline: Investigator -> Explainer -> Decision.
    Use this endpoint for high-risk transactions requiring human-readable reports.
    Typical latency: 5-15s (includes Claude API call)

    Returns full case report with verdict, explanation, and risk signals.
    """
    start = time.time()

    try:
        # Lazy import — orchestrator loads heavy dependencies
        from src.agents.orchestrator import FraudInvestigationOrchestrator

        transaction = {col: float(getattr(txn, col)) for col in FEATURE_COLS}
        orchestrator = FraudInvestigationOrchestrator()
        report       = orchestrator.investigate(txn.transaction_id, transaction)

        # Prometheus metrics
        latency = time.time() - start
        risk    = report.get("risk_level", "UNKNOWN")
        PREDICT_COUNTER.labels(endpoint="investigate", risk_level=risk).inc()
        INVESTIGATE_LATENCY.observe(latency)

        logger.info(f"[/investigate] {txn.transaction_id} | "
                    f"verdict={report.get('verdict')} | latency={latency:.1f}s")

        return InvestigateResponse(
            transaction_id = report["transaction_id"],
            verdict        = report["verdict"],
            confidence     = report["confidence"],
            fraud_score    = report["fraud_score"],
            risk_level     = report["risk_level"],
            risk_signals   = report.get("risk_signals", []),
            top_features   = report.get("top_features", []),
            explanation    = report.get("explanation", ""),
            timestamp      = report["timestamp"]
        )

    except Exception as e:
        logger.error(f"[/investigate] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["System"])
def metrics():
    """
    Prometheus metrics scrape endpoint.
    Exposes: prediction counts, latency histograms, fraud rate gauge.
    Prometheus scrapes this endpoint on a configured interval.
    """
    return Response(
        content      = generate_latest(),
        media_type   = CONTENT_TYPE_LATEST
    )
