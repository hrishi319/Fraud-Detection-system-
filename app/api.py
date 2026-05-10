"""
app/api.py
Real-Time Fraud Detection API
"""

import time
import asyncio
import joblib
import pandas as pd

from typing import List
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect
)

from fastapi.responses import Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from app.schemas import (
    TransactionRequest,
    PredictResponse,
    InvestigateResponse,
    HealthResponse
)

from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger


logger = get_logger(__name__)

# =============================================================================
# WEBSOCKET CONNECTIONS
# =============================================================================

active_connections: List[WebSocket] = []


async def broadcast_prediction(data: dict):
    """
    Broadcast prediction to all connected dashboard clients.
    """

    disconnected = []

    for connection in active_connections:

        try:
            await connection.send_json(data)

        except Exception:
            disconnected.append(connection)

    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

PREDICT_COUNTER = Counter(
    'fraud_predictions_total',
    'Total number of predictions made',
    ['endpoint', 'risk_level']
)

FRAUD_COUNTER = Counter(
    'fraud_detected_total',
    'Total number of fraud transactions'
)

PREDICT_LATENCY = Histogram(
    'fraud_prediction_latency_seconds',
    'Prediction latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

INVESTIGATE_LATENCY = Histogram(
    'fraud_investigation_latency_seconds',
    'Investigation latency',
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0]
)

FRAUD_RATE_GAUGE = Gauge(
    'fraud_rate_current',
    'Rolling fraud rate'
)

# =============================================================================
# MODEL ARTIFACTS
# =============================================================================

model = None
scaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, scaler

    logger.info("Loading model artifacts...")

    model = joblib.load(MODELS_DIR / "xgboost_model.pkl")

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    logger.info("Model loaded successfully.")

    yield

    logger.info("API shutdown")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Kafka + FastAPI + Agentic AI Fraud Detection System",
    version="2.0.0",
    lifespan=lifespan
)

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

FEATURE_COLS = [
    'merchant',
    'category',
    'amt',
    'gender',
    'city',
    'state',
    'zip',
    'city_pop',
    'job',
    'hour',
    'day_of_week',
    'month',
    'is_weekend',
    'is_night',
    'age',
    'geo_distance'
]

API_VERSION = "2.0.0"

# =============================================================================
# FRAUD RATE TRACKING
# =============================================================================

_recent_predictions = []

_WINDOW_SIZE = 100


def _update_fraud_rate(is_fraud: int):

    global _recent_predictions

    _recent_predictions.append(is_fraud)

    if len(_recent_predictions) > _WINDOW_SIZE:
        _recent_predictions = _recent_predictions[-_WINDOW_SIZE:]

    rate = sum(_recent_predictions) / len(_recent_predictions)

    FRAUD_RATE_GAUGE.set(rate)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def _build_feature_df(txn: TransactionRequest) -> pd.DataFrame:

    data = {
        col: getattr(txn, col)
        for col in FEATURE_COLS
    }

    return pd.DataFrame([data])[FEATURE_COLS]


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
def root():

    return {
        "message": "Real-Time Fraud Detection API Running",
        "version": API_VERSION,
        "docs": "/docs"
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    active_connections.append(websocket)

    logger.info("Dashboard client connected")

    try:

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:

        if websocket in active_connections:
            active_connections.remove(websocket)

        logger.info("Dashboard client disconnected")


# =============================================================================
# HEALTH ENDPOINT
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():

    model_ok = model is not None

    scaler_ok = scaler is not None

    if not model_ok or not scaler_ok:

        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded"
        )

    return HealthResponse(
        status="healthy",
        model_loaded=model_ok,
        scaler_loaded=scaler_ok,
        version=API_VERSION
    )


# =============================================================================
# PREDICT ENDPOINT
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"]
)
async def predict(txn: TransactionRequest):

    start = time.time()

    try:

        # ---------------------------------------------------------------------
        # FEATURE PREPARATION
        # ---------------------------------------------------------------------

        X = _build_feature_df(txn)

        X_scaled = scaler.transform(X)

        # ---------------------------------------------------------------------
        # MODEL PREDICTION
        # ---------------------------------------------------------------------

        prob = float(
            model.predict_proba(X_scaled)[0, 1]
        )

        pred = int(prob >= 0.5)

        risk = (
            "HIGH" if prob >= 0.8
            else "MEDIUM" if prob >= 0.5
            else "LOW"
        )

        latency = time.time() - start

        # ---------------------------------------------------------------------
        # METRICS
        # ---------------------------------------------------------------------

        PREDICT_COUNTER.labels(
            endpoint="predict",
            risk_level=risk
        ).inc()

        PREDICT_LATENCY.labels(
            endpoint="predict"
        ).observe(latency)

        if pred == 1:
            FRAUD_COUNTER.inc()

        _update_fraud_rate(pred)

        # ---------------------------------------------------------------------
        # RESPONSE
        # ---------------------------------------------------------------------

        response_data = {
            "transaction_id": txn.transaction_id,
            "fraud_probability": round(prob, 4),
            "prediction": pred,
            "risk_level": risk,
            "threshold": 0.5,
            "latency_ms": round(latency * 1000, 2)
        }

        # ---------------------------------------------------------------------
        # WEBSOCKET BROADCAST
        # ---------------------------------------------------------------------

        asyncio.create_task(
            broadcast_prediction(response_data)
        )

        logger.info(
            f"TXN={txn.transaction_id} | "
            f"PROB={prob:.4f} | "
            f"RISK={risk} | "
            f"LATENCY={latency:.3f}s"
        )

        return PredictResponse(**response_data)

    except Exception as e:

        logger.error(f"Prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =============================================================================
# INVESTIGATE ENDPOINT
# =============================================================================

@app.post(
    "/investigate",
    response_model=InvestigateResponse,
    tags=["Agent"]
)
def investigate(txn: TransactionRequest):

    start = time.time()

    try:

        from src.agents.orchestrator import (
            FraudInvestigationOrchestrator
        )

        transaction = {
            col: float(getattr(txn, col))
            for col in FEATURE_COLS
        }

        orchestrator = FraudInvestigationOrchestrator()

        report = orchestrator.investigate(
            txn.transaction_id,
            transaction
        )

        latency = time.time() - start

        risk = report.get(
            "risk_level",
            "UNKNOWN"
        )

        PREDICT_COUNTER.labels(
            endpoint="investigate",
            risk_level=risk
        ).inc()

        INVESTIGATE_LATENCY.observe(latency)

        logger.info(
            f"INVESTIGATION={txn.transaction_id} | "
            f"VERDICT={report.get('verdict')} | "
            f"LATENCY={latency:.2f}s"
        )

        return InvestigateResponse(
            transaction_id=report["transaction_id"],
            verdict=report["verdict"],
            confidence=report["confidence"],
            fraud_score=report["fraud_score"],
            risk_level=report["risk_level"],
            risk_signals=report.get("risk_signals", []),
            top_features=report.get("top_features", []),
            explanation=report.get("explanation", ""),
            timestamp=report["timestamp"]
        )

    except Exception as e:

        logger.error(f"Investigation error: {e}")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

@app.get("/metrics", tags=["System"])
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
