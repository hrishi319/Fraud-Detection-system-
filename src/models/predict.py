"""
src/models/predict.py
Inference — used by FastAPI and agent layer.

Model loading strategy:
  Primary   → MLflow Model Registry (alias: production)
  Fallback  → local pkl file (models/xgboost_model.pkl)

This means in production the registry is the source of truth.
Locally during development the pkl file is used as fallback
if the registry is not available.
"""
import os
import joblib
import numpy as np
import pandas as pd
import mlflow.xgboost
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

REGISTRY_MODEL_NAME  = "fraud-detection-xgboost"
REGISTRY_MODEL_ALIAS = "production"


def load_artifacts():
    """
    Load model and scaler.
    Tries MLflow registry first, falls back to local pkl.
    """
    # Try loading from MLflow registry
    try:
        mlflow.set_tracking_uri("mlruns")
        model_uri = f"models:/{REGISTRY_MODEL_NAME}@{REGISTRY_MODEL_ALIAS}"
        model     = mlflow.xgboost.load_model(model_uri)
        scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
        logger.info(f"Model loaded from registry: {model_uri}")
        return model, scaler
    except Exception as e:
        logger.warning(f"Registry load failed ({e}) — falling back to local pkl")
        model  = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        logger.info("Model loaded from local pkl file.")
        return model, scaler
    

def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Return fraud probability scores for preprocessed feature DataFrame."""
    model, scaler = load_artifacts()
    X = scaler.transform(df)
    return model.predict_proba(X)[:, 1]

def predict_single(transaction_features: pd.DataFrame,
                   threshold: float = 0.5) -> dict:
    """
    Predict a single transaction.
    Returns probability, binary prediction, and risk label.
    Used by agent layer in Phase 5.
    """
    model, scaler = load_artifacts()
    X     = scaler.transform(transaction_features)
    prob  = float(model.predict_proba(X)[0, 1])
    pred  = int(prob >= threshold)
    risk  = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "prediction"       : pred,
        "risk_level"       : risk,
        "threshold"        : threshold
    }
