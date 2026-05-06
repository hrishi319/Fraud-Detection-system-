"""
src/models/predict.py
Inference — used by Flask API and agent layer.
"""
import joblib
import numpy as np
import pandas as pd
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_artifacts():
    model  = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
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
