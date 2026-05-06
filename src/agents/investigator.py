"""
src/agents/investigator.py

Investigator Agent — Node 1 in the LangGraph pipeline.

Responsibilities:
  1. Run the transaction through XGBoost to get fraud probability
  2. Extract contextual signals from transaction fields
  3. Identify and flag specific risk anomalies
  4. Populate the shared state with findings

This agent does NOT call the LLM — it is pure logic.
The LLM is only used in the ExplainerAgent to generate narrative.

MLflow Traces: wraps execution with mlflow.trace() for full auditability.
"""

import mlflow
import numpy as np
import pandas as pd
import joblib
from src.agents.state import FraudInvestigationState
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_artifacts():
    model  = joblib.load(MODELS_DIR / "xgboost_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


def _extract_risk_signals(transaction: dict, fraud_score: float) -> list:
    """
    Rule-based risk signal extraction.
    These signals give the ExplainerAgent and DecisionAgent
    human-readable context about WHY this transaction is suspicious.
    """
    signals = []

    amt = transaction.get("amt", 0)
    if amt > 500:
        signals.append(f"High transaction amount: ${amt:.2f} (above $500 threshold)")
    if amt > 1000:
        signals.append(f"Very high transaction amount: ${amt:.2f} (above $1000 threshold)")

    hour = transaction.get("hour", 12)
    if hour >= 22 or hour <= 3:
        signals.append(f"Late night transaction: {int(hour)}:00 (peak fraud window 22:00-03:00)")

    geo = transaction.get("geo_distance", 0)
    if geo > 100:
        signals.append(f"Large geographic distance: {geo:.1f}km from cardholder home")
    if geo > 500:
        signals.append(f"Extreme geographic distance: {geo:.1f}km — possible card-not-present fraud")

    age = transaction.get("age", 40)
    if age > 70:
        signals.append(f"Elderly cardholder (age {age}) — higher vulnerability profile")

    is_night = transaction.get("is_night", 0)
    is_weekend = transaction.get("is_weekend", 0)
    if is_night and is_weekend:
        signals.append("Late night weekend transaction — elevated risk window")

    if fraud_score >= 0.8:
        signals.append(f"Model confidence very high: {fraud_score:.1%} fraud probability")
    elif fraud_score >= 0.5:
        signals.append(f"Model flagged: {fraud_score:.1%} fraud probability")

    return signals


@mlflow.trace(name="investigator_agent", span_type="AGENT")
def investigator_node(state: FraudInvestigationState) -> FraudInvestigationState:
    """
    LangGraph node — Investigator Agent.
    Reads: state['transaction']
    Writes: fraud_score, risk_level, risk_signals, *_context fields
    """
    logger.info(f"[Investigator] Processing transaction: {state['transaction_id']}")

    try:
        txn     = state["transaction"]
        model, scaler = load_artifacts()

        # Build feature vector — must match preprocessing order
        feature_cols = [
            'merchant', 'category', 'amt', 'gender', 'city', 'state',
            'zip', 'city_pop', 'job', 'hour', 'day_of_week', 'month',
            'is_weekend', 'is_night', 'age', 'geo_distance'
        ]
        feat_values = {col: txn.get(col, 0) for col in feature_cols}
        X = pd.DataFrame([feat_values])[feature_cols]
        X_scaled = scaler.transform(X)

        # Fraud probability
        fraud_score = float(model.predict_proba(X_scaled)[0, 1])
        risk_level  = "HIGH" if fraud_score >= 0.8 else \
                      "MEDIUM" if fraud_score >= 0.5 else "LOW"

        # Risk signals
        risk_signals = _extract_risk_signals(txn, fraud_score)

        # Context blocks for explainer and decision agents
        merchant_context = {
            "category"   : txn.get("category_raw", "unknown"),
            "merchant"   : txn.get("merchant_raw", "unknown"),
            "amount"     : txn.get("amt", 0),
        }
        cardholder_context = {
            "age"        : txn.get("age", 0),
            "gender"     : txn.get("gender_raw", "unknown"),
            "city"       : txn.get("city_raw", "unknown"),
            "state"      : txn.get("state_raw", "unknown"),
        }
        time_context = {
            "hour"       : txn.get("hour", 0),
            "day_of_week": txn.get("day_of_week", 0),
            "is_night"   : bool(txn.get("is_night", 0)),
            "is_weekend" : bool(txn.get("is_weekend", 0)),
        }
        geo_context = {
            "geo_distance_km" : round(txn.get("geo_distance", 0), 2),
            "city_population" : txn.get("city_pop", 0),
        }

        logger.info(f"[Investigator] Score={fraud_score:.4f} | "
                    f"Risk={risk_level} | Signals={len(risk_signals)}")

        return {
            **state,
            "fraud_score"       : round(fraud_score, 4),
            "risk_level"        : risk_level,
            "risk_signals"      : risk_signals,
            "merchant_context"  : merchant_context,
            "cardholder_context": cardholder_context,
            "time_context"      : time_context,
            "geo_context"       : geo_context,
        }

    except Exception as e:
        logger.error(f"[Investigator] Error: {e}")
        return {**state, "error": str(e)}
