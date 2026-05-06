"""
src/agents/explainer.py

Explainer Agent — Node 2 in the LangGraph pipeline.

Responsibilities:
  1. Extract top features driving the XGBoost prediction
     (using native feature_importances_ — no shap dependency)
  2. Call Claude API to generate a plain-English investigation narrative
  3. Populate state with top_features and explanation

This agent is the LLM-powered core of the system.
The narrative it generates is what makes this an "agentic" system —
not just a classifier output, but a reasoned explanation a human can act on.

MLflow Traces: LLM calls are traced with span_type="LLM" for token tracking.
"""

import os
import json
import mlflow
import numpy as np
import pandas as pd
import joblib
import anthropic
from src.agents.state import FraudInvestigationState
from src.utils.config import MODELS_DIR, ANTHROPIC_API_KEY
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Top N features to include in explanation
TOP_N_FEATURES = 5

# ── Mode config — set in .env ──────────────────────────────────────────────────
# MOCK_EXPLAINER=true  → skip API call, return placeholder (development/testing)
# MOCK_EXPLAINER=false → call Claude API (demo/production)
MOCK_MODE = os.getenv("MOCK_EXPLAINER", "false").lower() == "true"

# MODEL — switch to haiku for cheaper testing, sonnet for final demo
# claude-haiku-4-5-20251001  → ~20x cheaper, good for testing
# claude-sonnet-4-20250514   → higher quality, use for final demo
MODEL = os.getenv("EXPLAINER_MODEL", "claude-haiku-4-5-20251001")


def _get_top_features(transaction: dict, n: int = TOP_N_FEATURES) -> list:
    """
    Extract top N features driving the XGBoost prediction.
    Uses native XGBoost feature_importances_ — no shap required.

    Why not shap:
      shap requires numpy>=2 which conflicts with langchain's numpy<2 requirement.
      XGBoost native importance is sufficient for generating meaningful narratives.
    """
    model = joblib.load(MODELS_DIR / "xgboost_model.pkl")

    feature_cols = [
        'merchant', 'category', 'amt', 'gender', 'city', 'state',
        'zip', 'city_pop', 'job', 'hour', 'day_of_week', 'month',
        'is_weekend', 'is_night', 'age', 'geo_distance'
    ]

    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1], reverse=True
    )[:n]

    # Enrich with actual transaction values
    return [
        {
            "feature"   : feat,
            "importance": round(float(imp), 4),
            "value"     : transaction.get(feat, "N/A")
        }
        for feat, imp in feat_imp
    ]


def _build_investigation_prompt(state: FraudInvestigationState) -> str:
    """
    Build the prompt for Claude API.
    Structured to produce a concise, actionable investigation narrative.
    """
    txn     = state["transaction"]
    signals = state.get("risk_signals", [])
    top_f   = state.get("top_features", [])
    merch   = state.get("merchant_context", {})
    card    = state.get("cardholder_context", {})
    time    = state.get("time_context", {})
    geo     = state.get("geo_context", {})
    score   = state.get("fraud_score", 0)
    risk    = state.get("risk_level", "UNKNOWN")

    features_str = "\n".join([
        f"  - {f['feature']}: {f['value']} (importance: {f['importance']})"
        for f in top_f
    ])

    signals_str = "\n".join([f"  - {s}" for s in signals]) if signals \
        else "  - No specific signals flagged"

    prompt = f"""You are a senior fraud analyst at a financial institution.
A machine learning model has flagged the following transaction for review.
Your job is to write a concise investigation narrative for the fraud review team.

TRANSACTION DETAILS:
  Transaction ID : {state['transaction_id']}
  Amount         : ${txn.get('amt', 0):.2f}
  Merchant       : {merch.get('merchant', 'unknown')}
  Category       : {merch.get('category', 'unknown')}
  Time           : {int(time.get('hour', 0)):02d}:00 | Night: {time.get('is_night')} | Weekend: {time.get('is_weekend')}
  Cardholder     : Age {card.get('age', 'unknown')} | {card.get('gender', 'unknown')} | {card.get('city', 'unknown')}, {card.get('state', 'unknown')}
  Geo distance   : {geo.get('geo_distance_km', 0)} km from home

MODEL OUTPUT:
  Fraud probability : {score:.1%}
  Risk level        : {risk}

TOP FEATURES DRIVING THE PREDICTION:
{features_str}

RISK SIGNALS IDENTIFIED:
{signals_str}

Write a 3-4 sentence investigation narrative that:
1. Summarises the key reasons this transaction was flagged
2. Highlights the most suspicious signals in plain English
3. Notes any contextual factors that increase or decrease concern
4. Ends with a recommended action (BLOCK, REVIEW, or ALLOW)

Be concise, factual, and professional. Do not use bullet points."""

    return prompt


@mlflow.trace(name="explainer_agent", span_type="LLM")
def explainer_node(state: FraudInvestigationState) -> FraudInvestigationState:
    """
    LangGraph node — Explainer Agent.
    Reads: all investigator outputs from state
    Writes: top_features, explanation
    """
    logger.info(f"[Explainer] Generating explanation for: {state['transaction_id']}")

    if state.get("error"):
        logger.warning("[Explainer] Skipping — upstream error detected")
        return state

    try:
        txn = state["transaction"]

        # Get top features
        top_features = _get_top_features(txn)

        # Update state with features before API call
        state = {**state, "top_features": top_features}

        # Build prompt
        prompt = _build_investigation_prompt(state)

        # Mock mode — skip API call during development
        if MOCK_MODE:
            signals_summary = ', '.join(state.get('risk_signals', [])[:2]) or 'elevated risk score'
            explanation = (
                f"[MOCK MODE] Transaction {state['transaction_id']} flagged with "
                f"{state.get('fraud_score', 0):.1%} fraud probability. "
                f"Key signals: {signals_summary}. "
                f"Top feature: {state.get('top_features', [{}])[0].get('feature', 'unknown')}. "
                f"Recommended action: {state.get('risk_level', 'REVIEW')}."
            )
            logger.info("[Explainer] Mock mode — API call skipped")
        else:
            # Call Claude API
            client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model      = MODEL,
                max_tokens = 500,
                messages   = [{"role": "user", "content": prompt}]
            )
            explanation = response.content[0].text.strip()
            logger.info(f"[Explainer] Model={MODEL} | Response={len(explanation)} chars")

        return {
            **state,
            "top_features": top_features,
            "explanation" : explanation,
        }

    except Exception as e:
        logger.error(f"[Explainer] Error: {e}")
        return {
            **state,
            "top_features": top_features if 'top_features' in locals() else [],
            "explanation" : f"Explanation unavailable: {str(e)}",
            "error"       : str(e)
        }
