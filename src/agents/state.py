"""
src/agents/state.py

Shared state schema for the LangGraph fraud investigation pipeline.

This is the single object that flows between all three agents.
Each agent reads from it and adds its outputs back into it.
LangGraph manages the state transitions between nodes.

Design principle:
  Every field is Optional — agents populate fields progressively.
  The orchestrator can inspect state at any point to check progress.
"""

from typing import Optional, TypedDict


class FraudInvestigationState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph pipeline.

    Populated progressively:
      After InvestigatorAgent : transaction, fraud_score, risk_signals, context
      After ExplainerAgent    : top_features, explanation
      After DecisionAgent     : verdict, confidence, case_report
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    transaction_id    : str
    transaction       : dict           # raw transaction fields

    # ── Model output ──────────────────────────────────────────────────────────
    fraud_score       : Optional[float]   # XGBoost fraud probability 0-1
    risk_level        : Optional[str]     # HIGH / MEDIUM / LOW

    # ── Investigator outputs ───────────────────────────────────────────────────
    risk_signals      : Optional[list]    # list of flagged anomaly strings
    merchant_context  : Optional[dict]    # merchant category, frequency
    cardholder_context: Optional[dict]    # age, typical spend, geo pattern
    time_context      : Optional[dict]    # hour, is_night, day_of_week
    geo_context       : Optional[dict]    # distance from home, city

    # ── Explainer outputs ──────────────────────────────────────────────────────
    top_features      : Optional[list]    # top N features driving prediction
    explanation       : Optional[str]     # Claude-generated narrative

    # ── Decision outputs ───────────────────────────────────────────────────────
    verdict           : Optional[str]     # BLOCK / REVIEW / ALLOW
    confidence        : Optional[float]   # 0-1 confidence in verdict
    case_report       : Optional[dict]    # full structured report

    # ── Metadata ───────────────────────────────────────────────────────────────
    error             : Optional[str]     # error message if any agent fails
    trace_id          : Optional[str]     # MLflow trace ID for auditability
