"""
src/agents/decision.py

Decision Agent — Node 3 in the LangGraph pipeline.

Responsibilities:
  1. Read all state populated by Investigator and Explainer
  2. Apply decision logic to produce a final verdict
  3. Assemble the complete case report

Verdict logic:
  BLOCK  — fraud_score >= 0.8 AND 2+ HIGH risk signals
  REVIEW — fraud_score >= 0.5 OR 1+ risk signals
  ALLOW  — fraud_score < 0.5 AND no risk signals

This agent deliberately does NOT call the LLM for the verdict.
The verdict is deterministic rule-based logic for auditability.
The LLM's role is explanation (ExplainerAgent), not decision-making.
This is an important design principle: LLMs should explain, not decide
in high-stakes financial contexts.

MLflow Traces: wraps with span_type="AGENT" for pipeline tracing.
"""

import mlflow
from datetime import datetime
from src.agents.state import FraudInvestigationState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_verdict(fraud_score: float,
                     risk_signals: list,
                     risk_level: str) -> tuple[str, float]:
    """
    Deterministic verdict logic.

    Returns (verdict, confidence).

    Design rationale:
      LLMs should not make final fraud decisions — they hallucinate,
      lack consistency, and are not auditable in regulatory contexts.
      Rule-based decisions are transparent, reproducible, and defensible.
    """
    n_signals = len(risk_signals) if risk_signals else 0

    if fraud_score >= 0.8 and n_signals >= 2:
        return "BLOCK",  round(fraud_score, 4)

    elif fraud_score >= 0.8 and n_signals < 2:
        return "REVIEW", round(fraud_score * 0.9, 4)

    elif fraud_score >= 0.5:
        return "REVIEW", round(fraud_score, 4)

    elif fraud_score >= 0.3 and n_signals >= 1:
        return "REVIEW", round(0.4 + (n_signals * 0.05), 4)

    else:
        return "ALLOW",  round(1.0 - fraud_score, 4)


@mlflow.trace(name="decision_agent", span_type="AGENT")
def decision_node(state: FraudInvestigationState) -> FraudInvestigationState:
    """
    LangGraph node — Decision Agent.
    Reads: all state fields
    Writes: verdict, confidence, case_report
    """
    logger.info(f"[Decision] Producing verdict for: {state['transaction_id']}")

    if state.get("error") and not state.get("explanation"):
        logger.warning("[Decision] Processing with partial state due to upstream error")

    fraud_score  = state.get("fraud_score",  0.0)
    risk_signals = state.get("risk_signals", [])
    risk_level   = state.get("risk_level",   "LOW")

    verdict, confidence = _compute_verdict(fraud_score, risk_signals, risk_level)

    # Assemble complete case report
    case_report = {
        "transaction_id"    : state["transaction_id"],
        "timestamp"         : datetime.now().isoformat(),
        "verdict"           : verdict,
        "confidence"        : confidence,
        "fraud_score"       : fraud_score,
        "risk_level"        : risk_level,
        "risk_signals"      : risk_signals,
        "top_features"      : state.get("top_features",      []),
        "explanation"       : state.get("explanation",       ""),
        "merchant_context"  : state.get("merchant_context",  {}),
        "cardholder_context": state.get("cardholder_context",{}),
        "time_context"      : state.get("time_context",      {}),
        "geo_context"       : state.get("geo_context",       {}),
        "transaction"       : state.get("transaction",       {}),
    }

    logger.info(f"[Decision] Verdict={verdict} | Confidence={confidence:.4f}")

    return {
        **state,
        "verdict"    : verdict,
        "confidence" : confidence,
        "case_report": case_report,
    }
