"""
Investigation Agent — Phase 2 placeholder.
Given a flagged transaction, this agent will:
  1. Pull contextual signals (merchant history, time anomaly, geo distance)
  2. Reason via Claude API about whether the flag is genuine
  3. Return a structured case report

Will be implemented with LangGraph in Phase 5.
"""

class InvestigatorAgent:
    def __init__(self, model, scaler):
        self.model  = model
        self.scaler = scaler

    def investigate(self, transaction: dict) -> dict:
        raise NotImplementedError("Agent implementation in Phase 5.")
