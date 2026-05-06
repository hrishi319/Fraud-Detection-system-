"""
src/agents/orchestrator.py

LangGraph Orchestrator — defines the fraud investigation graph.

Graph structure:
  START → investigator → explainer → decision → END

State flows through each node sequentially.
Each node reads the full state and returns an updated copy.

MLflow integration:
  - Experiment: "fraud-detection-capstone" (same as training)
  - Each investigation run is a new MLflow run
  - Each agent node is traced as a span within that run
  - Full audit trail: transaction → agents → verdict → report
"""

import json
import mlflow
from langgraph.graph import StateGraph, START, END
from src.agents.state import FraudInvestigationState
from src.agents.investigator import investigator_node
from src.agents.explainer import explainer_node
from src.agents.decision import decision_node
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "fraud-detection-capstone"


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph investigation graph.

    Graph: START → investigator → explainer → decision → END
    """
    graph = StateGraph(FraudInvestigationState)

    # Add nodes
    graph.add_node("investigator", investigator_node)
    graph.add_node("explainer",    explainer_node)
    graph.add_node("decision",     decision_node)

    # Define edges — sequential pipeline
    graph.add_edge(START,          "investigator")
    graph.add_edge("investigator", "explainer")
    graph.add_edge("explainer",    "decision")
    graph.add_edge("decision",     END)

    return graph.compile()


class FraudInvestigationOrchestrator:
    """
    High-level interface for running fraud investigations.
    Wraps the LangGraph graph with MLflow experiment tracking.
    """

    def __init__(self):
        self.graph = build_graph()
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("FraudInvestigationOrchestrator initialised.")

    def investigate(self,
                    transaction_id: str,
                    transaction: dict) -> dict:
        """
        Run a full fraud investigation on a single transaction.

        Parameters
        ----------
        transaction_id : unique identifier for the transaction
        transaction    : preprocessed feature dict (post-preprocessing.py)

        Returns
        -------
        case_report dict with verdict, confidence, explanation, signals
        """
        logger.info(f"\nStarting investigation: {transaction_id}")

        # Initial state
        initial_state: FraudInvestigationState = {
            "transaction_id"    : transaction_id,
            "transaction"       : transaction,
            "fraud_score"       : None,
            "risk_level"        : None,
            "risk_signals"      : None,
            "merchant_context"  : None,
            "cardholder_context": None,
            "time_context"      : None,
            "geo_context"       : None,
            "top_features"      : None,
            "explanation"       : None,
            "verdict"           : None,
            "confidence"        : None,
            "case_report"       : None,
            "error"             : None,
            "trace_id"          : None,
        }

        # Run inside MLflow run for full traceability
        with mlflow.start_run(run_name=f"investigation_{transaction_id}"):

            # Log transaction metadata
            mlflow.log_param("transaction_id", transaction_id)
            mlflow.log_param("amount",         transaction.get("amt", 0))
            mlflow.log_param("hour",           transaction.get("hour", 0))

            # Run LangGraph pipeline
            final_state = self.graph.invoke(initial_state)

            # Log verdict metrics
            mlflow.log_metric("fraud_score", final_state.get("fraud_score", 0))
            mlflow.log_metric("confidence",  final_state.get("confidence",  0))
            mlflow.log_param("verdict",      final_state.get("verdict",     "UNKNOWN"))
            mlflow.log_param("risk_level",   final_state.get("risk_level",  "UNKNOWN"))

            # Log case report as artifact
            report = final_state.get("case_report", {})
            with open("/tmp/case_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            mlflow.log_artifact("/tmp/case_report.json")

            logger.info(f"Investigation complete: {final_state.get('verdict')} "
                        f"(confidence={final_state.get('confidence')})")

        return report
